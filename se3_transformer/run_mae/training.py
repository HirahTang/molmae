import logging
import pathlib
from random import vonmisesvariate
from typing import List

import numpy as np
import torch
from torch import alpha_dropout
import torch.distributed as dist
import torch.nn as nn
from apex.optimizers import FusedAdam, FusedLAMB
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from rdkit import RDLogger

import sys
sys.path.append('/home/sunyuancheng/molmae')

from se3_transformer.data_loading import QM9DataModule, QM9MAEModule, OGBMAEModule
from se3_transformer.model.mae import base_MAE, egnn_MAE
from se3_transformer.model.fiber import Fiber
from se3_transformer.runtime import gpu_affinity
from se3_transformer.runtime.arguments import PARSER
from se3_transformer.runtime.callbacks import PretrainLossCallback, QM9MetricCallback, QM9LRSchedulerCallback, BaseCallback, \
    PerformanceCallback
from se3_transformer.runtime.loggers import LoggerCollection, DLLogger, WandbLogger, Logger
from se3_transformer.runtime.utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity, get_loss_function
from se3_transformer.runtime.training import load_state, save_state
from se3_transformer.runtime.utils import str2bool
from se3_transformer.run_mae.utils import von_Mises_loss


def print_parameters_count(model):
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_params_trainable}')


def train(model: nn.Module,
          loss_fn: _Loss,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          callbacks: List[BaseCallback],
          logger: Logger,
          args):
    device = torch.cuda.current_device()
    model.to(device=device)
    local_rank = get_local_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        model._set_static_graph()

    model.train()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    if args.optimizer == 'adam':
        optimizer = FusedAdam(model.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999),
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'lamb':
        optimizer = FusedLAMB(model.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999),
                              weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    epoch_start = load_state(model, optimizer, args.load_ckpt_path, callbacks) if args.load_ckpt_path else 0

    for callback in callbacks:
        callback.on_fit_start(optimizer, args)

    for epoch_idx in range(epoch_start, args.epochs):
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch_idx)

        loss_dict = train_epoch(model, train_dataloader, loss_fn, epoch_idx, grad_scaler, optimizer, local_rank, callbacks,
                           args)
        if dist.is_initialized():
            for k, v in loss_dict.items():
                v = torch.tensor(v, dtype=torch.float, device=device)
                torch.distributed.all_reduce(v)
                loss_dict[k] = (v / world_size).item()

        for k, v in loss_dict.items():
            logging.info(f'{k}: {v}')
            logger.log_metrics({k: v}, epoch_idx)

        if epoch_idx + 1 == args.epochs:
            logger.log_metrics({'train loss': loss_dict['train_loss']})

        for callback in callbacks:
            callback.on_epoch_end()

        if not args.benchmark and args.save_ckpt_path is not None and args.ckpt_interval > 0 \
                and (epoch_idx + 1) % args.ckpt_interval == 0:
            save_state(model, optimizer, epoch_idx, args.save_ckpt_path, callbacks)

        if not args.benchmark and (
                (args.eval_interval > 0 and (epoch_idx + 1) % args.eval_interval == 0) or epoch_idx + 1 == args.epochs):
            validate(model, val_dataloader, callbacks, args)
            model.train()

            for callback in callbacks:
                callback.on_validation_end(epoch_idx)

    if args.save_ckpt_path is not None and not args.benchmark:
        save_state(model, optimizer, args.epochs, args.save_ckpt_path, callbacks)

    for callback in callbacks:
        callback.on_fit_end()


def train_epoch(model, train_dataloader, loss_fn, epoch_idx, grad_scaler, optimizer, local_rank, callbacks, args):
    loss_logger = []
    length_loss_logger = []
    angle_loss_logger = []
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit='batch',
                         desc=f'Epoch {epoch_idx}', disable=(args.silent or local_rank != 0)):
        # *inputs, target = to_cuda(batch)

        *inputs, _, pretrain_labels = to_cuda(batch)
        batched_graph = inputs[0]

        for callback in callbacks:
            callback.on_batch_start()

        with torch.cuda.amp.autocast(enabled=args.amp):
            # bond_length_pred, bond_angle_pred, radius_pred, orientaion_pred
            preds = model(*inputs, pretrain_labels=pretrain_labels)
            res_mask = pretrain_labels['batch_neighbor_masks'][~batched_graph.ndata['node_mask'].bool().cpu()]
            length_loss = loss_fn((preds[0] * pretrain_labels['batch_neighbor_masks'].cpu())[~batched_graph.ndata['node_mask'].bool().cpu()],
                                  (pretrain_labels['bond_length'].cpu() * pretrain_labels['batch_neighbor_masks'].cpu())[~batched_graph.ndata['node_mask'].bool().cpu()])
            angle_loss = torch.sum(von_Mises_loss((preds[1] * pretrain_labels['batch_angle_masks'].cpu())[~batched_graph.ndata['node_mask'].bool().cpu()],   \
                                    (pretrain_labels['bond_angle'].cpu() * pretrain_labels['batch_angle_masks'].cpu())[~batched_graph.ndata['node_mask'].bool().cpu()])) \
                                        /((~batched_graph.ndata['node_mask'].bool()).sum(dim=-1) + 1e-10)
            # radius_loss = loss_fn(preds[2]*batched_graph.ndata['node_mask'])
            # orientation_loss = loss_fn(preds[3]*batched_graph.ndata['node_mask'])
            loss = (length_loss - 0.5 * angle_loss) / args.accumulate_grad_batches

        grad_scaler.scale(loss).backward()

        # gradient accumulation
        if (i + 1) % args.accumulate_grad_batches == 0 or (i + 1) == len(train_dataloader):
            if args.gradient_clip:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            grad_scaler.step(optimizer)
            grad_scaler.update()
            model.zero_grad(set_to_none=True)

        loss_logger.append(loss.item())
        length_loss_logger.append(length_loss.item())
        angle_loss_logger.append(angle_loss.item())

    return {"train_loss":np.mean(loss_logger), 
            "length_loss":np.mean(length_loss_logger), 
            "angle_loss":np.mean(angle_loss_logger)}


@torch.inference_mode()
def validate(model: nn.Module,
             dataloader: DataLoader,
             callbacks: List[BaseCallback],
             args):
    # different from the evaluate function
    # this is for the validation of the reconstruction
    model.eval()
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), unit='batch', desc=f'Evaluation',
                         leave=False, disable=(args.silent or get_local_rank() != 0)):
        *inputs, _, pretrain_labels = to_cuda(batch)
        batched_graph = inputs[0]

        for callback in callbacks:
            callback.on_batch_start()

        with torch.cuda.amp.autocast(enabled=args.amp):
            preds = model(*inputs, pretrain_labels=pretrain_labels)
            length_loss = loss_fn(preds[0][~batched_graph.ndata['node_mask'].bool().cpu()], pretrain_labels['bond_length'].cpu()[~batched_graph.ndata['node_mask'].bool().cpu()])
            angle_loss = torch.sum(von_Mises_loss(preds[1][~batched_graph.ndata['node_mask'].bool().cpu()], pretrain_labels['bond_angle'].cpu()[~batched_graph.ndata['node_mask'].bool().cpu()])) \
                / ((~batched_graph.ndata['node_mask'].bool()).sum(dim=-1) + 1e-10)
            # radius_loss = loss_fn(preds[2]*batched_graph.ndata['node_mask'])
            # orientation_loss = loss_fn(preds[2]*batched_graph.ndata['node_mask'])
            loss = (length_loss - angle_loss) / args.accumulate_grad_batches

            for callback in callbacks:
                callback.on_validation_step(loss)




if __name__ == '__main__':
    is_distributed = init_distributed()
    local_rank = get_local_rank()
    args = PARSER.parse_args()

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL if local_rank != 0 or args.silent else logging.INFO)

    logging.info('========== SE(3)-MAE ==========')
    logging.info('|      Training procedure     |')
    logging.info('===============================')

    if args.seed is not None:
        logging.info(f'Using seed {args.seed}')
        seed_everything(args.seed)

    loggers = [DLLogger(save_dir=args.log_dir, filename=args.dllogger_name)]
    if args.wandb:
        loggers.append(WandbLogger(name=args.exp_name, save_dir=args.log_dir, project='se3-eva'))
    logger = LoggerCollection(loggers)

    # datamodule = QM9MAEModule(**vars(args))
    datamodule = OGBMAEModule(**vars(args))

    if args.encoder_type == 'se3':
        model = base_MAE(
            fiber_in=Fiber({0: datamodule.NODE_FEATURE_DIM}),
            fiber_hidden=Fiber.create(args.num_degrees, args.num_channels),
            # fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
            fiber_out=Fiber.create(2, args.num_degrees * args.num_channels),
            fiber_edge=Fiber({0: datamodule.EDGE_FEATURE_DIM}),
            # output_dim=datamodule.NODE_FEATURE_DIM+3,
            # output_dim=3,
            tensor_cores=using_tensor_cores(args.amp),  # use Tensor Cores more effectively
            # return_type=0,
            **vars(args)
        )
    elif args.encoder_type == 'egnn':
        model = egnn_MAE(
            num_node_features = datamodule.NODE_FEATURE_DIM,
            num_edge_features = datamodule.EDGE_FEATURE_DIM,
            **vars(args),
        )
    loss_fn = get_loss_function(args.loss_type)

    # for downstream test
    # if args.evaluate and get_local_rank() == 0:
    #     downstream_callbacks = [QM9MetricCallback(logger, targets_std=datamodule.targets_std, prefix='test')]
    #     evaluate(model,
    #          args.pooling,
    #          datamodule.train_dataloader(),
    #          downstream_callbacks,
    #          args)
    #     model.train()
    #     # for callback in downstream_callbacks:
    #     #     callback.on_validation_end()
    if get_local_rank() == 0:
        print(model)

    if args.benchmark:
        logging.info('Running benchmark mode')
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        callbacks = [PerformanceCallback(logger, args.batch_size * world_size)]
    else:
        callbacks = [PretrainLossCallback(logger, prefix='validation'),
                     QM9LRSchedulerCallback(logger, epochs=args.epochs)]

    if is_distributed:
        gpu_affinity.set_affinity(gpu_id=get_local_rank(), nproc_per_node=torch.cuda.device_count())

    print_parameters_count(model)
    logger.log_hyperparams(vars(args))
    increase_l2_fetch_granularity()
    train(model,
          loss_fn,
          datamodule.train_dataloader(),
          datamodule.val_dataloader(),
          callbacks,
          logger,
          args)

    # if args.evaluate and get_local_rank == 0:
    #     downstream_callbacks = [QM9MetricCallback(logger, targets_std=datamodule.targets_std, prefix='test')]
    #     evaluate(model,
    #          datamodule.train_dataloader(),
    #          downstream_callbacks,
    #          args)
    #     # for callback in downstream_callbacks:
    #     #     callback.on_validation_end()
    
    logging.info('Training finished successfully')