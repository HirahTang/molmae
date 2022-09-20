# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT
from ast import Assign
from typing import Tuple
from cv2 import split

import dgl
import pathlib
import torch
from dgl.data import QM9EdgeDataset
from dgl import DGLGraph
from torch import Tensor
from torch.utils.data import random_split, DataLoader, Dataset
from tqdm import tqdm
import copy
import math
import numpy as np
import random

from se3_transformer.data_loading.data_module import DataModule
from se3_transformer.model.basis import get_basis
from se3_transformer.runtime.utils import get_local_rank, str2bool, using_tensor_cores

from se3_transformer.run_mae.utils import assign_neighborhoods, ground_truth_local_stats


# allowable multiple choice node and edge features 
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['[MASK]', 'misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        '[MASK]',
        '[SELF]',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
        '[MASK]',
    ],
    'possible_is_conjugated_list': [False, True],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def _get_relative_pos(qm9_graph: DGLGraph) -> Tensor:
    x = qm9_graph.ndata['pos']
    src, dst = qm9_graph.edges()
    rel_pos = x[dst] - x[src]
    return rel_pos


def _get_split_sizes(full_dataset: Dataset, split_size_type: str) -> Tuple[int, int, int]:
    len_full = len(full_dataset)
    if split_size_type == 'pretrain':
        len_val = int(0.1 * len_full)
        len_train = len_full - len_val
        # return len_full-len_val, len_val, 0
        return len_train, len_val, 0
    elif split_size_type == 'finetune':
        len_train = 100_000
        len_test = int(0.1 * len_full)
        len_val = len_full - len_train - len_test
        return len_train, len_val, len_test
    else:
        raise NotImplementedError


class QM9DataModule(DataModule):
    """
    Datamodule wrapping https://docs.dgl.ai/en/latest/api/python/dgl.data.html#qm9edge-dataset
    Training set is 100k molecules. Test set is 10% of the dataset. Validation set is the rest.
    This includes all the molecules from QM9 except the ones that are uncharacterized.
    """

    NODE_FEATURE_DIM = 6
    EDGE_FEATURE_DIM = 4

    def __init__(self,
                 data_dir: pathlib.Path,
                 task: str = 'homo',
                 batch_size: int = 240,
                 num_workers: int = 8,
                 num_degrees: int = 4,
                 amp: bool = False,
                 precompute_bases: bool = False,
                 **kwargs):
        self.data_dir = data_dir  # This needs to be before __init__ so that prepare_data has access to it
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=self._collate)
        self.amp = amp
        self.task = task
        self.batch_size = batch_size
        self.num_degrees = num_degrees

        qm9_kwargs = dict(label_keys=[self.task], verbose=False, raw_dir=str(data_dir))
        if precompute_bases:
            bases_kwargs = dict(max_degree=num_degrees - 1, use_pad_trick=using_tensor_cores(amp), amp=amp)
            full_dataset = CachedBasesQM9EdgeDataset(bases_kwargs=bases_kwargs, batch_size=batch_size,
                                                     num_workers=num_workers, **qm9_kwargs)
        else:
            full_dataset = QM9EdgeDataset(**qm9_kwargs)

        self.ds_train, self.ds_val, self.ds_test = random_split(full_dataset, _get_split_sizes(full_dataset, kwargs['split_size_type']),
                                                                generator=torch.Generator().manual_seed(0))

        train_targets = full_dataset.targets[self.ds_train.indices, full_dataset.label_keys[0]]
        self.targets_mean = train_targets.mean()
        self.targets_std = train_targets.std()
        self.full_dataset = full_dataset

    def prepare_data(self):
        # Download the QM9 preprocessed data
        QM9EdgeDataset(verbose=True, raw_dir=str(self.data_dir))

    def _collate(self, samples):
        graphs, y, *bases = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        edge_feats = {'0': batched_graph.edata['edge_attr'][..., None]}
        batched_graph.edata['rel_pos'] = _get_relative_pos(batched_graph)
        # get node features
        node_feats = {'0': batched_graph.ndata['attr'][:, :6, None]}
        targets = (torch.cat(y) - self.targets_mean) / self.targets_std

        if bases:
            # collate bases
            all_bases = {
                key: torch.cat([b[key] for b in bases[0]], dim=0)
                for key in bases[0][0].keys()
            }

            return batched_graph, node_feats, edge_feats, all_bases, targets
        else:
            return batched_graph, node_feats, edge_feats, targets

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("QM9 dataset")
        parser.add_argument('--task', type=str, default='homo', const='homo', nargs='?',
                            choices=['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv',
                                     'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C'],
                            help='Regression task to train on')
        parser.add_argument('--precompute_bases', type=str2bool, nargs='?', const=False, default=False,
                            help='Precompute bases at the beginning of the script during dataset initialization,'
                                 ' instead of computing them at the beginning of each forward pass.')
        return parent_parser

    def __repr__(self):
        return f'QM9({self.task})'


class CachedBasesQM9EdgeDataset(QM9EdgeDataset):
    """ Dataset extending the QM9 dataset from DGL with precomputed (cached in RAM) pairwise bases """

    def __init__(self, bases_kwargs: dict, batch_size: int, num_workers: int, *args, **kwargs):
        """
        :param bases_kwargs:  Arguments to feed the bases computation function
        :param batch_size:    Batch size to use when iterating over the dataset for computing bases
        """
        self.bases_kwargs = bases_kwargs
        self.batch_size = batch_size
        self.bases = None
        self.num_workers = num_workers
        super().__init__(*args, **kwargs)

    def load(self):
        super().load()
        # Iterate through the dataset and compute bases (pairwise only)
        # Potential improvement: use multi-GPU and gather
        dataloader = DataLoader(self, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers,
                                collate_fn=lambda samples: dgl.batch([sample[0] for sample in samples]))
        bases = []
        for i, graph in tqdm(enumerate(dataloader), total=len(dataloader), desc='Precomputing QM9 bases',
                             disable=get_local_rank() != 0):
            rel_pos = _get_relative_pos(graph)
            # Compute the bases with the GPU but convert the result to CPU to store in RAM
            bases.append({k: v.cpu() for k, v in get_basis(rel_pos.cuda(), **self.bases_kwargs).items()})
        self.bases = bases  # Assign at the end so that __getitem__ isn't confused

    def __getitem__(self, idx: int):
        graph, label = super().__getitem__(idx)

        if self.bases:
            bases_idx = idx // self.batch_size
            bases_cumsum_idx = self.ne_cumsum[idx] - self.ne_cumsum[bases_idx * self.batch_size]
            bases_cumsum_next_idx = self.ne_cumsum[idx + 1] - self.ne_cumsum[bases_idx * self.batch_size]
            return graph, label, {key: basis[bases_cumsum_idx:bases_cumsum_next_idx] for key, basis in
                                  self.bases[bases_idx].items()}
        else:
            return graph, label




class QM9MAEModule(DataModule):
    """
    Datamodule wrapping https://docs.dgl.ai/en/latest/api/python/dgl.data.html#qm9edge-dataset
    Training set is 100k molecules. Test set is 10% of the dataset. Validation set is the rest.
    This includes all the molecules from QM9 except the ones that are uncharacterized.
    Specify the collate module to generate labels for pretraining.
    """

    NODE_FEATURE_DIM = 6
    EDGE_FEATURE_DIM = 4

    def __init__(self,
                 data_dir: pathlib.Path,
                 task: str = 'homo',
                 batch_size: int = 240,
                 num_workers: int = 8,
                 num_degrees: int = 4,
                 amp: bool = False,
                 precompute_bases: bool = False,
                 **kwargs):
        self.data_dir = data_dir  # This needs to be before __init__ so that prepare_data has access to it
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=self._collate)
        self.amp = amp
        self.task = task
        self.batch_size = batch_size
        self.num_degrees = num_degrees

        qm9_kwargs = dict(label_keys=[self.task], verbose=False, raw_dir=str(data_dir))
        if precompute_bases:
            bases_kwargs = dict(max_degree=num_degrees - 1, use_pad_trick=using_tensor_cores(amp), amp=amp)
            full_dataset = CachedBasesQM9EdgeDataset(bases_kwargs=bases_kwargs, batch_size=batch_size,
                                                     num_workers=num_workers, **qm9_kwargs)
        else:
            full_dataset = QM9EdgeDataset(**qm9_kwargs)

        self.ds_train, self.ds_val, self.ds_test = random_split(full_dataset, _get_split_sizes(full_dataset, kwargs['split_size_type']),
                                                                generator=torch.Generator().manual_seed(0))

        train_targets = full_dataset.targets[self.ds_train.indices, full_dataset.label_keys[0]]
        self.targets_mean = train_targets.mean()
        self.targets_std = train_targets.std()
        self.full_dataset = full_dataset
        self.prepare_pretrain = kwargs['prepare_pretrain']
        self.mask_type = kwargs['mask_type']
        self.mask_ratio = kwargs['mask_ratio']
        self.mask_all = kwargs['mask_all']


    def prepare_data(self):
        # Download the QM9 preprocessed data
        QM9EdgeDataset(verbose=True, raw_dir=str(self.data_dir))

    def _collate(self, samples):
        graphs, y, *bases = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        edge_feats = {'0': batched_graph.edata['edge_attr'][..., None]}
        # put after mask
        # batched_graph.edata['rel_pos'] = _get_relative_pos(batched_graph)
        # get node features
        node_feats = {'0': batched_graph.ndata['attr'][:, :6, None]}
        targets = (torch.cat(y) - self.targets_mean) / self.targets_std
        
        if self.prepare_pretrain:
            # calculate pretrain labels
            # get neighbors for each node, then calculate corresponding bond length and bond angle
            neighbors, neighbor_masks, batch_neighbor_masks, neighborhood_to_mol_map = assign_neighborhoods(batched_graph)

            bond_length, bond_angle, angle_mask = ground_truth_local_stats(batched_graph.ndata['pos'], neighbors, neighbor_masks)
            pretrain_labels = {
                'neighbors': neighbors,
                'bond_length': bond_length,
                'bond_angle': bond_angle,
                'neighbor_masks': neighbor_masks,
                'batch_neighbor_masks': batch_neighbor_masks,
                'batch_angle_masks': angle_mask,
                'neighborhood_to_mol_map': neighborhood_to_mol_map,
            }

            node_feats, edge_feats = self.mask_features(batched_graph, node_feats, edge_feats, pretrain_labels)
            batched_graph.edata['rel_pos'] = _get_relative_pos(batched_graph)
            # the orientation and the radius for each node 
        if bases:
            # collate bases
            all_bases = {
                key: torch.cat([b[key] for b in bases[0]], dim=0)
                for key in bases[0][0].keys()
            }
        
            return batched_graph, node_feats, edge_feats, all_bases, targets, pretrain_labels
        else:
            return batched_graph, node_feats, edge_feats, targets, pretrain_labels

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("QM9 dataset")
        parser.add_argument('--task', type=str, default='homo', const='homo', nargs='?',
                            choices=['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv',
                                     'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C'],
                            help='Regression task to train on')
        parser.add_argument('--precompute_bases', type=str2bool, nargs='?', default=False,
                            help='Precompute bases at the beginning of the script during dataset initialization,'
                                 ' instead of computing them at the beginning of each forward pass.')
        return parent_parser

    def __repr__(self):
        return f'QM9({self.task})'

    def mask_features(self, graph, node_feats, edge_feats, pretrain_labels):
        # basis_c = copy.deepcopy(basis)
        node_feats_c = copy.deepcopy(node_feats)
        edge_feats_c = copy.deepcopy(edge_feats)

        if self.mask_type == 'rand':
            # node_mask_booled, edge_mask_booled = self._mask_pos_rand(graph, num_node, num_edge)
            self._mask_pos_rand(graph, pretrain_labels)
        elif self.mask_type == 'block':
            node_mask_booled, edge_mask_booled = self._mask_pos_block(graph)
        else:
            raise NotImplementedError

        if self.mask_all:
            # mask node features, and edge features
            # n_f = node_feats_c['0'].squeeze()
            # n_m = graph.ndata['node_mask'].unsqueeze(-1).repeat(1,6)
            # node_feats_c['0'] = torch.mul(n_f, n_m).unsqueeze(-1)
            
            # e_f = edge_feats_c['0'].squeeze()
            # e_m = graph.edata['edge_mask'].unsqueeze(-1).repeat(1,4)
            # edge_feats_c['0'] = torch.mul(e_f, e_m).unsqueeze(-1)
            node_feats_c['0'][~graph.ndata['node_mask'].bool()] = 0
            edge_feats_c['0'][~graph.edata['edge_mask'].bool()] = 0
            # drop edge
            
            graph.ndata['pos'][~graph.ndata['node_mask'].bool()] = 0

            return  node_feats_c, edge_feats_c

        else:
            # mask node features only
            n_f = node_feats_c['0'].squeeze()
            n_m = node_mask_booled.to(torch.int).unsqueeze(-1).repeat(1,6)
            node_feats_c['0'] = torch.mul(n_f, n_m).unsqueeze(-1)
            
            return  node_feats_c, edge_feats_c

    def _mask_pos_rand(self, graph, pretrain_labels, noaug=False):
        num_node = len(graph.nodes())
        num_edge = len(graph.edges()[0])
        n_atoms_per_mol = graph.batch_num_nodes()
        n_edges_per_mol = graph.batch_num_edges()
        neighbors_bin = pretrain_labels['neighborhood_to_mol_map'].bincount()
        non_terminal = torch.tensor(list(pretrain_labels['neighbors'].keys()))

        graph.ndata['node_mask'] = torch.ones(num_node, device=graph.device)
        graph.edata['edge_mask'] = torch.ones(num_edge, device=graph.device)

        if self.mask_all:
            # besides node features, mask basis and edge features as well
            n_nt_prev_mol = 0
            for n in neighbors_bin:
                n_mask = math.ceil(n * self.mask_ratio)
                perm = torch.randperm(n, device=graph.device) 
                graph.ndata['node_mask'][non_terminal[perm[:n_mask]+n_nt_prev_mol]] = 0
                n_nt_prev_mol += n

            masked_edges_id = graph.filter_edges(self.is_node_masked)
            # masked_edges = graph.find_edges(masked_edges_id)
            graph.edata['edge_mask'][masked_edges_id] = 0


        else:
            n_atoms_prev_mol = 0
            for n in n_atoms_per_mol:
                n_mask = math.ceil(n * self.mask_ratio)
                perm = torch.randperm(n, device=graph.device)
                graph.ndata['node_mask'][perm[:n_mask]+n_atoms_prev_mol] = False
                n_atoms_prev_mol += n
                
            # return node_mask_booled, edge_mask_booled


    def _mask_pos_block(self, graph, num_node, num_edge, noaug=False):
        pass


    @staticmethod
    def is_node_masked(edges):
        # print(edges.src['node_mask'] == 0)
        return torch.logical_or(edges.src['node_mask'] == 0, (edges.dst['node_mask'] == 0))



class OGBMAEModule(DataModule):

    NODE_FEATURE_DIM = 6
    EDGE_FEATURE_DIM = 3

    def __init__(self,
                 data_dir: pathlib.Path,
                 batch_size: int = 240,
                 num_workers: int = 8,
                 num_degrees: int = 4,
                 amp: bool = False,
                 precompute_bases: bool = False,
                 **kwargs):
        self.data_dir = data_dir  # This needs to be before __init__ so that prepare_data has access to it
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=self._collate)
        self.amp = amp
        self.batch_size = batch_size
        self.num_degrees = num_degrees

        full_dataset = OGBDataset(data_dir=data_dir)

        self.ds_train, self.ds_val, self.ds_test = random_split(full_dataset, _get_split_sizes(full_dataset, kwargs['split_size_type']),
                                                                generator=torch.Generator().manual_seed(0))
        
        self.full_dataset = full_dataset
        self.prepare_pretrain = kwargs['prepare_pretrain']
        self.mask_type = kwargs['mask_type']
        self.mask_ratio = kwargs['mask_ratio']
        self.mask_all = kwargs['mask_all']




    def _collate(self, samples):
        batched_graph = dgl.batch(samples)
        edge_feats = {'0': batched_graph.edata['edge_attr'][..., None]}
        # put after mask
        # batched_graph.edata['rel_pos'] = _get_relative_pos(batched_graph)
        # get node features
        node_feats = {'0': batched_graph.ndata['attr'][:, :6, None]}
        
        if self.prepare_pretrain:
            # calculate pretrain labels
            # get neighbors for each node, then calculate corresponding bond length and bond angle
            neighbors, neighbor_masks, batch_neighbor_masks, neighborhood_to_mol_map = assign_neighborhoods(batched_graph)

            bond_length, bond_angle, angle_mask = ground_truth_local_stats(batched_graph.ndata['pos'], neighbors, neighbor_masks)
            pretrain_labels = {
                'neighbors': neighbors,
                'bond_length': bond_length,
                'bond_angle': bond_angle,
                'neighbor_masks': neighbor_masks,
                'batch_neighbor_masks': batch_neighbor_masks,
                'batch_angle_masks': angle_mask,
                'neighborhood_to_mol_map': neighborhood_to_mol_map,
            }

            node_feats, edge_feats = self.mask_features(batched_graph, node_feats, edge_feats, pretrain_labels)
            batched_graph.edata['rel_pos'] = _get_relative_pos(batched_graph)
            # the orientation and the radius for each node 
        
            return batched_graph, node_feats, edge_feats, None, pretrain_labels
            

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("QM9 dataset")
        parser.add_argument('--task', type=str, default='homo', const='homo', nargs='?',
                            choices=['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv',
                                     'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C'],
                            help='Regression task to train on')
        parser.add_argument('--precompute_bases', type=str2bool, nargs='?', default=False,
                            help='Precompute bases at the beginning of the script during dataset initialization,'
                                 ' instead of computing them at the beginning of each forward pass.')

        parser.add_argument('--mask_ratio', type=float, default=0.1, help="The mask ratio of the masked auto-encoder.")
        parser.add_argument('--mask_type', choices=['rand', 'block'], default='rand', help='The mask strategy on atoms, rand or block.')
        parser.add_argument('--mask_all', type=str2bool, default=False, help='If true, mask basis, node features and edge features. If false, only mask node features, may have data leakge')
        parser.add_argument('--prepare_pretrain', type=str2bool, default=True, help='prepare labels for pretraining tasks')

        return parent_parser

    def __repr__(self):
        return f'OGB for Pretrain'

    def mask_features(self, graph, node_feats, edge_feats, pretrain_labels):
        # basis_c = copy.deepcopy(basis)
        node_feats_c = copy.deepcopy(node_feats)
        edge_feats_c = copy.deepcopy(edge_feats)

        if self.mask_type == 'rand':
            # node_mask_booled, edge_mask_booled = self._mask_pos_rand(graph, num_node, num_edge)
            self._mask_pos_rand(graph, pretrain_labels)
        elif self.mask_type == 'block':
            node_mask_booled, edge_mask_booled = self._mask_pos_block(graph)
        else:
            raise NotImplementedError

        if self.mask_all:
            # mask node features, and edge features
            # n_f = node_feats_c['0'].squeeze()
            # n_m = graph.ndata['node_mask'].unsqueeze(-1).repeat(1,6)
            # node_feats_c['0'] = torch.mul(n_f, n_m).unsqueeze(-1)
            
            # e_f = edge_feats_c['0'].squeeze()
            # e_m = graph.edata['edge_mask'].unsqueeze(-1).repeat(1,4)
            # edge_feats_c['0'] = torch.mul(e_f, e_m).unsqueeze(-1)
            node_feats_c['0'][~graph.ndata['node_mask'].bool()] = 0
            edge_feats_c['0'][~graph.edata['edge_mask'].bool()] = 0
            # drop edge
            
            graph.ndata['pos'][~graph.ndata['node_mask'].bool()] = 0

            return  node_feats_c, edge_feats_c

        else:
            # mask node features only
            n_f = node_feats_c['0'].squeeze()
            n_m = node_mask_booled.to(torch.int).unsqueeze(-1).repeat(1,6)
            node_feats_c['0'] = torch.mul(n_f, n_m).unsqueeze(-1)
            
            return  node_feats_c, edge_feats_c

    def _mask_pos_rand(self, graph, pretrain_labels, noaug=False):
        num_node = len(graph.nodes())
        num_edge = len(graph.edges()[0])
        n_atoms_per_mol = graph.batch_num_nodes()
        n_edges_per_mol = graph.batch_num_edges()
        neighbors_bin = pretrain_labels['neighborhood_to_mol_map'].bincount()
        non_terminal = torch.tensor(list(pretrain_labels['neighbors'].keys()))

        graph.ndata['node_mask'] = torch.ones(num_node, device=graph.device)
        graph.edata['edge_mask'] = torch.ones(num_edge, device=graph.device)

        if self.mask_all:
            # besides node features, mask basis and edge features as well
            n_nt_prev_mol = 0
            for n in neighbors_bin:
                n_mask = math.ceil(n * self.mask_ratio)
                perm = torch.randperm(n, device=graph.device) 
                graph.ndata['node_mask'][non_terminal[perm[:n_mask]+n_nt_prev_mol]] = 0
                n_nt_prev_mol += n

            masked_edges_id = graph.filter_edges(self.is_node_masked)
            # masked_edges = graph.find_edges(masked_edges_id)
            graph.edata['edge_mask'][masked_edges_id] = 0


        else:
            n_atoms_prev_mol = 0
            for n in n_atoms_per_mol:
                n_mask = math.ceil(n * self.mask_ratio)
                perm = torch.randperm(n, device=graph.device)
                graph.ndata['node_mask'][perm[:n_mask]+n_atoms_prev_mol] = False
                n_atoms_prev_mol += n
                
            # return node_mask_booled, edge_mask_booled


    def _mask_pos_block(self, graph, num_node, num_edge, noaug=False):
        pass


    @staticmethod
    def is_node_masked(edges):
        # print(edges.src['node_mask'] == 0)
        return torch.logical_or(edges.src['node_mask'] == 0, (edges.dst['node_mask'] == 0))


class OGBDataset(Dataset):
    def __init__(self,
                 data_dir: pathlib.Path,
                 ):
        self.root = data_dir
        self.length = 3378606
        from rdkit import Chem
        self.all_mols = Chem.SDMolSupplier(str(self.root)) 


    def __len__(self):
        return self.length

    
    def __getitem__(self, index):
        dgl_graph = None
        while dgl_graph == None:
            try:
                mol = self.all_mols[index]
                dgl_graph = self.featurize_mol(mol)
            except:
                index = random.choice(range(self.length))
                dgl_graph = None
        return dgl_graph


    def featurize_mol(self, mol):

        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(OGBDataset.atom_to_feature_vector(atom))
        
        x = np.array(atom_features_list, dtype=np.int64)

        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = OGBDataset.bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            edge_index = np.array(edges_list, dtype=np.int64)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:  # mol has no bonds
            edge_index = np.empty((0, 2), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
        
        pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)

        # build dgl graph object
        src, dst = edge_index.T
        graph = dgl.graph((src, dst))
        graph.ndata['pos'] = pos.clone().type(torch.float32)
        graph.ndata['attr'] = torch.tensor(x, dtype=torch.float32)
        graph.edata['edge_attr'] = torch.tensor(edge_attr, dtype=torch.float32)

        return graph


    
    @staticmethod
    def bond_to_feature_vector(bond):
        """
        Converts rdkit bond object to feature list of indices
        :param mol: rdkit bond object
        :return: list
        """
        bond_feature = [
            safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
            allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
            allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
        ]
        return bond_feature

    
    @staticmethod
    def atom_to_feature_vector(atom):
        """
        Converts rdkit atom object to feature list of indices
        :param mol: rdkit atom object
        :return: list
        """
        atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
        ]
        return atom_feature
