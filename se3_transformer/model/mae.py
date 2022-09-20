from email.policy import default
import logging


import logging
from re import X
from secrets import choice
from turtle import forward
from typing import Optional, Literal, Dict
from se3_transformer.model.fiber import FiberEl
from sympy import false
import math
import copy

import torch
from torch import squeeze
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor
from egnn_pytorch import EGNN_Sparse_Network

from se3_transformer.model import *
from se3_transformer.runtime.utils import str2bool
from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.transformer import get_populated_edge_features
from se3_transformer.model.layers.convolution import ConvSE3FuseLevel
from se3_transformer.model.layers.pooling import GPooling
from se3_transformer.model.layers.linear import LinearSE3
from se3_transformer.data_loading.qm9 import _get_relative_pos
from se3_transformer.run_mae.utils import ground_truth_local_stats



class base_MAE(nn.Module):
    def __init__(self, 
                 num_layers: int, 
                 fiber_in: Fiber, 
                 fiber_hidden: Fiber, 
                 fiber_out: Fiber, 
                 num_heads: int, 
                 channels_div: int, 
                 fiber_edge: Fiber,
                #  output_dim: int,
                 return_type: Optional[int] = None, 
                 pooling: Optional[Literal['avg', 'max']] = None, 
                 norm: bool = True, 
                 use_layer_norm: bool = True, 
                 tensor_cores: bool = False, 
                 low_memory: bool = False, 
                 **kwargs):
        super().__init__()
        self.transformer = SE3Transformer(
                num_layers,
                fiber_in, 
                fiber_hidden, 
                fiber_out, 
                num_heads, 
                channels_div, 
                fiber_edge, 
                return_type, 
                pooling, 
                norm, 
                use_layer_norm, 
                tensor_cores, 
                low_memory, 
                **kwargs)
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.return_type = return_type
        self.pooling = pooling
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory
        if low_memory:
            self.fuse_level = ConvSE3FuseLevel.NONE
        else:
            # Fully fused convolutions when using Tensor Cores (and not low memory mode)
            self.fuse_level = ConvSE3FuseLevel.FULL if tensor_cores else ConvSE3FuseLevel.PARTIAL

        self.pooling_module = GPooling(pool='avg', feat_type=0)
        final_fiber = default(fiber_out, fiber_hidden)

        self.linear_out = LinearSE3(
            final_fiber,
            Fiber(list(map(lambda t: FiberEl(degree = t[0], channels = 1), final_fiber)))
        )

    @staticmethod
    def is_node_masked(edges):
        # print(edges.src['node_mask'] == 0)
        return torch.logical_or(edges.src['node_mask'] == 0, (edges.dst['node_mask'] == 0))

    def forward(self, graph: DGLGraph, node_feats: Dict[str, Tensor], 
                edge_feats: Optional[Dict[str, Tensor]] = None, 
                basis: Optional[Dict[str, Tensor]] = None,
                pretrain_labels=None,
                ):        
        # Compute bases if they weren't precomputed as part of the data loading
        basis = basis or get_basis(graph.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=False,
                                   use_pad_trick=self.tensor_cores and not self.low_memory,
                                   amp=torch.is_autocast_enabled())
        # add fused bases (per output degree, per input degree, and fully fused) to the dict
        basis = update_basis_with_fused(basis, self.max_degree, use_pad_trick=self.tensor_cores and self.low_memory, fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)

        edge_feats = get_populated_edge_features(graph.edata['rel_pos'], edge_feats)
        
        embedding = self.transformer.graph_modules(node_feats, edge_feats, graph=graph, basis=basis)
        pred_coord = self.linear_out(embedding)
        pred_coord = map_values(lambda t: t.squeeze(dim=1), pred_coord)

        # embedding_node = embedding['0'].squeeze(-1)
        # embedding_mol = self.pooling_module(embedding, graph=graph) 

        bond_length_pred, bond_angle_pred, _ = ground_truth_local_stats(pos=pred_coord['1'], neighbors=pretrain_labels['neighbors'], neighbor_mask=pretrain_labels['neighbor_masks'])
        # radius_pred = self.radius_head(embedding_node)

        # orientation_pred = self.orientation_head(embedding_mol)

        # node_feats = self.graph_module(node_feats, edge_feats, graph=graph, basis=basis)
        # return node_feats['0'].squeeze()[~node_mask_booled], node_feats_recon[~node_mask_booled]
        # node_gt = torch.concat([graph.ndata['pos'], node_feats['0'].squeeze()], dim=-1)
        # edge_gt = torch.concat([graph.edata['pos'], node_feats['0'].squeeze()], dim=-1)
        return (bond_length_pred, bond_angle_pred)#, radius_pred, orientation_pred)


    def embed(self, graph, node_feats, edge_feats, basis=None):
        basis = basis or get_basis(graph.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=False,
                                use_pad_trick=self.tensor_cores and not self.low_memory,
                                amp=torch.is_autocast_enabled())
        basis = update_basis_with_fused(basis, self.max_degree, use_pad_trick=self.tensor_cores and self.low_memory, fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)
        edge_feats = get_populated_edge_features(graph.edata['rel_pos'], edge_feats)
        node_feats = self.graph_modules(node_feats, edge_feats, graph=graph, basis=basis)

        return  node_feats


    @staticmethod
    def add_argparse_args(parser):
        return parser
    

## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B, C)
        x = self.proj(x)
        x - self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_layers=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
            )
            for i in range(num_layers)
        ])

    def forward(self, x, pos):
        for  _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, num_layers=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(num_layers)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x



class egnn_MAE(nn.Module):
    def __init__(self, num_node_features, num_edge_features, **kwargs):
        super().__init__()
        self.egnn_encoder = EGNN_Sparse_Network(
            # n_layers=kwargs['egnn_layers'],
            n_layers=4,
            feats_dim=num_node_features, # 6
            pos_dim=3,
            edge_attr_dim=num_edge_features,  # 4
            m_dim = 128,
            embedding_nums=[8],  embedding_dims=[16],
            edge_embedding_nums=[4], edge_embedding_dims=[8],
            update_coors=True, update_feats=True,
            norm_feats=False, norm_coors=False, recalc=False
        )
        kwargs['trans_dim'] = kwargs.get("trans_dim", 128)
        self.mask_ratio = kwargs['mask_ratio']
        self.mask_type = kwargs['mask_type']
        self.mask_all = kwargs['mask_all']
        self.trans_dim = 128 if kwargs['trans_dim'] != None else 128
        # self.config = kwargs
        self.num_heads = kwargs['num_heads']
        self.num_layers = kwargs['num_layers']
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            num_layers = self.num_layers,
            num_heads = self.num_heads
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.trans_dim, 128),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )
    
    def forward(self, graph: DGLGraph, node_feats: Dict[str, Tensor], 
                edge_feats: Optional[Dict[str, Tensor]] = None, ):

        # edge_feats = get_populated_edge_features(graph.edata['rel_pos'], edge_feats)
        egnn_x = torch.cat([graph.ndata['pos'], node_feats['0'].squeeze()], dim=-1)
        edge_index = torch.stack([graph.edges()[0],graph.edges()[1]], dim=0)
        embeddings = self.egnn_encoder(x=egnn_x,
                                       edge_index=edge_index,
                                       edge_attr=edge_feats['0'].squeeze(),
                                       batch=graph._batch_num_nodes['_N'])
        
        if self.mask_type == 'rand':
            # bool_masked_pos = self._mask_pos_rand(graph, node_feats.shape[0])
            self._mask_pos_rand(graph, node_feats.shape[0]) # bool_masked_pos = graph.ndata['node_mask']
        elif self.mask_type == 'block':
            bool_masked_pos = self._mask_pos_block(graph, node_feats.shape[0])
        else:
            raise NotImplementedError
        
        # x_vis = embeddings[graph.ndata['node_mask']]
        embeddings[~graph.ndata['node_mask']] = self.mask_token
        pos = self.pos_embed(x_vis[:3])

        # transformer encoder
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        # reconstruct
        # B, _, C = x_vis.shape
        # _, N, _ = pos_emb_mask.shape
        # pos_emb_vis = self.decoder_pos_embed(embeddings[graph.ndata['node_mask']])
        # pos_emb_mask = self.decoder_pos_embed(embeddings[~graph.ndata['node_mask']])
        x_recon = self.decoder(x_vis)
        x_gt = graph.ndata['pos'][~graph.ndata['node_mask']]

        return x_gt




        # pred = self.egnn(x=x_egnn, edge_index=edge_index, edge_attr=edge_attr_stoch, batch=batch)

        node_feats_recon = self.transformer.graph_modules(node_feats_masked, edge_feats_masked, graph=graph, basis=basis)
        # node_feats = self.graph_module(node_feats, edge_feats, graph=graph, basis=basis)

        node_feats_recon = self.mlp(node_feats_recon[str(self.return_type)].squeeze())

        # return node_feats['0'].squeeze()[~node_mask_booled], node_feats_recon[~node_mask_booled]
        node_gt = torch.concat([graph.ndata['pos'], node_feats['0'].squeeze()], dim=-1)
        # edge_gt = torch.concat([graph.edata['pos'], node_feats['0'].squeeze()], dim=-1)
        node_mask_booled = graph.ndata['node_mask'].to(torch.bool)
        return node_gt[~node_mask_booled], node_feats_recon[~node_mask_booled]

    def _mask_pos_rand(self, graph, num_node, noaug=False):
        n_atoms_per_mol = graph._batch_num_nodes['_N']
        graph.ndata['node_mask'] = torch.ones(num_node, device=graph.device)

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

    def mask_features(self, graph, node_feats):
        num_node = node_feats['0'].shape[0]
        node_feats_c = copy.deepcopy(node_feats)

        if self.mask_type == 'rand':
            # node_mask_booled, edge_mask_booled = self._mask_pos_rand(graph, num_node, num_edge)
            self._mask_pos_rand(graph, num_node)
        elif self.mask_type == 'block':
            node_mask_booled, edge_mask_booled = self._mask_pos_block(graph, num_node)
        else:
            raise NotImplementedError

        # mask node embedding
        n_f = node_feats_c['0'].squeeze()
        n_m = node_mask_booled.to(torch.int).unsqueeze(-1).repeat(1,6)
        node_feats_c['0'] = torch.mul(n_f, n_m).unsqueeze(-1)
        
        return  node_feats_c


def map_values(fn, d):
    return {k: fn(v) for k, v in d.items()}


def default(val, d):
    if val is not None:
        return val
    else:
        return d
