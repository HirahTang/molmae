import logging


import logging
from secrets import choice
from typing import Optional, Literal, Dict
from sympy import false
import math
import copy

import torch
from torch import squeeze
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

from se3_transformer.model import *
from se3_transformer.runtime.utils import str2bool
from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.transformer import get_populated_edge_features
from se3_transformer.model.layers.convolution import ConvSE3FuseLevel


class base_MAE(nn.Module):
    def __init__(self, 
                 num_layers: int, 
                 fiber_in: Fiber, 
                 fiber_hidden: Fiber, 
                 fiber_out: Fiber, 
                 num_heads: int, 
                 channels_div: int, 
                 fiber_edge: Fiber,
                 output_dim: int,
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


        n_out_features = fiber_out.num_features
        self.mlp = nn.Sequential(
            nn.Linear(n_out_features, n_out_features),
            nn.ReLU(),
            nn.Linear(n_out_features, output_dim)
        )

        self.mask_ratio = kwargs['mask_ratio']
        self.mask_type = kwargs['mask_type']
        self.mask_all = kwargs['mask_all']

        
    def _mask_pos_rand(self, graph, num_node, num_edge, noaug=False):
        n_atoms_per_mol = graph._batch_num_nodes['_N']
        n_edges_per_mol = graph._batch_num_edges[('_N', '_E', '_N')]

        node_mask_booled = torch.ones(num_node, device=graph.device).to(torch.bool)
        edge_mask_booled = torch.ones(num_node, device=graph.device).to(torch.bool)

        if self.mask_all:
            # besides node features, mask basis and edge features as well
            n_atoms_prev_mol = 0
            for n in n_atoms_per_mol:
                n_mask = math.ceil(n * self.mask_ratio)
                perm = torch.randperm(n, device=graph.device)
                node_mask_booled[perm[:n_mask]+n_atoms_prev_mol] = False
                n_atoms_prev_mol += n

            for n in n_edges_per_mol:
                n_mask = math.ceil(n * self.mask_ratio)

        else:
            n_atoms_prev_mol = 0
            for n in n_atoms_per_mol:
                n_mask = math.ceil(n * self.mask_ratio)
                perm = torch.randperm(n, device=graph.device)
                node_mask_booled[perm[:n_mask]+n_atoms_prev_mol] = False
                n_atoms_prev_mol += n
                
            return node_mask_booled, edge_mask_booled


    def _mask_pos_block(self, graph, num_node, num_edge, noaug=False):
        pass


    def forward(self, graph: DGLGraph, node_feats: Dict[str, Tensor], 
                edge_feats: Optional[Dict[str, Tensor]] = None, 
                basis: Optional[Dict[str, Tensor]] = None):
        # Compute bases if they weren't precomputed as part of the data loading
        basis = basis or get_basis(graph.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=False,
                                   use_pad_trick=self.tensor_cores and not self.low_memory,
                                   amp=torch.is_autocast_enabled())
        # add fused bases (per output degree, per input degree, and fully fused) to the dict
        basis = update_basis_with_fused(basis, self.max_degree, use_pad_trick=self.tensor_cores and self.low_memory, fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)

        edge_feats = get_populated_edge_features(graph.edata['rel_pos'], edge_feats)
        
        (node_mask_booled, edge_mask_booled), basis_masked, node_feats_masked, edge_feats_masked = self.mask_features(graph, basis, node_feats, edge_feats)

        node_feats_recon = self.transformer.graph_modules(node_feats_masked, edge_feats_masked, graph=graph, basis=basis_masked)
        # node_feats = self.graph_module(node_feats, edge_feats, graph=graph, basis=basis)

        node_feats_recon = self.mlp(node_feats_recon[str(self.return_type)].squeeze())

        # return node_feats['0'].squeeze()[~node_mask_booled], node_feats_recon[~node_mask_booled]
        return graph.ndata['pos'][~node_mask_booled], node_feats_recon[~node_mask_booled]


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
        # arguments added for mask

        parser.add_argument('--mask_ratio', type=float, default=0.3, help="The mask ratio of the masked auto-encoder.")
        parser.add_argument('--mask_type', choices=['rand', 'block'], default='rand', help='The mask strategy on atoms, rand or block.')
        parser.add_argument('--mask_all', type=bool, default=False, help='If true, mask basis, node features and edge features. If false, only mask node features, may have data leakge')

        return parser
        
    def mask_features(self, graph, basis, node_feats, edge_feats):
        num_node = node_feats['0'].shape[0]
        num_edge = edge_feats['0'].shape[0]
        basis_c = copy.deepcopy(basis)
        node_feats_c = copy.deepcopy(node_feats)
        edge_feats_c = copy.deepcopy(edge_feats)

        if self.mask_type == 'rand':
            node_mask_booled, edge_mask_booled = self._mask_pos_rand(graph, num_node, num_edge)
        elif self.mask_type == 'block':
            node_mask_booled, edge_mask_booled = self._mask_pos_block(graph, num_node, num_edge)
        else:
            raise NotImplementedError

        if self.mask_all:
            # one degree for node features
            n_f = node_feats_c['0'].squeeze()
            n_m = node_mask_booled.to(torch.int).unsqueeze(-1).repeat(1,6)
            node_feats_c['0'] = torch.mul(n_f, n_m).unsqueeze(-1)

            # drop edge



        else:
            # one degree for node features
            n_f = node_feats_c['0'].squeeze()
            n_m = node_mask_booled.to(torch.int).unsqueeze(-1).repeat(1,6)
            node_feats_c['0'] = torch.mul(n_f, n_m).unsqueeze(-1)
            
            return (node_mask_booled, edge_mask_booled), basis_c, node_feats_c, edge_feats_c
