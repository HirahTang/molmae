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

import logging
from typing import Optional, Literal, Dict

import torch
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.layers.attention import AttentionBlockSE3
from se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from se3_transformer.model.layers.norm import NormSE3
from se3_transformer.model.layers.pooling import GPooling
from se3_transformer.runtime.utils import str2bool
from se3_transformer.model.fiber import Fiber
from se3_transformer.model.transformer import *

class MAE(SE3TransformerPooled):
    def __init__(self, 
                 fiber_in: Fiber, 
                 fiber_out: Fiber, 
                 fiber_edge: Fiber, 
                 num_degrees: int, 
                 num_channels: int, 
                 output_dim: int, 
                 mask_token:
                 **kwargs):
        super().__init__(fiber_in, fiber_out, fiber_edge, num_degrees, num_channels, output_dim, **kwargs)
