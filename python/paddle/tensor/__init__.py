# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function

#from .math import *
#from .creation import *
#from .linalg import *

# TODO: define alias in tensor and framework directory

from .random import randperm
from .attribute import rank  #DEFINE_ALIAS
from .attribute import shape  #DEFINE_ALIAS
from .creation import to_tensor  #DEFINE_ALIAS
from .creation import diag  #DEFINE_ALIAS
from .creation import eye  #DEFINE_ALIAS
# from .creation import fill_constant  #DEFINE_ALIAS
# from .creation import get_tensor_from_selected_rows        #DEFINE_ALIAS
from .creation import linspace  #DEFINE_ALIAS
from .creation import ones  #DEFINE_ALIAS
from .creation import ones_like  #DEFINE_ALIAS
from .creation import zeros  #DEFINE_ALIAS
from .creation import zeros_like  #DEFINE_ALIAS
from .creation import arange  #DEFINE_ALIAS
from .creation import eye  #DEFINE_ALIAS
from .creation import full  #DEFINE_ALIAS
from .creation import full_like  #DEFINE_ALIAS
from .creation import triu  #DEFINE_ALIAS
from .creation import tril  #DEFINE_ALIAS
from .creation import meshgrid  #DEFINE_ALIAS
from .creation import empty  #DEFINE_ALIAS
from .creation import empty_like  #DEFINE_ALIAS
from .linalg import matmul  #DEFINE_ALIAS
from .linalg import dot  #DEFINE_ALIAS
# from .linalg import einsum        #DEFINE_ALIAS
from .linalg import norm  #DEFINE_ALIAS
from .linalg import transpose  #DEFINE_ALIAS
from .linalg import dist  #DEFINE_ALIAS
from .linalg import t  #DEFINE_ALIAS
from .linalg import cross  #DEFINE_ALIAS
from .linalg import cholesky  #DEFINE_ALIAS
# from .linalg import tensordot        #DEFINE_ALIAS
from .linalg import bmm  #DEFINE_ALIAS
from .linalg import histogram  #DEFINE_ALIAS
from .linalg import mv  #DEFINE_ALIAS
from .logic import equal  #DEFINE_ALIAS
from .logic import greater_equal  #DEFINE_ALIAS
from .logic import greater_than  #DEFINE_ALIAS
from .logic import is_empty  #DEFINE_ALIAS
#from .logic import isfinite  #DEFINE_ALIAS
from .logic import less_equal  #DEFINE_ALIAS
from .logic import less_than  #DEFINE_ALIAS
from .logic import logical_and  #DEFINE_ALIAS
from .logic import logical_not  #DEFINE_ALIAS
from .logic import logical_or  #DEFINE_ALIAS
from .logic import logical_xor  #DEFINE_ALIAS
from .logic import not_equal  #DEFINE_ALIAS
# from .logic import reduce_all  #DEFINE_ALIAS
# from .logic import reduce_any  #DEFINE_ALIAS
from .logic import allclose  #DEFINE_ALIAS
from .logic import equal_all  #DEFINE_ALIAS
# from .logic import isnan        #DEFINE_ALIAS
from .manipulation import cast  #DEFINE_ALIAS
from .manipulation import concat  #DEFINE_ALIAS
from .manipulation import expand  #DEFINE_ALIAS
from .manipulation import broadcast_to  #DEFINE_ALIAS
from .manipulation import expand_as  #DEFINE_ALIAS
from .manipulation import tile  #DEFINE_ALIAS
from .manipulation import flatten  #DEFINE_ALIAS
from .manipulation import gather  #DEFINE_ALIAS
from .manipulation import gather_nd  #DEFINE_ALIAS
from .manipulation import reshape  #DEFINE_ALIAS
from .manipulation import flip as reverse  #DEFINE_ALIAS
from .manipulation import scatter  #DEFINE_ALIAS
from .manipulation import scatter_nd_add  #DEFINE_ALIAS
from .manipulation import scatter_nd  #DEFINE_ALIAS
from .manipulation import shard_index  #DEFINE_ALIAS
from .manipulation import slice  #DEFINE_ALIAS
from .manipulation import split  #DEFINE_ALIAS
from .manipulation import squeeze  #DEFINE_ALIAS
from .manipulation import stack  #DEFINE_ALIAS
from .manipulation import strided_slice  #DEFINE_ALIAS
from .manipulation import transpose  #DEFINE_ALIAS
from .manipulation import unique  #DEFINE_ALIAS
from .manipulation import unsqueeze  #DEFINE_ALIAS
from .manipulation import unstack  #DEFINE_ALIAS
from .manipulation import flip  #DEFINE_ALIAS
from .manipulation import unbind  #DEFINE_ALIAS
from .manipulation import roll  #DEFINE_ALIAS
from .manipulation import chunk  #DEFINE_ALIAS
from .math import abs  #DEFINE_ALIAS
from .math import acos  #DEFINE_ALIAS
from .math import asin  #DEFINE_ALIAS
from .math import atan  #DEFINE_ALIAS
from .math import ceil  #DEFINE_ALIAS
from .math import cos  #DEFINE_ALIAS
from .math import cosh  #DEFINE_ALIAS
from .math import cumsum  #DEFINE_ALIAS
# from .math import elementwise_add  #DEFINE_ALIAS
# from .math import elementwise_div  #DEFINE_ALIAS
# from .math import elementwise_floordiv  #DEFINE_ALIAS
# from .math import elementwise_mul  #DEFINE_ALIAS
# from .math import elementwise_mod  #DEFINE_ALIAS
# from .math import elementwise_pow  #DEFINE_ALIAS
# from .math import elementwise_sub  #DEFINE_ALIAS
from .math import exp  #DEFINE_ALIAS
from .math import floor  #DEFINE_ALIAS
from .math import increment  #DEFINE_ALIAS
from .math import log  #DEFINE_ALIAS
from .math import multiplex  #DEFINE_ALIAS
from .math import pow  #DEFINE_ALIAS
from .math import reciprocal  #DEFINE_ALIAS
# from .math import reduce_max  #DEFINE_ALIAS
# from .math import reduce_min  #DEFINE_ALIAS
# from .math import reduce_prod  #DEFINE_ALIAS
# from .math import reduce_sum  #DEFINE_ALIAS
from .math import round  #DEFINE_ALIAS
from .math import rsqrt  #DEFINE_ALIAS
from .math import scale  #DEFINE_ALIAS
from .math import sign  #DEFINE_ALIAS
from .math import sin  #DEFINE_ALIAS
from .math import sinh  #DEFINE_ALIAS
from .math import sqrt  #DEFINE_ALIAS
from .math import square  #DEFINE_ALIAS
from .math import stanh  #DEFINE_ALIAS
from .math import sum  #DEFINE_ALIAS
from .math import tanh  #DEFINE_ALIAS
from .math import add_n  #DEFINE_ALIAS
from .math import max  #DEFINE_ALIAS
from .math import maximum  #DEFINE_ALIAS
from .math import min  #DEFINE_ALIAS
from .math import minimum  #DEFINE_ALIAS
from .math import mm  #DEFINE_ALIAS
from .math import divide  #DEFINE_ALIAS
from .math import floor_divide  #DEFINE_ALIAS
from .math import remainder  #DEFINE_ALIAS
from .math import mod  #DEFINE_ALIAS
from .math import floor_mod  #DEFINE_ALIAS
from .math import multiply  #DEFINE_ALIAS
from .math import add  #DEFINE_ALIAS
from .math import atan  #DEFINE_ALIAS
from .math import logsumexp  #DEFINE_ALIAS
from .math import inverse  #DEFINE_ALIAS
from .math import log1p  #DEFINE_ALIAS
from .math import erf  #DEFINE_ALIAS
# from .math import addcmul  #DEFINE_ALIAS
from .math import addmm  #DEFINE_ALIAS
from .math import clip  #DEFINE_ALIAS
from .math import trace  #DEFINE_ALIAS
from .math import kron  #DEFINE_ALIAS
from .math import isfinite  #DEFINE_ALIAS
from .math import isinf  #DEFINE_ALIAS
from .math import isnan  #DEFINE_ALIAS
from .math import prod  #DEFINE_ALIAS
from .random import multinomial  #DEFINE_ALIAS
from .random import standard_normal
from .random import normal
from .random import uniform  #DEFINE_ALIAS
from .random import randn  #DEFINE_ALIAS
from .random import rand  #DEFINE_ALIAS
from .random import randint  #DEFINE_ALIAS
from .random import randperm  #DEFINE_ALIAS
from .search import argmax  #DEFINE_ALIAS
from .search import argmin  #DEFINE_ALIAS
from .search import argsort  #DEFINE_ALIAS
# from .search import has_inf  #DEFINE_ALIAS
# from .search import has_nan  #DEFINE_ALIAS
# from .search import masked_select        #DEFINE_ALIAS
from .search import topk  #DEFINE_ALIAS
from .search import where  #DEFINE_ALIAS
from .search import index_select  #DEFINE_ALIAS
from .search import nonzero  #DEFINE_ALIAS
from .search import sort  #DEFINE_ALIAS
from .search import index_sample  #DEFINE_ALIAS
from .search import masked_select  #DEFINE_ALIAS
from .stat import mean  #DEFINE_ALIAS
# from .stat import reduce_mean  #DEFINE_ALIAS
from .stat import std  #DEFINE_ALIAS
from .stat import var  #DEFINE_ALIAS
from .stat import numel  #DEFINE_ALIAS
# from .tensor import Tensor        #DEFINE_ALIAS
# from .tensor import LoDTensor        #DEFINE_ALIAS
# from .tensor import LoDTensorArray        #DEFINE_ALIAS
from .to_string import set_printoptions
