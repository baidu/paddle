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

import os

try:
    from paddle.version import full_version as __version__
    from paddle.version import commit as __git_commit__

except ImportError:
    import sys
    sys.stderr.write('''Warning with import paddle: you should not
     import paddle from the source directory; please install paddlepaddle*.whl firstly.'''
                     )

import paddle.batch
#batch = batch.batch
from .fluid import monkey_patch_variable
from .fluid.dygraph import monkey_patch_math_varbase
monkey_patch_variable()
monkey_patch_math_varbase()
import paddle.framework
from .framework import VarBase as Tensor
import paddle.compat
import paddle.distributed
import paddle.sysconfig
import paddle.tensor
import paddle.distribution
import paddle.nn
import paddle.distributed.fleet
import paddle.optimizer
import paddle.metric
import paddle.device
import paddle.regularizer
import paddle.incubate

# TODO: define alias in tensor and framework directory

from .tensor.random import randperm
from .tensor.random import bernoulli

from .tensor.attribute import rank  #DEFINE_ALIAS
from .tensor.attribute import shape  #DEFINE_ALIAS
from .tensor.attribute import real  #DEFINE_ALIAS
from .tensor.attribute import imag  #DEFINE_ALIAS
from .tensor.creation import to_tensor  #DEFINE_ALIAS
from .tensor.creation import diag  #DEFINE_ALIAS
from .tensor.creation import eye  #DEFINE_ALIAS
# from .tensor.creation import fill_constant  #DEFINE_ALIAS
# from .tensor.creation import get_tensor_from_selected_rows        #DEFINE_ALIAS
from .tensor.creation import linspace  #DEFINE_ALIAS
from .tensor.creation import ones  #DEFINE_ALIAS
from .tensor.creation import ones_like  #DEFINE_ALIAS
from .tensor.creation import zeros  #DEFINE_ALIAS
from .tensor.creation import zeros_like  #DEFINE_ALIAS
from .tensor.creation import arange  #DEFINE_ALIAS
from .tensor.creation import eye  #DEFINE_ALIAS
from .tensor.creation import full  #DEFINE_ALIAS
from .tensor.creation import full_like  #DEFINE_ALIAS
from .tensor.creation import triu  #DEFINE_ALIAS
from .tensor.creation import tril  #DEFINE_ALIAS
from .tensor.creation import meshgrid  #DEFINE_ALIAS
from .tensor.creation import empty  #DEFINE_ALIAS
from .tensor.creation import empty_like  #DEFINE_ALIAS
from .tensor.creation import assign  #DEFINE_ALIAS
from .tensor.linalg import matmul  #DEFINE_ALIAS
from .tensor.linalg import dot  #DEFINE_ALIAS
# from .tensor.linalg import einsum        #DEFINE_ALIAS
from .tensor.linalg import norm  #DEFINE_ALIAS
from .tensor.linalg import transpose  #DEFINE_ALIAS
from .tensor.linalg import dist  #DEFINE_ALIAS
from .tensor.linalg import t  #DEFINE_ALIAS
from .tensor.linalg import cross  #DEFINE_ALIAS
from .tensor.linalg import cholesky  #DEFINE_ALIAS
# from .tensor.linalg import tensordot        #DEFINE_ALIAS
from .tensor.linalg import bmm  #DEFINE_ALIAS
from .tensor.linalg import histogram  #DEFINE_ALIAS
from .tensor.linalg import mv  #DEFINE_ALIAS
from .tensor.logic import equal  #DEFINE_ALIAS
from .tensor.logic import greater_equal  #DEFINE_ALIAS
from .tensor.logic import greater_than  #DEFINE_ALIAS
from .tensor.logic import is_empty  #DEFINE_ALIAS
#from .tensor.logic import isfinite  #DEFINE_ALIAS
from .tensor.logic import less_equal  #DEFINE_ALIAS
from .tensor.logic import less_than  #DEFINE_ALIAS
from .tensor.logic import logical_and  #DEFINE_ALIAS
from .tensor.logic import logical_not  #DEFINE_ALIAS
from .tensor.logic import logical_or  #DEFINE_ALIAS
from .tensor.logic import logical_xor  #DEFINE_ALIAS
from .tensor.logic import not_equal  #DEFINE_ALIAS
from .tensor.logic import allclose  #DEFINE_ALIAS
from .tensor.logic import equal_all  #DEFINE_ALIAS
# from .tensor.logic import isnan        #DEFINE_ALIAS
from .tensor.logic import is_tensor  #DEFINE_ALIAS
from .tensor.manipulation import cast  #DEFINE_ALIAS
from .tensor.manipulation import concat  #DEFINE_ALIAS
from .tensor.manipulation import expand  #DEFINE_ALIAS
from .tensor.manipulation import broadcast_to  #DEFINE_ALIAS
from .tensor.manipulation import expand_as  #DEFINE_ALIAS
from .tensor.manipulation import tile  #DEFINE_ALIAS
from .tensor.manipulation import flatten  #DEFINE_ALIAS
from .tensor.manipulation import gather  #DEFINE_ALIAS
from .tensor.manipulation import gather_nd  #DEFINE_ALIAS
from .tensor.manipulation import reshape  #DEFINE_ALIAS
from .tensor.manipulation import reshape_  #DEFINE_ALIAS
from .tensor.manipulation import flip as reverse  #DEFINE_ALIAS
from .tensor.manipulation import scatter  #DEFINE_ALIAS
from .tensor.manipulation import scatter_  #DEFINE_ALIAS
from .tensor.manipulation import scatter_nd_add  #DEFINE_ALIAS
from .tensor.manipulation import scatter_nd  #DEFINE_ALIAS
from .tensor.manipulation import shard_index  #DEFINE_ALIAS
from .tensor.manipulation import slice  #DEFINE_ALIAS
from .tensor.manipulation import split  #DEFINE_ALIAS
from .tensor.manipulation import squeeze  #DEFINE_ALIAS
from .tensor.manipulation import squeeze_  #DEFINE_ALIAS
from .tensor.manipulation import stack  #DEFINE_ALIAS
from .tensor.manipulation import strided_slice  #DEFINE_ALIAS
from .tensor.manipulation import transpose  #DEFINE_ALIAS
from .tensor.manipulation import unique  #DEFINE_ALIAS
from .tensor.manipulation import unsqueeze  #DEFINE_ALIAS
from .tensor.manipulation import unsqueeze_  #DEFINE_ALIAS
from .tensor.manipulation import unstack  #DEFINE_ALIAS
from .tensor.manipulation import flip  #DEFINE_ALIAS
from .tensor.manipulation import unbind  #DEFINE_ALIAS
from .tensor.manipulation import roll  #DEFINE_ALIAS
from .tensor.manipulation import chunk  #DEFINE_ALIAS
from .tensor.math import abs  #DEFINE_ALIAS
from .tensor.math import acos  #DEFINE_ALIAS
from .tensor.math import asin  #DEFINE_ALIAS
from .tensor.math import atan  #DEFINE_ALIAS
from .tensor.math import ceil  #DEFINE_ALIAS
from .tensor.math import cos  #DEFINE_ALIAS
from .tensor.math import cosh  #DEFINE_ALIAS
from .tensor.math import cumsum  #DEFINE_ALIAS
# from .tensor.math import elementwise_add  #DEFINE_ALIAS
# from .tensor.math import elementwise_div  #DEFINE_ALIAS
# from .tensor.math import elementwise_floordiv  #DEFINE_ALIAS
# from .tensor.math import elementwise_mod  #DEFINE_ALIAS
# from .tensor.math import elementwise_pow  #DEFINE_ALIAS
# from .tensor.math import elementwise_sub  #DEFINE_ALIAS
from .tensor.math import exp  #DEFINE_ALIAS
from .tensor.math import floor  #DEFINE_ALIAS
from .tensor.math import increment  #DEFINE_ALIAS
from .tensor.math import log  #DEFINE_ALIAS
from .tensor.math import log2  #DEFINE_ALIAS
from .tensor.math import log10  #DEFINE_ALIAS
from .tensor.math import multiplex  #DEFINE_ALIAS
from .tensor.math import pow  #DEFINE_ALIAS
from .tensor.math import reciprocal  #DEFINE_ALIAS
# from .tensor.math import reduce_max  #DEFINE_ALIAS
# from .tensor.math import reduce_min  #DEFINE_ALIAS
# from .tensor.math import reduce_prod  #DEFINE_ALIAS
# from .tensor.math import reduce_sum  #DEFINE_ALIAS
from .tensor.math import all  #DEFINE_ALIAS
from .tensor.math import any  #DEFINE_ALIAS
from .tensor.math import round  #DEFINE_ALIAS
from .tensor.math import rsqrt  #DEFINE_ALIAS
from .tensor.math import scale  #DEFINE_ALIAS
from .tensor.math import sign  #DEFINE_ALIAS
from .tensor.math import sin  #DEFINE_ALIAS
from .tensor.math import sinh  #DEFINE_ALIAS
from .tensor.math import sqrt  #DEFINE_ALIAS
from .tensor.math import square  #DEFINE_ALIAS
from .tensor.math import stanh  #DEFINE_ALIAS
from .tensor.math import sum  #DEFINE_ALIAS
from .tensor.math import tanh  #DEFINE_ALIAS
from .tensor.math import tanh_  #DEFINE_ALIAS
from .tensor.math import add_n  #DEFINE_ALIAS
from .tensor.math import max  #DEFINE_ALIAS
from .tensor.math import maximum  #DEFINE_ALIAS
from .tensor.math import min  #DEFINE_ALIAS
from .tensor.math import minimum  #DEFINE_ALIAS
from .tensor.math import mm  #DEFINE_ALIAS
from .tensor.math import divide  #DEFINE_ALIAS
from .tensor.math import floor_divide  #DEFINE_ALIAS
from .tensor.math import remainder  #DEFINE_ALIAS
from .tensor.math import mod  #DEFINE_ALIAS
from .tensor.math import floor_mod  #DEFINE_ALIAS
from .tensor.math import multiply  #DEFINE_ALIAS
from .tensor.math import add  #DEFINE_ALIAS
from .tensor.math import subtract  #DEFINE_ALIAS
from .tensor.math import atan  #DEFINE_ALIAS
from .tensor.math import logsumexp  #DEFINE_ALIAS
from .tensor.math import inverse  #DEFINE_ALIAS
from .tensor.math import log1p  #DEFINE_ALIAS
from .tensor.math import erf  #DEFINE_ALIAS
from .tensor.math import addmm  #DEFINE_ALIAS
from .tensor.math import clip  #DEFINE_ALIAS
from .tensor.math import trace  #DEFINE_ALIAS
from .tensor.math import kron  #DEFINE_ALIAS
from .tensor.math import isfinite  #DEFINE_ALIAS
from .tensor.math import isinf  #DEFINE_ALIAS
from .tensor.math import isnan  #DEFINE_ALIAS
from .tensor.math import prod  #DEFINE_ALIAS
from .tensor.math import broadcast_shape  #DEFINE_ALIAS
from .tensor.math import conj  #DEFINE_ALIAS

from .tensor.random import multinomial  #DEFINE_ALIAS
from .tensor.random import standard_normal
from .tensor.random import normal
from .tensor.random import uniform  #DEFINE_ALIAS
from .tensor.random import randn  #DEFINE_ALIAS
from .tensor.random import rand  #DEFINE_ALIAS
from .tensor.random import randint  #DEFINE_ALIAS
from .tensor.random import randperm  #DEFINE_ALIAS
from .tensor.search import argmax  #DEFINE_ALIAS
from .tensor.search import argmin  #DEFINE_ALIAS
from .tensor.search import argsort  #DEFINE_ALIAS
# from .tensor.search import has_inf  #DEFINE_ALIAS
# from .tensor.search import has_nan  #DEFINE_ALIAS
from .tensor.search import masked_select  #DEFINE_ALIAS
from .tensor.search import topk  #DEFINE_ALIAS
from .tensor.search import where  #DEFINE_ALIAS
from .tensor.search import index_select  #DEFINE_ALIAS
from .tensor.search import nonzero  #DEFINE_ALIAS
from .tensor.search import sort  #DEFINE_ALIAS

from .tensor.to_string import set_printoptions  #DEFINE_ALIAS

from .framework.random import seed  #DEFINE_ALIAS
from .framework.random import get_cuda_rng_state  #DEFINE_ALIAS
from .framework.random import set_cuda_rng_state  #DEFINE_ALIAS
from .framework import ParamAttr  #DEFINE_ALIAS
# from .framework import create_global_var  #DEFINE_ALIAS
from .framework import create_parameter  #DEFINE_ALIAS
from .framework import CPUPlace  #DEFINE_ALIAS
from .framework import CUDAPlace  #DEFINE_ALIAS
from .framework import CUDAPinnedPlace  #DEFINE_ALIAS

from .framework import grad  #DEFINE_ALIAS
from .framework import no_grad  #DEFINE_ALIAS
from .framework import save  #DEFINE_ALIAS
from .framework import load  #DEFINE_ALIAS
from .framework import DataParallel  #DEFINE_ALIAS

from .framework import set_default_dtype  #DEFINE_ALIAS
from .framework import get_default_dtype  #DEFINE_ALIAS

from .tensor.search import index_sample  #DEFINE_ALIAS
from .tensor.stat import mean  #DEFINE_ALIAS
# from .tensor.stat import reduce_mean  #DEFINE_ALIAS
from .tensor.stat import std  #DEFINE_ALIAS
from .tensor.stat import var  #DEFINE_ALIAS
# from .fluid.data import data
from .tensor.stat import numel  #DEFINE_ALIAS
from .tensor.stat import median  #DEFINE_ALIAS
from .device import get_cudnn_version
from .device import set_device
from .device import get_device
from .device import is_compiled_with_cuda  #DEFINE_ALIAS
from .device import is_compiled_with_xpu
from .device import XPUPlace
# from .tensor.tensor import Tensor        #DEFINE_ALIAS
# from .tensor.tensor import LoDTensor        #DEFINE_ALIAS
# from .tensor.tensor import LoDTensorArray        #DEFINE_ALIAS

from .fluid.dygraph.base import enable_dygraph as disable_static  #DEFINE_ALIAS
from .fluid.dygraph.base import disable_dygraph as enable_static  #DEFINE_ALIAS
from .fluid.framework import in_dygraph_mode as in_dynamic_mode  #DEFINE_ALIAS
from .fluid.layers import crop_tensor as crop  #DEFINE_ALIAS

from . import jit
from . import static
from . import amp
from . import onnx

# high-level api
from .hapi import Model
from .hapi import callbacks
from .hapi import summary
from .hapi import flops
import paddle.text
import paddle.vision

disable_static()
