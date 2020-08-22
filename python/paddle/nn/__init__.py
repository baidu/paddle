#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: import all neural network related api under this directory,
# including layers, linear, conv, rnn etc.

from .layer import norm
from .functional import extension
from .layer import common
from .utils import weight_norm_hook

from . import initializer

__all__ = []
__all__ += norm.__all__
__all__ += extension.__all__
__all__ += common.__all__
__all__ += weight_norm_hook.__all__

# TODO: define alias in nn directory
# from .clip import ErrorClipByValue        #DEFINE_ALIAS
from .clip import GradientClipByGlobalNorm  #DEFINE_ALIAS
from .clip import GradientClipByNorm  #DEFINE_ALIAS
from .clip import GradientClipByValue  #DEFINE_ALIAS
# from .clip import set_gradient_clip        #DEFINE_ALIAS
from .clip import clip  #DEFINE_ALIAS
from .clip import clip_by_norm  #DEFINE_ALIAS
from .control_flow import case  #DEFINE_ALIAS
from .control_flow import cond  #DEFINE_ALIAS
# from .control_flow import DynamicRNN        #DEFINE_ALIAS
# from .control_flow import StaticRNN        #DEFINE_ALIAS
from .control_flow import switch_case  #DEFINE_ALIAS
from .control_flow import while_loop  #DEFINE_ALIAS
# from .control_flow import rnn        #DEFINE_ALIAS
# from .decode import BeamSearchDecoder        #DEFINE_ALIAS
# from .decode import Decoder        #DEFINE_ALIAS
from .decode import beam_search  #DEFINE_ALIAS
from .decode import beam_search_decode  #DEFINE_ALIAS
# from .decode import crf_decoding        #DEFINE_ALIAS
# from .decode import ctc_greedy_decoder        #DEFINE_ALIAS
# from .decode import dynamic_decode        #DEFINE_ALIAS
from .decode import gather_tree  #DEFINE_ALIAS
# from .input import Input        #DEFINE_ALIAS
from .layer.activation import ELU
from .layer.activation import GELU
from .layer.activation import Tanh
from .layer.activation import Hardshrink
from .layer.activation import Hardtanh
from .layer.activation import PReLU
from .layer.activation import ReLU
from .layer.activation import ReLU6  #DEFINE_ALIAS
from .layer.activation import SELU  #DEFINE_ALIAS
from .layer.activation import LeakyReLU  #DEFINE_ALIAS
from .layer.activation import Sigmoid  #DEFINE_ALIAS
from .layer.activation import LogSigmoid
from .layer.activation import Softmax  #DEFINE_ALIAS
from .layer.activation import Softplus  #DEFINE_ALIAS
from .layer.activation import Softshrink  #DEFINE_ALIAS
from .layer.activation import Softsign  #DEFINE_ALIAS
from .layer.activation import Tanhshrink  #DEFINE_ALIAS
from .layer.activation import LogSoftmax  #DEFINE_ALIAS
from .layer.activation import HSigmoid  #DEFINE_ALIAS
from .layer.common import BilinearTensorProduct  #DEFINE_ALIAS
from .layer.common import Pool2D  #DEFINE_ALIAS
from .layer.common import Pad2D  #DEFINE_ALIAS
from .layer.common import ReflectionPad1d  #DEFINE_ALIAS
from .layer.common import ReplicationPad1d  #DEFINE_ALIAS
from .layer.common import ConstantPad1d  #DEFINE_ALIAS
from .layer.common import ReflectionPad2d  #DEFINE_ALIAS
from .layer.common import ReplicationPad2d  #DEFINE_ALIAS
from .layer.common import ConstantPad2d  #DEFINE_ALIAS
from .layer.common import ZeroPad2d  #DEFINE_ALIAS
from .layer.common import ReplicationPad3d  #DEFINE_ALIAS
from .layer.common import ConstantPad3d  #DEFINE_ALIAS
from .layer.common import CosineSimilarity  #DEFINE_ALIAS
from .layer.common import Embedding  #DEFINE_ALIAS
from .layer.common import Linear  #DEFINE_ALIAS
from .layer.common import Flatten  #DEFINE_ALIAS
from .layer.common import UpSample  #DEFINE_ALIAS
from .layer.common import Dropout  #DEFINE_ALIAS
from .layer.common import Dropout2D  #DEFINE_ALIAS
from .layer.common import Dropout3D  #DEFINE_ALIAS
from .layer.pooling import AdaptiveAvgPool2d  #DEFINE_ALIAS
from .layer.pooling import AdaptiveAvgPool3d  #DEFINE_ALIAS
from .layer.conv import Conv2D  #DEFINE_ALIAS
from .layer.conv import Conv2DTranspose  #DEFINE_ALIAS
from .layer.conv import Conv3D  #DEFINE_ALIAS
from .layer.conv import Conv3DTranspose  #DEFINE_ALIAS
# from .layer.conv import TreeConv        #DEFINE_ALIAS
# from .layer.conv import Conv1D        #DEFINE_ALIAS
from .layer.extension import RowConv  #DEFINE_ALIAS
# from .layer.learning_rate import CosineDecay        #DEFINE_ALIAS
# from .layer.learning_rate import ExponentialDecay        #DEFINE_ALIAS
# from .layer.learning_rate import InverseTimeDecay        #DEFINE_ALIAS
# from .layer.learning_rate import NaturalExpDecay        #DEFINE_ALIAS
# from .layer.learning_rate import NoamDecay        #DEFINE_ALIAS
# from .layer.learning_rate import PiecewiseDecay        #DEFINE_ALIAS
# from .layer.learning_rate import PolynomialDecay        #DEFINE_ALIAS
# from .layer.loss import NCELoss        #DEFINE_ALIAS
from .layer.loss import BCEWithLogitsLoss  #DEFINE_ALIAS
from .layer.loss import CrossEntropyLoss  #DEFINE_ALIAS
from .layer.loss import MSELoss  #DEFINE_ALIAS
from .layer.loss import L1Loss  #DEFINE_ALIAS
from .layer.loss import NLLLoss  #DEFINE_ALIAS
from .layer.loss import BCELoss  #DEFINE_ALIAS
from .layer.loss import KLDivLoss  #DEFINE_ALIAS
from .layer.loss import MarginRankingLoss  #DEFINE_ALIAS
from .layer.loss import CTCLoss  #DEFINE_ALIAS
from .layer.loss import SmoothL1Loss  #DEFINE_ALIAS
from .layer.norm import BatchNorm  #DEFINE_ALIAS
from .layer.norm import SyncBatchNorm  #DEFINE_ALIAS
from .layer.norm import GroupNorm  #DEFINE_ALIAS
from .layer.norm import LayerNorm  #DEFINE_ALIAS
from .layer.norm import SpectralNorm  #DEFINE_ALIAS
from .layer.norm import InstanceNorm  #DEFINE_ALIAS
# from .layer.rnn import RNNCell        #DEFINE_ALIAS
# from .layer.rnn import GRUCell        #DEFINE_ALIAS
# from .layer.rnn import LSTMCell        #DEFINE_ALIAS
from .layer.distance import PairwiseDistance  #DEFINE_ALIAS

from .layer import loss  #DEFINE_ALIAS
from .layer import conv  #DEFINE_ALIAS
from ..fluid.dygraph.layers import Layer  #DEFINE_ALIAS
from ..fluid.dygraph.container import LayerList, ParameterList, Sequential  #DEFINE_ALIAS
