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

# TODO: define activation functions of neural network

from . import activation
from . import loss
from . import conv
from . import extension
from . import activation
from . import norm
from . import rnn
from . import vision
from . import distance
from . import transformer

from .activation import *
from .loss import *
from .conv import *
from .extension import *
from .activation import *
from .norm import *
from .rnn import *
from .vision import *

from .transformer import *
# from .activation import PReLU        #DEFINE_ALIAS
from .activation import ReLU  #DEFINE_ALIAS
from .activation import LeakyReLU  #DEFINE_ALIAS
from .activation import Sigmoid  #DEFINE_ALIAS
# from .activation import Softmax        #DEFINE_ALIAS
from .activation import LogSoftmax  #DEFINE_ALIAS
from .common import BilinearTensorProduct  #DEFINE_ALIAS
from .common import Bilinear  #DEFINE_ALIAS
from .common import Pool2D  #DEFINE_ALIAS
from .common import Pad2D  #DEFINE_ALIAS
from .common import ReflectionPad1d  #DEFINE_ALIAS
from .common import ReplicationPad1d  #DEFINE_ALIAS
from .common import ConstantPad1d  #DEFINE_ALIAS
from .common import ReflectionPad2d  #DEFINE_ALIAS
from .common import ReplicationPad2d  #DEFINE_ALIAS
from .common import ConstantPad2d  #DEFINE_ALIAS
from .common import ZeroPad2d  #DEFINE_ALIAS
from .common import ReplicationPad3d  #DEFINE_ALIAS
from .common import ConstantPad3d  #DEFINE_ALIAS
from .common import CosineSimilarity  #DEFINE_ALIAS
from .common import Embedding  #DEFINE_ALIAS
from .common import Linear  #DEFINE_ALIAS
from .common import Flatten  #DEFINE_ALIAS
from .common import Upsample  #DEFINE_ALIAS
from .common import UpsamplingNearest2d  #DEFINE_ALIAS
from .common import UpsamplingBilinear2d  #DEFINE_ALIAS
from .common import Dropout  #DEFINE_ALIAS
from .common import Dropout2d  #DEFINE_ALIAS
from .common import Dropout3d  #DEFINE_ALIAS
from .common import AlphaDropout  #DEFINE_ALIAS
from .pooling import AvgPool1d  #DEFINE_ALIAS
from .pooling import AvgPool2d  #DEFINE_ALIAS
from .pooling import AvgPool3d  #DEFINE_ALIAS
from .pooling import MaxPool1d  #DEFINE_ALIAS
from .pooling import MaxPool2d  #DEFINE_ALIAS
from .pooling import MaxPool3d  #DEFINE_ALIAS
from .pooling import AdaptiveAvgPool1d  #DEFINE_ALIAS
from .pooling import AdaptiveAvgPool2d  #DEFINE_ALIAS
from .pooling import AdaptiveAvgPool3d  #DEFINE_ALIAS
from .pooling import AdaptiveMaxPool1d  #DEFINE_ALIAS
from .pooling import AdaptiveMaxPool2d  #DEFINE_ALIAS
from .pooling import AdaptiveMaxPool3d  #DEFINE_ALIAS
from .conv import Conv1d  #DEFINE_ALIAS
from .conv import Conv2d  #DEFINE_ALIAS
from .conv import Conv3d  #DEFINE_ALIAS
from .conv import ConvTranspose1d  #DEFINE_ALIAS
from .conv import ConvTranspose2d  #DEFINE_ALIAS
from .conv import ConvTranspose3d  #DEFINE_ALIAS
# from .conv import TreeConv        #DEFINE_ALIAS
# from .conv import Conv1D        #DEFINE_ALIAS
from .extension import RowConv  #DEFINE_ALIAS
# from .loss import NCELoss        #DEFINE_ALIAS
from .loss import BCEWithLogitsLoss  #DEFINE_ALIAS
from .loss import CrossEntropyLoss  #DEFINE_ALIAS
from .loss import MSELoss  #DEFINE_ALIAS
from .loss import L1Loss  #DEFINE_ALIAS
from .loss import NLLLoss  #DEFINE_ALIAS
from .loss import BCELoss  #DEFINE_ALIAS
from .loss import KLDivLoss  #DEFINE_ALIAS
from .loss import MarginRankingLoss  #DEFINE_ALIAS
from .loss import CTCLoss  #DEFINE_ALIAS
from .loss import SmoothL1Loss  #DEFINE_ALIAS
from .norm import BatchNorm  #DEFINE_ALIAS
from .norm import SyncBatchNorm  #DEFINE_ALIAS
from .norm import GroupNorm  #DEFINE_ALIAS
from .norm import LayerNorm  #DEFINE_ALIAS
from .norm import SpectralNorm  #DEFINE_ALIAS
#from .norm import InstanceNorm  #DEFINE_ALIAS
from .norm import LocalResponseNorm  #DEFINE_ALIAS
# from .rnn import RNNCell        #DEFINE_ALIAS
# from .rnn import GRUCell        #DEFINE_ALIAS
# from .rnn import LSTMCell        #DEFINE_ALIAS

from .vision import PixelShuffle  #DEFINE_ALIAS
from .distance import PairwiseDistance  #DEFINE_ALIAS
