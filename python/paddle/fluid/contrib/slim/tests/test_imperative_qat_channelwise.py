#   copyright (c) 2018 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

from __future__ import print_function

import os
import numpy as np
import random
import unittest
import logging

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.log_helper import get_logger

from test_imperative_qat import TestImperativeQat

paddle.enable_static()

os.environ["CPU_NUM"] = "1"
if core.is_compiled_with_cuda():
    fluid.set_flags({"FLAGS_cudnn_deterministic": True})

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class TestImperativeQatChannelWise(TestImperativeQat):
    def set_vars(self):
        self.weight_quantize_type = 'channel_wise_abs_max'
        self.activation_quantize_type = 'moving_average_abs_max'
        print('weight_quantize_type', self.weight_quantize_type)


if __name__ == '__main__':
    unittest.main()
