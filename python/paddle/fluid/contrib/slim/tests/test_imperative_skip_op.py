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
import paddle.fluid.layers as layers
from paddle.fluid import core
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.contrib.slim.quantization import ImperativeQuantAware
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
from paddle.nn.layer import ReLU, LeakyReLU, Sigmoid, Softmax, ReLU6
from paddle.nn import Linear, Conv2D, Softmax, BatchNorm
from paddle.fluid.dygraph.nn import Pool2D
from paddle.fluid.log_helper import get_logger

from imperative_test_utils import fix_model_dict, train_lenet, ImperativeLenetWithSkipQuant

os.environ["CPU_NUM"] = "1"
if core.is_compiled_with_cuda():
    fluid.set_flags({"FLAGS_cudnn_deterministic": True})

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class TestImperativeOutSclae(unittest.TestCase):
    def test_out_scale_acc(self):
        seed = 1000
        lr = 0.1

        qat = ImperativeQuantAware()

        np.random.seed(seed)
        reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=512, drop_last=True)

        lenet = ImperativeLenetWithSkipQuant()
        lenet = fix_model_dict(lenet)
        qat.quantize(lenet)

        adam = AdamOptimizer(
            learning_rate=lr, parameter_list=lenet.parameters())
        dynamic_loss_rec = []
        lenet.train()
        loss_list = train_lenet(lenet, reader, adam)

        lenet.eval()

        path = "./save_dynamic_quant_infer_model/lenet"
        save_dir = "./save_dynamic_quant_infer_model"

        qat.save_quantized_model(
            layer=lenet,
            path=path,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 1, 28, 28], dtype='float32')
            ])

        paddle.enable_static()

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)

        [inference_program, feed_target_names, fetch_targets] = (
            fluid.io.load_inference_model(
                dirname=save_dir,
                executor=exe,
                model_filename="lenet" + INFER_MODEL_SUFFIX,
                params_filename="lenet" + INFER_PARAMS_SUFFIX))
        model_ops = inference_program.global_block().ops

        conv2d_count, matmul_count = 0, 0
        conv2d_skip_count, matmul_skip_count = 0, 0
        for i, op in enumerate(model_ops):
            if op.type == 'conv2d':
                if op.has_attr("skip_quant"):
                    conv2d_skip_count += 1
                if conv2d_count > 0:
                    self.assertTrue(
                        'fake_quantize_dequantize' in model_ops[i - 1].type)
                else:
                    self.assertTrue(
                        'fake_quantize_dequantize' not in model_ops[i - 1].type)
                conv2d_count += 1

            if op.type == 'matmul':
                if op.has_attr("skip_quant"):
                    matmul_skip_count += 1
                if matmul_count > 0:
                    self.assertTrue(
                        'fake_quantize_dequantize' in model_ops[i - 1].type)
                else:
                    self.assertTrue(
                        'fake_quantize_dequantize' not in model_ops[i - 1].type)
                matmul_count += 1

        self.assertTrue(conv2d_skip_count == 1)
        self.assertTrue(matmul_skip_count == 1)


if __name__ == '__main__':
    unittest.main()
