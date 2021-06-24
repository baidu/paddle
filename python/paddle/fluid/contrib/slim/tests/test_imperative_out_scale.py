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
import warnings

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid import core
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import ImperativeQuantAware
from paddle.fluid.dygraph.container import Sequential
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
from paddle.nn.layer import ReLU, LeakyReLU, Sigmoid, Softmax, PReLU
from paddle.nn import Linear, Conv2D, Softmax, BatchNorm2D, MaxPool2D
from paddle.fluid.log_helper import get_logger
from paddle.fluid.dygraph import nn

from imperative_test_utils import fix_model_dict, train_lenet, ImperativeLenet

paddle.enable_static()

os.environ["CPU_NUM"] = "1"
if core.is_compiled_with_cuda():
    fluid.set_flags({"FLAGS_cudnn_deterministic": True})

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


def get_vaild_warning_num(warning, w):
    num = 0
    for i in range(len(w)):
        if warning in str(w[i].message):
            num += 1
    return num


class ImperativeLenet(fluid.dygraph.Layer):
    def __init__(self, num_classes=10):
        super(ImperativeLenet, self).__init__()
        conv2d_w1_attr = fluid.ParamAttr(name="conv2d_w_1")
        conv2d_w2_attr = fluid.ParamAttr(name="conv2d_w_2")
        fc_w1_attr = fluid.ParamAttr(name="fc_w_1")
        fc_w2_attr = fluid.ParamAttr(name="fc_w_2")
        fc_w3_attr = fluid.ParamAttr(name="fc_w_3")
        conv2d_b2_attr = fluid.ParamAttr(name="conv2d_b_2")
        fc_b1_attr = fluid.ParamAttr(name="fc_b_1")
        fc_b2_attr = fluid.ParamAttr(name="fc_b_2")
        fc_b3_attr = fluid.ParamAttr(name="fc_b_3")
        self.features = Sequential(
            Conv2D(
                in_channels=1,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=conv2d_w1_attr,
                bias_attr=False),
            BatchNorm2D(6),
            ReLU(),
            MaxPool2D(
                kernel_size=2, stride=2),
            Conv2D(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
                weight_attr=conv2d_w2_attr,
                bias_attr=conv2d_b2_attr),
            BatchNorm2D(16),
            PReLU(),
            MaxPool2D(
                kernel_size=2, stride=2))

        self.fc = Sequential(
            Linear(
                in_features=400,
                out_features=120,
                weight_attr=fc_w1_attr,
                bias_attr=fc_b1_attr),
            LeakyReLU(),
            Linear(
                in_features=120,
                out_features=84,
                weight_attr=fc_w2_attr,
                bias_attr=fc_b2_attr),
            Sigmoid(),
            Linear(
                in_features=84,
                out_features=num_classes,
                weight_attr=fc_w3_attr,
                bias_attr=fc_b3_attr),
            Softmax())

    def forward(self, inputs):
        x = self.features(inputs)

        x = fluid.layers.flatten(x, 1)
        x = self.fc(x)
        return x


class TestImperativeOutSclae(unittest.TestCase):
    def test_out_scale_acc(self):
        seed = 1000
        lr = 0.001

        weight_quantize_type = 'abs_max'
        activation_quantize_type = 'moving_average_abs_max'
        imperative_out_scale = ImperativeQuantAware(
            weight_quantize_type=weight_quantize_type,
            activation_quantize_type=activation_quantize_type)

        with fluid.dygraph.guard():
            np.random.seed(seed)
            fluid.default_main_program().random_seed = seed
            fluid.default_startup_program().random_seed = seed

            lenet = ImperativeLenet()
            lenet = fix_model_dict(lenet)
            imperative_out_scale.quantize(lenet)

            reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=32, drop_last=True)
            adam = AdamOptimizer(
                learning_rate=lr, parameter_list=lenet.parameters())
            loss_list = train_lenet(lenet, reader, adam)
            lenet.eval()

        param_save_path = "test_save_quantized_model/lenet.pdparams"
        save_dict = lenet.state_dict()
        paddle.save(save_dict, param_save_path)

        save_path = "./dynamic_outscale_infer_model/lenet"
        imperative_out_scale.save_quantized_model(
            layer=lenet,
            path=save_path,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 1, 28, 28], dtype='float32')
            ])

        for i in range(len(loss_list) - 1):
            self.assertTrue(
                loss_list[i] > loss_list[i + 1],
                msg='Failed to do the imperative qat.')


class TestSaveQuanztizedModelFromCheckPoint(unittest.TestCase):
    def test_save_quantized_model(self):
        lr = 0.001

        load_param_path = "test_save_quantized_model/lenet.pdparams"
        save_path = "./dynamic_outscale_infer_model_from_checkpoint/lenet"

        weight_quantize_type = 'abs_max'
        activation_quantize_type = 'moving_average_abs_max'
        imperative_out_scale = ImperativeQuantAware(
            weight_quantize_type=weight_quantize_type,
            activation_quantize_type=activation_quantize_type)

        with fluid.dygraph.guard():
            lenet = ImperativeLenet()
            load_dict = paddle.load(load_param_path)
            imperative_out_scale.quantize(lenet)
            lenet.set_dict(load_dict)

            reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=32, drop_last=True)
            adam = AdamOptimizer(
                learning_rate=lr, parameter_list=lenet.parameters())
            loss_list = train_lenet(lenet, reader, adam)
            lenet.eval()

        imperative_out_scale.save_quantized_model(
            layer=lenet,
            path=save_path,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 1, 28, 28], dtype='float32')
            ])

        for i in range(len(loss_list) - 1):
            self.assertTrue(
                loss_list[i] > loss_list[i + 1],
                msg='Failed to do the imperative qat.')


if __name__ == '__main__':
    unittest.main()
