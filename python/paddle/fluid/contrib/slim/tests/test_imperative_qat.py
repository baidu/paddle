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
import shutil
import time
import unittest
import logging

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.contrib.slim.quantization import ImperativeQuantAware
from paddle.fluid.dygraph.container import Sequential
from paddle.nn import Linear, Conv2D, Softmax
from paddle.fluid.log_helper import get_logger
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
from paddle.fluid.contrib.slim.quantization.imperative.quant_nn import QuantizedConv2D

from imperative_test_utils import fix_model_dict, ImperativeLenet

paddle.enable_static()

os.environ["CPU_NUM"] = "1"
if core.is_compiled_with_cuda():
    fluid.set_flags({"FLAGS_cudnn_deterministic": True})

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class TestImperativeQat(unittest.TestCase):
    """
    QAT = quantization-aware training
    """

    @classmethod
    def setUpClass(cls):
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        cls.root_path = os.path.join(os.getcwd(), "imperative_qat_" + timestamp)
        cls.save_path = os.path.join(cls.root_path, "lenet")

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(cls.root_path)
        except Exception as e:
            print("Failed to delete {} due to {}".format(cls.root_path, str(e)))

    def set_vars(self):
        self.weight_quantize_type = 'abs_max'
        self.activation_quantize_type = 'moving_average_abs_max'
        print('weight_quantize_type', self.weight_quantize_type)

    def test_qat(self):
        self.set_vars()

        imperative_qat = ImperativeQuantAware(
            weight_quantize_type=self.weight_quantize_type,
            activation_quantize_type=self.activation_quantize_type)

        with fluid.dygraph.guard():
            # For CI coverage
            conv1 = Conv2D(
                in_channels=3,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode='replicate')
            quant_conv1 = QuantizedConv2D(conv1)
            data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
            quant_conv1(fluid.dygraph.to_variable(data))

            seed = 1
            np.random.seed(seed)
            fluid.default_main_program().random_seed = seed
            fluid.default_startup_program().random_seed = seed

            lenet = ImperativeLenet()
            lenet = fix_model_dict(lenet)
            imperative_qat.quantize(lenet)
            adam = AdamOptimizer(
                learning_rate=0.001, parameter_list=lenet.parameters())

            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=32, drop_last=True)
            test_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=32)

            epoch_num = 1
            for epoch in range(epoch_num):
                lenet.train()
                for batch_id, data in enumerate(train_reader()):
                    x_data = np.array([x[0].reshape(1, 28, 28)
                                       for x in data]).astype('float32')
                    y_data = np.array(
                        [x[1] for x in data]).astype('int64').reshape(-1, 1)

                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)
                    out = lenet(img)
                    acc = fluid.layers.accuracy(out, label)
                    loss = fluid.layers.cross_entropy(out, label)
                    avg_loss = fluid.layers.mean(loss)
                    avg_loss.backward()
                    adam.minimize(avg_loss)
                    lenet.clear_gradients()
                    if batch_id % 100 == 0:
                        _logger.info(
                            "Train | At epoch {} step {}: loss = {:}, acc= {:}".
                            format(epoch, batch_id,
                                   avg_loss.numpy(), acc.numpy()))
                    if batch_id == 500:  # For shortening CI time
                        break

                lenet.eval()
                eval_acc_top1_list = []
                for batch_id, data in enumerate(test_reader()):
                    x_data = np.array([x[0].reshape(1, 28, 28)
                                       for x in data]).astype('float32')
                    y_data = np.array(
                        [x[1] for x in data]).astype('int64').reshape(-1, 1)

                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)

                    out = lenet(img)
                    acc_top1 = fluid.layers.accuracy(
                        input=out, label=label, k=1)
                    acc_top5 = fluid.layers.accuracy(
                        input=out, label=label, k=5)

                    if batch_id % 100 == 0:
                        eval_acc_top1_list.append(float(acc_top1.numpy()))
                        _logger.info(
                            "Test | At epoch {} step {}: acc1 = {:}, acc5 = {:}".
                            format(epoch, batch_id,
                                   acc_top1.numpy(), acc_top5.numpy()))

                # check eval acc
                eval_acc_top1 = sum(eval_acc_top1_list) / len(
                    eval_acc_top1_list)
                print('eval_acc_top1', eval_acc_top1)
                self.assertTrue(
                    eval_acc_top1 > 0.9,
                    msg="The test acc {%f} is less than 0.9." % eval_acc_top1)

            # test the correctness of `paddle.jit.save`
            data = next(test_reader())
            test_data = np.array([x[0].reshape(1, 28, 28)
                                  for x in data]).astype('float32')
            test_img = fluid.dygraph.to_variable(test_data)
            lenet.eval()
            before_save = lenet(test_img)

        # save inference quantized model
        imperative_qat.save_quantized_model(
            layer=lenet,
            path=self.save_path,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 1, 28, 28], dtype='float32')
            ])
        print('Quantized model saved in {%s}' % self.save_path)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
             dirname=self.root_path,
             executor=exe,
             model_filename="lenet" + INFER_MODEL_SUFFIX,
             params_filename="lenet" + INFER_PARAMS_SUFFIX)
        after_save, = exe.run(inference_program,
                              feed={feed_target_names[0]: test_data},
                              fetch_list=fetch_targets)
        # check
        self.assertTrue(
            np.allclose(after_save, before_save.numpy()),
            msg='Failed to save the inference quantized model.')


if __name__ == '__main__':
    unittest.main()
