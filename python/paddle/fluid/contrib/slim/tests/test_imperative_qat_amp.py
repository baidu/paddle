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
from paddle.fluid.contrib.slim.quantization import ImperativeQuantAware
from paddle.fluid.log_helper import get_logger
from paddle.dataset.common import download

from imperative_test_utils import fix_model_dict, ImperativeLenet

os.environ["CPU_NUM"] = "1"
if paddle.is_compiled_with_cuda():
    fluid.set_flags({"FLAGS_cudnn_deterministic": True})

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class TestImperativeQatAmp(unittest.TestCase):
    """
    Test the combination of qat and amp.
    """

    @classmethod
    def setUpClass(cls):
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        cls.root_path = os.path.join(os.getcwd(),
                                     "imperative_qat_amp_" + timestamp)
        cls.save_path = os.path.join(cls.root_path, "model")

        cls.download_path = 'dygraph_int8/download'
        cls.cache_folder = os.path.expanduser('~/.cache/paddle/dataset/' +
                                              cls.download_path)

        cls.lenet_url = "https://paddle-inference-dist.cdn.bcebos.com/int8/unittest_model_data/lenet_pretrained.tar.gz"
        cls.lenet_md5 = "953b802fb73b52fae42896e3c24f0afb"

        seed = 1
        np.random.seed(seed)
        paddle.static.default_main_program().random_seed = seed
        paddle.static.default_startup_program().random_seed = seed

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(cls.root_path)
        except Exception as e:
            print("Failed to delete {} due to {}".format(cls.root_path, str(e)))

    def cache_unzipping(self, target_folder, zip_path):
        if not os.path.exists(target_folder):
            cmd = 'mkdir {0} && tar xf {1} -C {0}'.format(target_folder,
                                                          zip_path)
            os.system(cmd)

    def download_model(self, data_url, data_md5, folder_name):
        download(data_url, self.download_path, data_md5)
        file_name = data_url.split('/')[-1]
        zip_path = os.path.join(self.cache_folder, file_name)
        print('Data is downloaded at {0}'.format(zip_path))

        data_cache_folder = os.path.join(self.cache_folder, folder_name)
        self.cache_unzipping(data_cache_folder, zip_path)
        return data_cache_folder

    def set_vars(self):
        self.qat = ImperativeQuantAware()

        self.train_batch_num = 30
        self.train_batch_size = 32
        self.test_batch_num = 100
        self.test_batch_size = 32
        self.eval_acc_top1 = 0.99

    def model_train(self, model, batch_num=-1, batch_size=32, use_amp=False):
        model.train()

        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=batch_size)
        adam = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=model.parameters())
        scaler = paddle.amp.GradScaler(init_loss_scaling=500)

        for batch_id, data in enumerate(train_reader()):
            x_data = np.array([x[0].reshape(1, 28, 28)
                               for x in data]).astype('float32')
            y_data = np.array(
                [x[1] for x in data]).astype('int64').reshape(-1, 1)

            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)

            if use_amp:
                with paddle.amp.auto_cast():
                    out = model(img)
                    acc = fluid.layers.accuracy(out, label)
                    loss = fluid.layers.cross_entropy(out, label)
                    avg_loss = fluid.layers.mean(loss)
                scaled_loss = scaler.scale(avg_loss)
                scaled_loss.backward()

                scaler.minimize(adam, scaled_loss)
                adam.clear_gradients()
            else:
                out = model(img)
                acc = fluid.layers.accuracy(out, label)
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()

                adam.minimize(avg_loss)
                model.clear_gradients()

            if batch_id % 100 == 0:
                _logger.info("Train | step {}: loss = {:}, acc= {:}".format(
                    batch_id, avg_loss.numpy(), acc.numpy()))

            if batch_num > 0 and batch_id + 1 >= batch_num:
                break

    def model_test(self, model, batch_num=-1, batch_size=32, use_amp=False):
        model.eval()

        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)

        acc_top1_list = []
        for batch_id, data in enumerate(test_reader()):
            x_data = np.array([x[0].reshape(1, 28, 28)
                               for x in data]).astype('float32')
            y_data = np.array(
                [x[1] for x in data]).astype('int64').reshape(-1, 1)

            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)

            with paddle.amp.auto_cast(use_amp):
                out = model(img)
                acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
                acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

            acc_top1_list.append(float(acc_top1.numpy()))
            if batch_id % 100 == 0:
                _logger.info("Test | At step {}: acc1 = {:}, acc5 = {:}".format(
                    batch_id, acc_top1.numpy(), acc_top5.numpy()))

            if batch_num > 0 and batch_id + 1 >= batch_num:
                break

        acc_top1 = sum(acc_top1_list) / len(acc_top1_list)
        return acc_top1

    def test_ptq(self):
        start_time = time.time()

        self.set_vars()

        params_path = self.download_model(self.lenet_url, self.lenet_md5,
                                          "lenet")
        params_path += "/lenet_pretrained/lenet.pdparams"

        with fluid.dygraph.guard():
            model = ImperativeLenet()
            model_state_dict = paddle.load(params_path)
            model.set_state_dict(model_state_dict)

            _logger.info("Test fp32 model")
            fp32_acc_top1 = self.model_test(model, self.test_batch_num,
                                            self.test_batch_size)

            self.qat.quantize(model)

            use_amp = True
            self.model_train(model, self.train_batch_num, self.train_batch_size,
                             use_amp)

            _logger.info("Test int8 model")
            int8_acc_top1 = self.model_test(model, self.test_batch_num,
                                            self.test_batch_size, use_amp)

            _logger.info('fp32_acc_top1: %f, int8_acc_top1: %f' %
                         (fp32_acc_top1, int8_acc_top1))
            self.assertTrue(
                int8_acc_top1 > fp32_acc_top1 - 0.01,
                msg='fp32_acc_top1: %f, int8_acc_top1: %f' %
                (fp32_acc_top1, int8_acc_top1))

        input_spec = [
            paddle.static.InputSpec(
                shape=[None, 1, 28, 28], dtype='float32')
        ]
        paddle.jit.save(layer=model, path=self.save_path, input_spec=input_spec)
        print('Quantized model saved in {%s}' % self.save_path)

        end_time = time.time()
        print("total time: %ss" % (end_time - start_time))


if __name__ == '__main__':
    unittest.main()
