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
from paddle.fluid.contrib.slim.quantization import *
from paddle.fluid.log_helper import get_logger
from paddle.dataset.common import download

from imperative_test_utils import fix_model_dict, ImperativeLenet

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class TestImperativePTQ(unittest.TestCase):
    """
    """

    @classmethod
    def setUpClass(cls):
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        cls.root_path = os.path.join(os.getcwd(), "imperative_ptq_" + timestamp)
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
        self.ptq = ImperativePTQ(default_ptq_config)

        self.batch_num = 10
        self.batch_size = 10
        self.eval_acc_top1 = 0.99

        self.gt_thresholds = {
            'conv2d_0': [[1.0], [0.37673383951187134], [0.10933732241392136]],
            'batch_norm2d_0': [[0.37673383951187134], [0.44249194860458374]],
            're_lu_0': [[0.44249194860458374], [0.25804123282432556]],
            'max_pool2d_0': [[0.25804123282432556], [0.25804123282432556]],
            'linear_0':
            [[1.7058950662612915], [14.405526161193848], [0.4373355209827423]],
            'add_0': [[1.7058950662612915, 0.0], [1.7058950662612915]],
        }

    def model_train(self, model, train_reader, max_step=-1):
        model.train()
        adam = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=model.parameters())

        for batch_id, data in enumerate(train_reader()):
            x_data = np.array([x[0].reshape(1, 28, 28)
                               for x in data]).astype('float32')
            y_data = np.array(
                [x[1] for x in data]).astype('int64').reshape(-1, 1)

            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)

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

            if max_step > 0 and batch_id > max_step:  # For shortening CI time
                break

    def model_test(self, model, batch_num=-1, batch_size=8):
        model.eval()

        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)

        eval_acc_top1_list = []
        for batch_id, data in enumerate(test_reader()):
            x_data = np.array([x[0].reshape(1, 28, 28)
                               for x in data]).astype('float32')
            y_data = np.array(
                [x[1] for x in data]).astype('int64').reshape(-1, 1)

            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)

            out = model(img)
            acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
            acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

            if batch_id % 100 == 0:
                eval_acc_top1_list.append(float(acc_top1.numpy()))
                _logger.info("Test | At step {}: acc1 = {:}, acc5 = {:}".format(
                    batch_id, acc_top1.numpy(), acc_top5.numpy()))

            if batch_num > 0 and batch_id + 1 >= batch_num:
                break

        eval_acc_top1 = sum(eval_acc_top1_list) / len(eval_acc_top1_list)

        return eval_acc_top1

    def check_thresholds(self, model):
        check_num = 0
        for name, layer in model.named_sublayers():
            layer_name = layer.full_name()
            if layer_name in self.gt_thresholds:
                ref_val = self.gt_thresholds[layer_name]
                assert hasattr(layer, '_quant_config')

                quant_config = layer._quant_config
                in_val = quant_config.in_act_quantizer.thresholds
                out_val = quant_config.out_act_quantizer.thresholds
                wt_val = quant_config.wt_quantizer.thresholds
                check_num += 1

                self.assertTrue(
                    np.allclose(
                        ref_val[0], in_val, atol=1e-3),
                    "%s | The thresholds(%s) is different "
                    "from the ground truth(%s)." %
                    (layer_name, str(in_val), str(ref_val[0])))
                self.assertTrue(
                    np.allclose(
                        ref_val[1], out_val, atol=1e-3),
                    "%s | The thresholds(%s) is different "
                    "from the ground truth(%s)." %
                    (layer_name, str(out_val), str(ref_val[1])))
                if len(ref_val) > 2 and ref_val[2] != []:
                    self.assertTrue(
                        np.allclose(
                            ref_val[2], wt_val, atol=1e-3),
                        "%s | The thresholds(%s) is different "
                        "from the ground truth(%s)." %
                        (layer_name, str(wt_val), str(ref_val[2])))

        self.assertTrue(check_num == len(self.gt_thresholds))

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

            self.ptq.quantize(model, inplace=True)

            acc_top1 = self.model_test(model, self.batch_num, self.batch_size)
            print('acc_top1: %s' % acc_top1)
            self.assertTrue(
                acc_top1 > self.eval_acc_top1,
                msg="The test acc {%f} is less than {%f}." %
                (acc_top1, self.eval_acc_top1))

        self.ptq.convert(model)

        self.check_thresholds(model)

        input_spec = [
            paddle.static.InputSpec(
                shape=[None, 1, 28, 28], dtype='float32')
        ]
        paddle.jit.save(layer=model, path=self.save_path, input_spec=input_spec)
        print('Quantized model saved in {%s}' % self.save_path)

        end_time = time.time()
        print("total time: %ss" % (end_time - start_time))


class TestImperativePTQHist(TestImperativePTQ):
    """
    """

    def set_vars(self):
        config = PTQConfig(HistQuantizer(), AbsmaxQuantizer())
        self.ptq = ImperativePTQ(config)

        self.batch_num = 10
        self.batch_size = 10
        self.eval_acc_top1 = 0.99

        self.gt_thresholds = {
            'conv2d_0':
            [[0.99853515625], [0.35732391771364225], [0.10933732241392136]],
            'batch_norm2d_0': [[0.35732391771364225], [0.4291427868761275]],
            're_lu_0': [[0.4291427868761275], [0.2359918110742001]],
            'max_pool2d_0': [[0.2359918110742001], [0.25665526917146053]],
            'linear_0':
            [[1.7037603475152991], [14.395224522473026], [0.4373355209827423]],
            'add_0': [[1.7037603475152991, 0.0], [1.7037603475152991]],
        }


class TestImperativePTQKL(TestImperativePTQ):
    """
    """

    def set_vars(self):
        config = PTQConfig(KLQuantizer(), PerChannelAbsmaxQuantizer())
        self.ptq = ImperativePTQ(config)

        self.batch_num = 10
        self.batch_size = 10
        self.eval_acc_top1 = 0.99

        conv2d_1_wt_thresholds = [
            0.18116560578346252, 0.17079241573810577, 0.1702047884464264,
            0.179476797580719, 0.1454375684261322, 0.22981858253479004
        ]
        self.gt_thresholds = {
            'conv2d_0': [[0.99267578125], [0.37695913558696836]],
            'conv2d_1': [[0.19189296757394914], [0.24514256547263358],
                         [conv2d_1_wt_thresholds]],
            'batch_norm2d_0': [[0.37695913558696836], [0.27462541429440535]],
            're_lu_0': [[0.27462541429440535], [0.19189296757394914]],
            'max_pool2d_0': [[0.19189296757394914], [0.19189296757394914]],
            'linear_0': [[1.2839322163611087], [8.957185942414352]],
            'add_0': [[1.2839322163611087, 0.0], [1.2839322163611087]],
        }


if __name__ == '__main__':
    unittest.main()
