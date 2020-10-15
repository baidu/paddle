# copyright (c) 2020 paddlepaddle authors. all rights reserved.
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

from __future__ import division
from __future__ import print_function

import unittest

import os
import numpy as np
import shutil
import tempfile

import paddle
from paddle import fluid
from paddle import to_tensor
from paddle.nn import Conv2d, Linear, ReLU, Sequential, Softmax

from paddle import Model
from paddle.static import InputSpec
from paddle.nn.layer.loss import CrossEntropyLoss
from paddle.metric import Accuracy
from paddle.vision.datasets import MNIST
from paddle.vision.models import LeNet
from paddle.io import DistributedBatchSampler, Dataset
from paddle.hapi.model import prepare_distributed_context
from paddle.fluid.dygraph.jit import declarative
from paddle.fluid.dygraph.dygraph_to_static.program_translator import ProgramTranslator


class LeNetDygraph(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNetDygraph, self).__init__()
        self.num_classes = num_classes
        self.features = Sequential(
            Conv2d(
                1, 6, 3, stride=1, padding=1),
            ReLU(),
            paddle.fluid.dygraph.Pool2D(2, 'max', 2),
            Conv2d(
                6, 16, 5, stride=1, padding=0),
            ReLU(),
            paddle.fluid.dygraph.Pool2D(2, 'max', 2))

        if num_classes > 0:
            self.fc = Sequential(
                Linear(400, 120), Linear(120, 84), Linear(84, 10))

    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = fluid.layers.flatten(x, 1)
            x = self.fc(x)
        return x


class MnistDataset(MNIST):
    def __init__(self, mode, return_label=True, sample_num=None):
        super(MnistDataset, self).__init__(mode=mode)
        self.return_label = return_label
        if sample_num:
            self.images = self.images[:sample_num]
            self.labels = self.labels[:sample_num]

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        img = np.reshape(img, [1, 28, 28])
        if self.return_label:
            return img, np.array(self.labels[idx]).astype('int64')
        return img,

    def __len__(self):
        return len(self.images)


def compute_acc(pred, label):
    pred = np.argmax(pred, -1)
    label = np.array(label)
    correct = pred[:, np.newaxis] == label
    return np.sum(correct) / correct.shape[0]


def dynamic_train(model, dataloader):
    optim = fluid.optimizer.Adam(
        learning_rate=0.001, parameter_list=model.parameters())
    model.train()
    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = CrossEntropyLoss(reduction="sum")(outputs, labels)
        avg_loss = fluid.layers.reduce_sum(loss)
        avg_loss.backward()
        optim.minimize(avg_loss)
        model.clear_gradients()


def dynamic_evaluate(model, dataloader):
    with fluid.dygraph.no_grad():
        model.eval()
        cnt = 0
        for inputs, labels in dataloader:
            outputs = model(inputs)

            cnt += (np.argmax(outputs.numpy(), -1)[:, np.newaxis] ==
                    labels.numpy()).astype('int').sum()

    return cnt / len(dataloader.dataset)


@unittest.skipIf(not fluid.is_compiled_with_cuda(),
                 'CPU testing is not supported')
class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not fluid.is_compiled_with_cuda():
            self.skipTest('module not tested when ONLY_CPU compling')
        cls.device = paddle.set_device('gpu')
        fluid.enable_dygraph(cls.device)

        sp_num = 1280
        cls.train_dataset = MnistDataset(mode='train', sample_num=sp_num)
        cls.val_dataset = MnistDataset(mode='test', sample_num=sp_num)
        cls.test_dataset = MnistDataset(
            mode='test', return_label=False, sample_num=sp_num)

        cls.train_loader = fluid.io.DataLoader(
            cls.train_dataset, places=cls.device, batch_size=64)
        cls.val_loader = fluid.io.DataLoader(
            cls.val_dataset, places=cls.device, batch_size=64)
        cls.test_loader = fluid.io.DataLoader(
            cls.test_dataset, places=cls.device, batch_size=64)

        seed = 333
        paddle.manual_seed(seed)
        paddle.framework.random._manual_program_seed(seed)

        dy_lenet = LeNetDygraph()
        cls.init_param = dy_lenet.state_dict()
        dynamic_train(dy_lenet, cls.train_loader)

        cls.acc1 = dynamic_evaluate(dy_lenet, cls.val_loader)

        cls.inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
        cls.labels = [InputSpec([None, 1], 'int64', 'label')]

        cls.save_dir = tempfile.mkdtemp()
        cls.weight_path = os.path.join(cls.save_dir, 'lenet')
        fluid.dygraph.save_dygraph(dy_lenet.state_dict(), cls.weight_path)

        fluid.disable_dygraph()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.save_dir)

    def test_fit_dygraph(self):
        self.fit(True)

    def test_fit_static(self):
        self.fit(False)

    def test_fit_dynamic_with_rank(self):
        self.fit(True, 2, 0)

    def test_fit_static_with_rank(self):
        self.fit(False, 2, 0)

    def test_evaluate_dygraph(self):
        self.evaluate(True)

    def test_evaluate_static(self):
        self.evaluate(False)

    def test_predict_dygraph(self):
        self.predict(True)

    def test_predict_static(self):
        self.predict(False)

    def test_prepare_context(self):
        prepare_distributed_context()

    def fit(self, dynamic, num_replicas=None, rank=None):
        fluid.enable_dygraph(self.device) if dynamic else None
        seed = 333
        paddle.manual_seed(seed)
        paddle.framework.random._manual_program_seed(seed)

        net = LeNet()
        optim_new = fluid.optimizer.Adam(
            learning_rate=0.001, parameter_list=net.parameters())
        model = Model(net, inputs=self.inputs, labels=self.labels)
        model.prepare(
            optim_new,
            loss=CrossEntropyLoss(reduction="sum"),
            metrics=Accuracy())
        model.fit(self.train_dataset, batch_size=64, shuffle=False)

        result = model.evaluate(self.val_dataset, batch_size=64)
        np.testing.assert_allclose(result['acc'], self.acc1)

        train_sampler = DistributedBatchSampler(
            self.train_dataset,
            batch_size=64,
            shuffle=False,
            num_replicas=num_replicas,
            rank=rank)
        val_sampler = DistributedBatchSampler(
            self.val_dataset,
            batch_size=64,
            shuffle=False,
            num_replicas=num_replicas,
            rank=rank)

        train_loader = fluid.io.DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            places=self.device,
            return_list=True)

        val_loader = fluid.io.DataLoader(
            self.val_dataset,
            batch_sampler=val_sampler,
            places=self.device,
            return_list=True)

        model.fit(train_loader, val_loader)
        fluid.disable_dygraph() if dynamic else None

    def evaluate(self, dynamic):
        fluid.enable_dygraph(self.device) if dynamic else None
        model = Model(LeNet(), self.inputs, self.labels)
        model.prepare(metrics=Accuracy())
        model.load(self.weight_path)
        result = model.evaluate(self.val_dataset, batch_size=64)
        np.testing.assert_allclose(result['acc'], self.acc1)

        sampler = DistributedBatchSampler(
            self.val_dataset, batch_size=64, shuffle=False)

        val_loader = fluid.io.DataLoader(
            self.val_dataset,
            batch_sampler=sampler,
            places=self.device,
            return_list=True)

        model.evaluate(val_loader)

        fluid.disable_dygraph() if dynamic else None

    def predict(self, dynamic):
        fluid.enable_dygraph(self.device) if dynamic else None
        model = Model(LeNet(), self.inputs)
        model.prepare()
        model.load(self.weight_path)
        output = model.predict(
            self.test_dataset, batch_size=64, stack_outputs=True)
        np.testing.assert_equal(output[0].shape[0], len(self.test_dataset))

        acc = compute_acc(output[0], self.val_dataset.labels)
        np.testing.assert_allclose(acc, self.acc1)

        sampler = DistributedBatchSampler(
            self.test_dataset, batch_size=64, shuffle=False)

        test_loader = fluid.io.DataLoader(
            self.test_dataset,
            batch_sampler=sampler,
            places=self.device,
            return_list=True)

        model.evaluate(test_loader)

        fluid.disable_dygraph() if dynamic else None


class MyModel(paddle.nn.Layer):
    def __init__(self):
        super(MyModel, self).__init__()
        self._fc = Linear(20, 10)

    def forward(self, x):
        y = self._fc(x)
        return y


class MyDataset(Dataset):
    def __getitem__(self, idx):
        return np.random.random(size=(20,)).astype(np.float32), \
               np.random.randint(0, 10, size=(1,)).astype(np.int64)

    def __len__(self):
        return 40


class TestModelFunction(unittest.TestCase):
    def set_seed(self, seed=1024):
        paddle.manual_seed(seed)
        paddle.framework.random._manual_program_seed(seed)

    def test_train_batch(self, dynamic=True):
        dim = 20
        data = np.random.random(size=(4, dim)).astype(np.float32)
        label = np.random.randint(0, 10, size=(4, 1)).astype(np.int64)

        def get_expect():
            fluid.enable_dygraph(fluid.CPUPlace())
            self.set_seed()
            m = MyModel()
            optim = fluid.optimizer.SGD(learning_rate=0.001,
                                        parameter_list=m.parameters())
            m.train()
            output = m(to_tensor(data))
            loss = CrossEntropyLoss(reduction='sum')(output, to_tensor(label))
            avg_loss = fluid.layers.reduce_sum(loss)
            avg_loss.backward()
            optim.minimize(avg_loss)
            m.clear_gradients()
            fluid.disable_dygraph()
            return avg_loss.numpy()

        ref = get_expect()
        for dynamic in [True, False]:
            device = paddle.set_device('cpu')
            fluid.enable_dygraph(device) if dynamic else None
            self.set_seed()

            net = MyModel()
            optim2 = fluid.optimizer.SGD(learning_rate=0.001,
                                         parameter_list=net.parameters())

            inputs = [InputSpec([None, dim], 'float32', 'x')]
            labels = [InputSpec([None, 1], 'int64', 'label')]
            model = Model(net, inputs, labels)
            model.prepare(optim2, loss=CrossEntropyLoss(reduction="sum"))
            loss, = model.train_batch([data], [label])
            np.testing.assert_allclose(loss.flatten(), ref.flatten())
            fluid.disable_dygraph() if dynamic else None

    def test_test_batch(self):
        dim = 20
        data = np.random.random(size=(4, dim)).astype(np.float32)

        def get_expect():
            fluid.enable_dygraph(fluid.CPUPlace())
            self.set_seed()
            m = MyModel()
            m.eval()
            output = m(to_tensor(data))
            fluid.disable_dygraph()
            return output.numpy()

        ref = get_expect()
        for dynamic in [True, False]:
            device = paddle.set_device('cpu')
            fluid.enable_dygraph(device) if dynamic else None
            self.set_seed()
            net = MyModel()
            inputs = [InputSpec([None, dim], 'float32', 'x')]
            model = Model(net, inputs)
            model.prepare()
            out, = model.test_batch([data])

            np.testing.assert_allclose(out, ref, rtol=1e-6)
            fluid.disable_dygraph() if dynamic else None

    def test_save_load(self):
        path = tempfile.mkdtemp()
        for dynamic in [True, False]:
            device = paddle.set_device('cpu')
            fluid.enable_dygraph(device) if dynamic else None
            net = MyModel()
            inputs = [InputSpec([None, 20], 'float32', 'x')]
            labels = [InputSpec([None, 1], 'int64', 'label')]
            optim = fluid.optimizer.SGD(learning_rate=0.001,
                                        parameter_list=net.parameters())
            model = Model(net, inputs, labels)
            model.prepare(
                optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))
            model.save(path + '/test')
            model.load(path + '/test')
            shutil.rmtree(path)
            fluid.disable_dygraph() if dynamic else None

    def test_dynamic_load(self):
        mnist_data = MnistDataset(mode='train')
        for new_optimizer in [True, False]:
            path = tempfile.mkdtemp()
            paddle.disable_static()
            net = LeNet()
            inputs = [InputSpec([None, 1, 28, 28], 'float32', 'x')]
            labels = [InputSpec([None, 1], 'int64', 'label')]
            if new_optimizer:
                optim = paddle.optimizer.Adam(
                    learning_rate=0.001, parameters=net.parameters())
            else:
                optim = fluid.optimizer.Adam(
                    learning_rate=0.001, parameter_list=net.parameters())
            model = Model(net, inputs, labels)
            model.prepare(
                optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))
            model.fit(mnist_data, batch_size=64, verbose=0)
            model.save(path + '/test')
            model.load(path + '/test')
            shutil.rmtree(path)
            paddle.enable_static()

    def test_dynamic_save_static_load(self):
        path = tempfile.mkdtemp()
        # dynamic saving
        device = paddle.set_device('cpu')
        fluid.enable_dygraph(device)
        model = Model(MyModel())
        optim = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=model.parameters())
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))
        model.save(path + '/test')
        fluid.disable_dygraph()

        inputs = [InputSpec([None, 20], 'float32', 'x')]
        labels = [InputSpec([None, 1], 'int64', 'label')]
        model = Model(MyModel(), inputs, labels)
        optim = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=model.parameters())
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))
        model.load(path + '/test')
        shutil.rmtree(path)

    def test_static_save_dynamic_load(self):
        path = tempfile.mkdtemp()

        net = MyModel()
        inputs = [InputSpec([None, 20], 'float32', 'x')]
        labels = [InputSpec([None, 1], 'int64', 'label')]
        optim = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=net.parameters())
        model = Model(net, inputs, labels)
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))
        model.save(path + '/test')

        device = paddle.set_device('cpu')
        fluid.enable_dygraph(device)  #if dynamic else None

        net = MyModel()
        inputs = [InputSpec([None, 20], 'float32', 'x')]
        labels = [InputSpec([None, 1], 'int64', 'label')]
        optim = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=net.parameters())
        model = Model(net, inputs, labels)
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))
        model.load(path + '/test')
        shutil.rmtree(path)
        fluid.disable_dygraph()

    def test_parameters(self):
        for dynamic in [True, False]:
            device = paddle.set_device('cpu')
            fluid.enable_dygraph(device) if dynamic else None
            net = MyModel()
            inputs = [InputSpec([None, 20], 'float32', 'x')]
            model = Model(net, inputs)
            model.prepare()
            params = model.parameters()
            self.assertTrue(params[0].shape[0] == 20)
            self.assertTrue(params[0].shape[1] == 10)
            fluid.disable_dygraph() if dynamic else None

    def test_summary(self):
        def _get_param_from_state_dict(state_dict):
            params = 0
            for k, v in state_dict.items():
                params += np.prod(v.numpy().shape)
            return params

        for dynamic in [True, False]:
            device = paddle.set_device('cpu')
            fluid.enable_dygraph(device) if dynamic else None
            net = MyModel()
            inputs = [InputSpec([None, 20], 'float32', 'x')]
            model = Model(net, inputs)
            model.prepare()
            params_info = model.summary()
            gt_params = _get_param_from_state_dict(net.state_dict())

            np.testing.assert_allclose(params_info['total_params'], gt_params)
            print(params_info)

            model.summary(input_size=(20))
            model.summary(input_size=[(20)])
            model.summary(input_size=(20), dtype='float32')

    def test_summary_nlp(self):
        paddle.enable_static()
        nlp_net = paddle.nn.GRU(input_size=2,
                                hidden_size=3,
                                num_layers=3,
                                direction="bidirectional")
        paddle.summary(nlp_net, (1, 1, 2))
        rnn = paddle.nn.LSTM(16, 32, 2)
        paddle.summary(rnn, [(-1, 23, 16), ((2, None, 32), (2, -1, 32))])

    def test_summary_dtype(self):
        input_shape = (3, 1)
        net = paddle.nn.Embedding(10, 3, sparse=True)
        paddle.summary(net, input_shape, dtypes='int64')

    def test_summary_error(self):
        with self.assertRaises(TypeError):
            nlp_net = paddle.nn.GRU(input_size=2, hidden_size=3, num_layers=3)
            paddle.summary(nlp_net, (1, 1, '2'))

        with self.assertRaises(ValueError):
            nlp_net = paddle.nn.GRU(input_size=2, hidden_size=3, num_layers=3)
            paddle.summary(nlp_net, (-1, -1))

        paddle.disable_static()
        nlp_net = paddle.nn.GRU(input_size=2, hidden_size=3, num_layers=3)
        paddle.summary(nlp_net, (1, 1, 2))

    def test_export_deploy_model(self):
        self.set_seed()
        np.random.seed(2020)
        for dynamic in [True, False]:
            paddle.disable_static() if dynamic else None
            prog_translator = ProgramTranslator()
            prog_translator.enable(False) if not dynamic else None
            net = LeNet()
            inputs = [InputSpec([None, 1, 28, 28], 'float32', 'x')]
            model = Model(net, inputs)
            model.prepare()
            save_dir = tempfile.mkdtemp()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            tensor_img = np.array(
                np.random.random((1, 1, 28, 28)), dtype=np.float32)

            model.save(save_dir, training=False)
            ori_results = model.test_batch(tensor_img)
            fluid.disable_dygraph() if dynamic else None

            place = fluid.CPUPlace() if not fluid.is_compiled_with_cuda(
            ) else fluid.CUDAPlace(0)
            new_scope = fluid.Scope()
            with fluid.scope_guard(new_scope):
                exe = fluid.Executor(place)
                [inference_program, feed_target_names, fetch_targets] = (
                    fluid.io.load_inference_model(
                        dirname=save_dir, executor=exe))
                results = exe.run(inference_program,
                                  feed={feed_target_names[0]: tensor_img},
                                  fetch_list=fetch_targets)
                np.testing.assert_allclose(
                    results, ori_results, rtol=1e-5, atol=1e-7)
                shutil.rmtree(save_dir)
            paddle.enable_static()

    def test_dygraph_export_deploy_model_about_inputs(self):
        mnist_data = MnistDataset(mode='train')
        paddle.disable_static()
        # without inputs
        for initial in ["fit", "train_batch", "eval_batch", "test_batch"]:
            save_dir = tempfile.mkdtemp()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net = LeNet()
            model = Model(net)
            optim = fluid.optimizer.Adam(
                learning_rate=0.001, parameter_list=model.parameters())
            model.prepare(
                optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))
            if initial == "fit":
                model.fit(mnist_data, batch_size=64, verbose=0)
            else:
                img = np.array(
                    np.random.random((1, 1, 28, 28)), dtype=np.float32)
                label = np.array(np.random.rand(1, 1), dtype=np.int64)
                if initial == "train_batch":
                    model.train_batch([img], [label])
                elif initial == "eval_batch":
                    model.eval_batch([img], [label])
                else:
                    model.test_batch([img])

            model.save(save_dir, training=False)
            shutil.rmtree(save_dir)
        # with inputs, and the type of inputs is InputSpec
        save_dir = tempfile.mkdtemp()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        net = LeNet()
        inputs = InputSpec([None, 1, 28, 28], 'float32', 'x')
        model = Model(net, inputs)
        optim = fluid.optimizer.Adam(
            learning_rate=0.001, parameter_list=model.parameters())
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))
        model.save(save_dir, training=False)
        shutil.rmtree(save_dir)


class TestModelWithLRScheduler(unittest.TestCase):
    def test_fit(self):
        def make_optimizer(parameters=None):
            base_lr = 1e-3
            momentum = 0.9
            weight_decay = 5e-4
            boundaries = [5, 8]
            values = [base_lr * (0.1**i) for i in range(len(boundaries) + 1)]
            learning_rate = paddle.optimizer.lr.PiecewiseDecay(
                boundaries=boundaries, values=values)
            learning_rate = paddle.optimizer.lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=4,
                start_lr=base_lr / 5.,
                end_lr=base_lr,
                verbose=True)
            optimizer = paddle.optimizer.Momentum(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
                parameters=parameters)
            return optimizer

        device = paddle.set_device('cpu')
        fluid.enable_dygraph(device)
        net = MyModel()
        inputs = [InputSpec([None, 20], 'float32', 'x')]
        labels = [InputSpec([None, 1], 'int64', 'label')]
        optim = make_optimizer(net.parameters())
        model = Model(net, inputs, labels)
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))

        dataset = MyDataset()
        model.fit(dataset, dataset, batch_size=4, epochs=10, num_workers=0)

        paddle.enable_static()


class TestRaiseError(unittest.TestCase):
    def test_input_without_name(self):
        net = MyModel()

        inputs = [InputSpec([None, 10], 'float32')]
        labels = [InputSpec([None, 1], 'int64', 'label')]
        with self.assertRaises(ValueError):
            model = Model(net, inputs, labels)

    def test_static_without_inputs(self):
        paddle.enable_static()
        net = MyModel()
        with self.assertRaises(TypeError):
            model = Model(net)

    def test_save_infer_model_without_inputs_and_run_in_dygraph(self):
        paddle.disable_static()
        net = MyModel()
        save_dir = tempfile.mkdtemp()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with self.assertRaises(RuntimeError):
            model = Model(net)
            model.save(save_dir, training=False)
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
