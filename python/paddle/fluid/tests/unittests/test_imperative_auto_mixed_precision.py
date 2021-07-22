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

import unittest
import paddle
import paddle.fluid as fluid
import numpy as np
import six
from test_imperative_resnet import ResNet, BottleneckBlock, ConvBNLayer, train_parameters, optimizer_setting


class SimpleConv(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(SimpleConv, self).__init__()
        self._conv = fluid.dygraph.Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=None,
            use_cudnn=True)

    def forward(self, inputs):
        return self._conv(inputs)


class TestAutoCast(unittest.TestCase):
    def test_amp_guard_white_op(self):
        data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
        with fluid.dygraph.guard():
            conv2d = fluid.dygraph.Conv2D(3, 2, 3, bias_attr=False, act=None)
            data = fluid.dygraph.to_variable(data)
            with fluid.dygraph.amp_guard(True):
                out_fp16 = conv2d(data)

            with fluid.dygraph.amp_guard(False):
                out_fp32 = conv2d(data)

        self.assertTrue(data.dtype == fluid.core.VarDesc.VarType.FP32)
        self.assertTrue(out_fp16.dtype == fluid.core.VarDesc.VarType.FP16)
        self.assertTrue(out_fp32.dtype == fluid.core.VarDesc.VarType.FP32)

    def test_amp_guard_black_op(self):
        data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
        with fluid.dygraph.guard():
            data = fluid.dygraph.to_variable(data)
            with fluid.dygraph.amp_guard(True):
                out_fp32 = fluid.layers.mean(data)

        self.assertTrue(data.dtype == fluid.core.VarDesc.VarType.FP32)
        self.assertTrue(out_fp32.dtype == fluid.core.VarDesc.VarType.FP32)

    def test_custom_op_list(self):
        with fluid.dygraph.guard():
            tracer = fluid.framework._dygraph_tracer()
            base_white_list = fluid.dygraph.amp.auto_cast.WHITE_LIST
            base_black_list = fluid.dygraph.amp.auto_cast.BLACK_LIST
            with fluid.dygraph.amp_guard(
                    custom_white_list=["log"], custom_black_list=["conv2d"]):
                white_list, black_list = tracer._get_amp_op_list()
                self.assertTrue(
                    set(white_list) ==
                    (set(base_white_list) | {"log"}) - {"conv2d"})

                self.assertTrue(
                    set(black_list) ==
                    (set(base_black_list) - {"log"}) | {"conv2d"})

    def test_custom_op_list_exception(self):
        inp_np = np.random.random(size=[1, 3, 128, 128]).astype(np.float32)

        def func():
            with fluid.dygraph.guard():
                model = SimpleConv(
                    num_channels=3,
                    num_filters=64,
                    filter_size=7,
                    stride=2,
                    act='relu')

                with fluid.dygraph.amp_guard(
                        custom_white_list=["conv2d"],
                        custom_black_list=["conv2d"]):
                    inp = fluid.dygraph.to_variable(inp_np)
                    out = model(inp)

        self.assertRaises(ValueError, func)

    def test_amp_guard_upsupported_fp16_op(self):
        data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
        with fluid.dygraph.guard():
            conv2d = fluid.dygraph.Conv2D(3, 2, 3, bias_attr=False, act=None)
            data = fluid.dygraph.to_variable(data)
            with fluid.dygraph.amp_guard(True):
                out_fp16 = conv2d(data)
                out_fp32 = paddle.expand_as(
                    out_fp16, out_fp16)  # expand_as_v2 has no fp16 kernel

        self.assertTrue(data.dtype == fluid.core.VarDesc.VarType.FP32)
        self.assertTrue(out_fp16.dtype == fluid.core.VarDesc.VarType.FP16)
        self.assertTrue(out_fp32.dtype == fluid.core.VarDesc.VarType.FP32)


class TestAmpScaler(unittest.TestCase):
    def test_scale(self):
        with fluid.dygraph.guard():
            data = paddle.rand([10, 1024])
            scaler = paddle.fluid.dygraph.AmpScaler(init_loss_scaling=1024)
            scaled_data = scaler.scale(data)
            self.assertEqual(
                np.array_equal(scaled_data.numpy(), data.numpy() * 1024), True)

    def test_minimize(self):
        inp_np = np.random.random(size=[1, 3, 128, 128]).astype(np.float32)

        def run_simple_conv(inp_np, use_scaler=True):
            paddle.seed(10)
            paddle.framework.random._manual_program_seed(10)
            with fluid.dygraph.guard():
                model = SimpleConv(
                    num_channels=3,
                    num_filters=64,
                    filter_size=7,
                    stride=2,
                    act='relu')
                optimizer = fluid.optimizer.SGDOptimizer(
                    learning_rate=0.01, parameter_list=model.parameters())
                scaler = fluid.dygraph.AmpScaler(init_loss_scaling=1024)
                data = fluid.dygraph.to_variable(inp_np)

                out = model(data)
                loss = fluid.layers.mean(out)
                if use_scaler:
                    print('use scaler')
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
                    optimize_ops, params_grads = scaler.minimize(optimizer,
                                                                 scaled_loss)
                else:
                    print('use no scaler')
                    loss.backward()
                    optimize_ops, params_grads = optimizer.minimize(loss)
            return optimize_ops, params_grads

        outs_with_scaler = run_simple_conv(inp_np, use_scaler=True)
        outs_no_scaler = run_simple_conv(inp_np, use_scaler=False)

        self.assertEqual(outs_with_scaler[0],
                         [])  # optimize_ops is [] in dygraph mode
        self.assertEqual(outs_no_scaler[0],
                         [])  # optimize_ops is [] in dygraph mode
        for i in range(len(outs_with_scaler[1])):
            # check each grad
            self.assertEqual(
                np.allclose(outs_with_scaler[1][i][1].numpy(),
                            outs_no_scaler[1][i][1].numpy()), True)
            # check each parameter
            self.assertEqual(
                np.allclose(outs_with_scaler[1][i][0].numpy(),
                            outs_no_scaler[1][i][0].numpy()), True)

    def test_nan_inf(self):
        inp_np = np.random.random(size=[1, 3, 128, 128]).astype(np.float32)
        inp_np[0][1][2][3] = np.nan
        with fluid.dygraph.guard():
            model = SimpleConv(
                num_channels=3,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu')
            params_init = {}
            for param in model.parameters():
                params_init[param.name] = param.numpy()
            optimizer = fluid.optimizer.SGDOptimizer(
                learning_rate=0.01, parameter_list=model.parameters())
            scaler = fluid.dygraph.AmpScaler(init_loss_scaling=1024)
            data = fluid.dygraph.to_variable(inp_np)

            out = model(data)
            loss = fluid.layers.mean(out)
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            optimize_ops, params_grads = scaler.minimize(optimizer, scaled_loss)
            self.assertEqual(scaler._found_inf.numpy() == 1, True)

            for param in model.parameters():
                # param not update when tensor contains nan or inf
                self.assertTrue(
                    np.array_equal(param.numpy(), params_init[param.name]))

    def test_get_and_set(self):
        with fluid.dygraph.guard():
            scaler = paddle.amp.GradScaler(
                enable=True,
                init_loss_scaling=1024,
                incr_ratio=2.0,
                decr_ratio=0.5,
                incr_every_n_steps=1000,
                decr_every_n_nan_or_inf=2,
                use_dynamic_loss_scaling=True)
            self.assertEqual(scaler.is_enable() == True, True)
            self.assertEqual(scaler.get_init_loss_scaling() == 1024, True)
            self.assertEqual(scaler.get_incr_ratio() == 2.0, True)
            self.assertEqual(scaler.get_decr_ratio() == 0.5, True)
            self.assertEqual(scaler.get_incr_every_n_steps() == 1000, True)
            self.assertEqual(scaler.get_decr_every_n_nan_or_inf() == 2, True)
            self.assertEqual(scaler.is_use_dynamic_loss_scaling() == True, True)
            scaler.set_decr_every_n_nan_or_inf(4)
            self.assertEqual(scaler.get_decr_every_n_nan_or_inf() == 4, True)
            scaler.set_decr_ratio(0.1)
            self.assertEqual(scaler.get_decr_ratio() == 0.1, True)
            scaler.set_incr_every_n_steps(200)
            self.assertEqual(scaler.get_incr_every_n_steps() == 200, True)
            scaler.set_incr_ratio(3.0)
            self.assertEqual(scaler.get_incr_ratio() == 3.0, True)
            scaler.set_init_loss_scaling(100)
            self.assertEqual(scaler.get_init_loss_scaling() == 100, True)

    def test_state_dict_and_load_state_dict(self):
        with fluid.dygraph.guard():
            scaler1 = paddle.amp.GradScaler(
                enable=True,
                init_loss_scaling=14,
                incr_ratio=233.0,
                decr_ratio=0.523,
                incr_every_n_steps=1090,
                decr_every_n_nan_or_inf=20,
                use_dynamic_loss_scaling=True)
            scaler_state = scaler1.state_dict()
            scaler2 = paddle.amp.GradScaler(enable=True)
            scaler2.load_state_dict(scaler_state)
            self.assertEqual(scaler2.get_init_loss_scaling() == 14, True)
            self.assertEqual(scaler2.get_incr_ratio() == 233.0, True)
            self.assertEqual(scaler2.get_decr_ratio() == 0.523, True)
            self.assertEqual(scaler2.get_incr_every_n_steps() == 1090, True)
            self.assertEqual(scaler2.get_decr_every_n_nan_or_inf() == 20, True)


def reader_decorator(reader):
    def __reader__():
        for item in reader():
            img = np.array(item[0]).astype('float32').reshape(3, 224, 224)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield img, label

    return __reader__


class TestResnet2(unittest.TestCase):
    """
    Use paddle-2.0 API
    """

    def train_resnet(self, enable_amp=True, use_data_loader=False):
        seed = 90

        batch_size = train_parameters["batch_size"]
        batch_num = 1

        paddle.seed(seed)
        paddle.framework.random._manual_program_seed(seed)

        resnet = ResNet(use_cudnn=True)
        optimizer = optimizer_setting(
            train_parameters, parameter_list=resnet.parameters())
        np.random.seed(seed)
        train_reader = paddle.batch(
            paddle.dataset.flowers.train(use_xmap=False), batch_size=batch_size)

        dy_param_init_value = {}
        for param in resnet.parameters():
            dy_param_init_value[param.name] = param.numpy()

        program = None
        scaler = paddle.amp.GradScaler(
            enable=enable_amp, init_loss_scaling=2.**10)

        if use_data_loader:
            train_reader = paddle.batch(
                reader_decorator(paddle.dataset.flowers.train(use_xmap=False)),
                batch_size=batch_size,
                drop_last=True)
            train_loader = fluid.io.DataLoader.from_generator(
                capacity=4,
                use_double_buffer=True,
                iterable=True,
                return_list=True)
            train_loader.set_sample_list_generator(train_reader)
            train_reader = train_loader

        for batch_id, data in enumerate(train_reader()):
            if batch_id >= batch_num:
                break
            if use_data_loader:
                img, label = data
            else:
                dy_x_data = np.array(
                    [x[0].reshape(3, 224, 224) for x in data]).astype('float32')
                if len(np.array([x[1]
                                 for x in data]).astype('int64')) != batch_size:
                    continue
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                    -1, 1)

                img = paddle.to_tensor(dy_x_data)
                label = paddle.to_tensor(y_data)
            label.stop_gradient = True

            with paddle.amp.auto_cast(enable=enable_amp):
                out = resnet(img)

            loss = paddle.nn.functional.cross_entropy(input=out, label=label)
            avg_loss = paddle.mean(x=loss)

            dy_out = avg_loss.numpy()

            scaled_loss = scaler.scale(avg_loss)
            scaled_loss.backward()

            scaler.minimize(optimizer, scaled_loss)

            dy_grad_value = {}
            for param in resnet.parameters():
                if param.trainable:
                    np_array = np.array(param._grad_ivar().value().get_tensor())
                    dy_grad_value[param.name + fluid.core.grad_var_suffix(
                    )] = np_array

            resnet.clear_gradients()

            dy_param_value = {}
            for param in resnet.parameters():
                dy_param_value[param.name] = param.numpy()
        if use_data_loader:
            train_reader._reset()
        return dy_out, dy_param_value, dy_grad_value

    def test_resnet(self):
        with fluid.dygraph.guard():
            out_fp32 = self.train_resnet(enable_amp=False)
            out_amp = self.train_resnet(enable_amp=True)
        print(out_fp32[0], out_amp[0])
        self.assertTrue(np.allclose(out_fp32[0], out_amp[0], atol=1.e-2))

    def test_with_data_loader(self):
        with fluid.dygraph.guard():
            out_fp32 = self.train_resnet(enable_amp=False, use_data_loader=True)
            out_amp = self.train_resnet(enable_amp=True, use_data_loader=True)
        print(out_fp32[0], out_amp[0])
        self.assertTrue(np.allclose(out_fp32[0], out_amp[0], atol=1.e-2))


class TestResnet(unittest.TestCase):
    """
    Use paddle-1.x API
    """

    def train_resnet(self, enable_amp=True):
        seed = 90

        batch_size = train_parameters["batch_size"]
        batch_num = 1

        with fluid.dygraph.guard():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            resnet = ResNet(use_cudnn=True)
            optimizer = optimizer_setting(
                train_parameters, parameter_list=resnet.parameters())
            np.random.seed(seed)
            train_reader = paddle.batch(
                paddle.dataset.flowers.train(use_xmap=False),
                batch_size=batch_size)

            dy_param_init_value = {}
            for param in resnet.parameters():
                dy_param_init_value[param.name] = param.numpy()

            program = None
            scaler = paddle.fluid.dygraph.AmpScaler(
                enable=enable_amp, init_loss_scaling=2.**10)
            for batch_id, data in enumerate(train_reader()):
                if batch_id >= batch_num:
                    break
                dy_x_data = np.array(
                    [x[0].reshape(3, 224, 224) for x in data]).astype('float32')
                if len(np.array([x[1]
                                 for x in data]).astype('int64')) != batch_size:
                    continue
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                    -1, 1)
                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True
                with paddle.fluid.dygraph.amp_guard(enable=enable_amp):
                    out = resnet(img)

                loss = fluid.layers.cross_entropy(input=out, label=label)
                avg_loss = fluid.layers.mean(x=loss)

                dy_out = avg_loss.numpy()

                scaled_loss = scaler.scale(avg_loss)
                scaled_loss.backward()

                scaler.minimize(optimizer, scaled_loss)

                dy_grad_value = {}
                for param in resnet.parameters():
                    if param.trainable:
                        np_array = np.array(param._grad_ivar().value()
                                            .get_tensor())
                        dy_grad_value[param.name + fluid.core.grad_var_suffix(
                        )] = np_array

                resnet.clear_gradients()

                dy_param_value = {}
                for param in resnet.parameters():
                    dy_param_value[param.name] = param.numpy()

        return dy_out, dy_param_value, dy_grad_value

    def test_resnet(self):
        out_fp32 = self.train_resnet(enable_amp=False)
        out_amp = self.train_resnet(enable_amp=True)
        print(out_fp32[0], out_amp[0])
        self.assertTrue(np.allclose(out_fp32[0], out_amp[0], atol=1.e-2))


class TestLayerNormFp16(unittest.TestCase):
    r''' layer_norm and batch_norm support mixed inputs, i.e., only input x is fp16
    and other params are fp32.
    '''

    def test_layer_norm_fp16(self):
        if fluid.is_compiled_with_cuda():
            with fluid.dygraph.guard(fluid.CUDAPlace(0)):
                x = paddle.rand([2, 2, 2, 3])
                layer_norm = paddle.nn.LayerNorm(x.shape[1:])
                with paddle.amp.auto_cast(custom_white_list=['layer_norm']):
                    out = layer_norm(x)

                self.assertTrue(out.dtype == fluid.core.VarDesc.VarType.FP16)


if __name__ == '__main__':
    unittest.main()
