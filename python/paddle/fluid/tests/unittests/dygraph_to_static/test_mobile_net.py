# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import time
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph import declarative, ProgramTranslator
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX

import unittest

from predictor_utils import PredictorTools

# Note: Set True to eliminate randomness.
#     1. For one operation, cuDNN has several algorithms,
#        some algorithm results are non-deterministic, like convolution algorithms.
if fluid.is_compiled_with_cuda():
    fluid.set_flags({'FLAGS_cudnn_deterministic': True})

SEED = 2020
program_translator = ProgramTranslator()


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 act='relu',
                 use_cudnn=True,
                 name=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(
                initializer=MSRA(), name=self.full_name() + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=self.full_name() + "_bn" + "_scale"),
            bias_attr=ParamAttr(name=self.full_name() + "_bn" + "_offset"),
            moving_mean_name=self.full_name() + "_bn" + '_mean',
            moving_variance_name=self.full_name() + "_bn" + '_variance')

    def forward(self, inputs, if_act=False):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if if_act:
            y = fluid.layers.relu6(y)
        return y


class DepthwiseSeparable(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters1,
                 num_filters2,
                 num_groups,
                 stride,
                 scale,
                 name=None):
        super(DepthwiseSeparable, self).__init__()

        self._depthwise_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=int(num_filters1 * scale),
            filter_size=3,
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=True)

        self._pointwise_conv = ConvBNLayer(
            num_channels=int(num_filters1 * scale),
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)

    def forward(self, inputs):
        y = self._depthwise_conv(inputs)
        y = self._pointwise_conv(y)
        return y


class MobileNetV1(fluid.dygraph.Layer):
    def __init__(self, scale=1.0, class_dim=1000):
        super(MobileNetV1, self).__init__()
        self.scale = scale
        self.dwsl = []

        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            channels=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1)

        dws21 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(32 * scale),
                num_filters1=32,
                num_filters2=64,
                num_groups=32,
                stride=1,
                scale=scale),
            name="conv2_1")
        self.dwsl.append(dws21)

        dws22 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(64 * scale),
                num_filters1=64,
                num_filters2=128,
                num_groups=64,
                stride=2,
                scale=scale),
            name="conv2_2")
        self.dwsl.append(dws22)

        dws31 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(128 * scale),
                num_filters1=128,
                num_filters2=128,
                num_groups=128,
                stride=1,
                scale=scale),
            name="conv3_1")
        self.dwsl.append(dws31)

        dws32 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(128 * scale),
                num_filters1=128,
                num_filters2=256,
                num_groups=128,
                stride=2,
                scale=scale),
            name="conv3_2")
        self.dwsl.append(dws32)

        dws41 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(256 * scale),
                num_filters1=256,
                num_filters2=256,
                num_groups=256,
                stride=1,
                scale=scale),
            name="conv4_1")
        self.dwsl.append(dws41)

        dws42 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(256 * scale),
                num_filters1=256,
                num_filters2=512,
                num_groups=256,
                stride=2,
                scale=scale),
            name="conv4_2")
        self.dwsl.append(dws42)

        for i in range(5):
            tmp = self.add_sublayer(
                sublayer=DepthwiseSeparable(
                    num_channels=int(512 * scale),
                    num_filters1=512,
                    num_filters2=512,
                    num_groups=512,
                    stride=1,
                    scale=scale),
                name="conv5_" + str(i + 1))
            self.dwsl.append(tmp)

        dws56 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(512 * scale),
                num_filters1=512,
                num_filters2=1024,
                num_groups=512,
                stride=2,
                scale=scale),
            name="conv5_6")
        self.dwsl.append(dws56)

        dws6 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                num_channels=int(1024 * scale),
                num_filters1=1024,
                num_filters2=1024,
                num_groups=1024,
                stride=1,
                scale=scale),
            name="conv6")
        self.dwsl.append(dws6)

        self.pool2d_avg = Pool2D(pool_type='avg', global_pooling=True)

        self.out = Linear(
            int(1024 * scale),
            class_dim,
            param_attr=ParamAttr(
                initializer=MSRA(), name=self.full_name() + "fc7_weights"),
            bias_attr=ParamAttr(name="fc7_offset"))

    @declarative
    def forward(self, inputs):
        y = self.conv1(inputs)
        for dws in self.dwsl:
            y = dws(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.reshape(y, shape=[-1, 1024])
        y = self.out(y)
        return y


class InvertedResidualUnit(fluid.dygraph.Layer):
    def __init__(
            self,
            num_channels,
            num_in_filter,
            num_filters,
            stride,
            filter_size,
            padding,
            expansion_factor, ):
        super(InvertedResidualUnit, self).__init__()
        num_expfilter = int(round(num_in_filter * expansion_factor))
        self._expand_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            num_groups=1)

        self._bottleneck_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            num_groups=num_expfilter,
            act=None,
            use_cudnn=True)

        self._linear_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            num_groups=1)

    def forward(self, inputs, ifshortcut):
        y = self._expand_conv(inputs, if_act=True)
        y = self._bottleneck_conv(y, if_act=True)
        y = self._linear_conv(y, if_act=False)
        if ifshortcut:
            y = fluid.layers.elementwise_add(inputs, y)
        return y


class InvresiBlocks(fluid.dygraph.Layer):
    def __init__(self, in_c, t, c, n, s):
        super(InvresiBlocks, self).__init__()

        self._first_block = InvertedResidualUnit(
            num_channels=in_c,
            num_in_filter=in_c,
            num_filters=c,
            stride=s,
            filter_size=3,
            padding=1,
            expansion_factor=t)

        self._inv_blocks = []
        for i in range(1, n):
            tmp = self.add_sublayer(
                sublayer=InvertedResidualUnit(
                    num_channels=c,
                    num_in_filter=c,
                    num_filters=c,
                    stride=1,
                    filter_size=3,
                    padding=1,
                    expansion_factor=t),
                name=self.full_name() + "_" + str(i + 1))
            self._inv_blocks.append(tmp)

    def forward(self, inputs):
        y = self._first_block(inputs, ifshortcut=False)
        for inv_block in self._inv_blocks:
            y = inv_block(y, ifshortcut=True)
        return y


class MobileNetV2(fluid.dygraph.Layer):
    def __init__(self, class_dim=1000, scale=1.0):
        super(MobileNetV2, self).__init__()
        self.scale = scale
        self.class_dim = class_dim

        bottleneck_params_list = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        #1. conv1
        self._conv1 = ConvBNLayer(
            num_channels=3,
            num_filters=int(32 * scale),
            filter_size=3,
            stride=2,
            act=None,
            padding=1)

        #2. bottleneck sequences
        self._invl = []
        i = 1
        in_c = int(32 * scale)
        for layer_setting in bottleneck_params_list:
            t, c, n, s = layer_setting
            i += 1
            tmp = self.add_sublayer(
                sublayer=InvresiBlocks(
                    in_c=in_c, t=t, c=int(c * scale), n=n, s=s),
                name='conv' + str(i))
            self._invl.append(tmp)
            in_c = int(c * scale)

        #3. last_conv
        self._out_c = int(1280 * scale) if scale > 1.0 else 1280
        self._conv9 = ConvBNLayer(
            num_channels=in_c,
            num_filters=self._out_c,
            filter_size=1,
            stride=1,
            act=None,
            padding=0)

        #4. pool
        self._pool2d_avg = Pool2D(pool_type='avg', global_pooling=True)

        #5. fc
        tmp_param = ParamAttr(name=self.full_name() + "fc10_weights")
        self._fc = Linear(
            self._out_c,
            class_dim,
            param_attr=tmp_param,
            bias_attr=ParamAttr(name="fc10_offset"))

    @declarative
    def forward(self, inputs):
        y = self._conv1(inputs, if_act=True)
        for inv in self._invl:
            y = inv(y)
        y = self._conv9(y, if_act=True)
        y = self._pool2d_avg(y)
        y = fluid.layers.reshape(y, shape=[-1, self._out_c])
        y = self._fc(y)
        return y


def create_optimizer(args, parameter_list):
    optimizer = fluid.optimizer.Momentum(
        learning_rate=args.lr,
        momentum=args.momentum_rate,
        regularization=fluid.regularizer.L2Decay(args.l2_decay),
        parameter_list=parameter_list)

    return optimizer


def fake_data_reader(batch_size, label_size):
    local_random = np.random.RandomState(SEED)

    def reader():
        batch_data = []
        while True:
            img = local_random.random_sample([3, 224, 224]).astype('float32')
            label = local_random.randint(0, label_size, [1]).astype('int64')
            batch_data.append([img, label])
            if len(batch_data) == batch_size:
                yield batch_data
                batch_data = []

    return reader


class Args(object):
    batch_size = 4
    model = "MobileNetV1"
    lr = 0.001
    momentum_rate = 0.99
    l2_decay = 0.1
    num_epochs = 1
    class_dim = 50
    print_step = 1
    train_step = 10
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
    ) else fluid.CPUPlace()
    model_save_dir = "./inference"
    model_save_prefix = "./inference/" + model
    model_filename = model + INFER_MODEL_SUFFIX
    params_filename = model + INFER_PARAMS_SUFFIX
    dy_state_dict_save_path = model + ".dygraph"


def train_mobilenet(args, to_static):
    program_translator.enable(to_static)
    with fluid.dygraph.guard(args.place):

        np.random.seed(SEED)
        paddle.manual_seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

        if args.model == "MobileNetV1":
            net = MobileNetV1(class_dim=args.class_dim, scale=1.0)
        elif args.model == "MobileNetV2":
            net = MobileNetV2(class_dim=args.class_dim, scale=1.0)
        else:
            print(
                "wrong model name, please try model = MobileNetV1 or MobileNetV2"
            )
            exit()

        optimizer = create_optimizer(args=args, parameter_list=net.parameters())

        # 3. reader
        train_reader = fake_data_reader(args.batch_size, args.class_dim)
        train_data_loader = fluid.io.DataLoader.from_generator(capacity=16)
        train_data_loader.set_sample_list_generator(train_reader)

        # 4. train loop
        loss_data = []
        for eop in range(args.num_epochs):
            net.train()
            batch_id = 0
            t_last = 0
            for img, label in train_data_loader():
                t1 = time.time()
                t_start = time.time()
                out = net(img)

                t_end = time.time()
                softmax_out = fluid.layers.softmax(out, use_cudnn=False)
                loss = fluid.layers.cross_entropy(
                    input=softmax_out, label=label)
                avg_loss = fluid.layers.mean(x=loss)
                acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
                acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
                t_start_back = time.time()

                loss_data.append(avg_loss.numpy())
                avg_loss.backward()
                t_end_back = time.time()
                optimizer.minimize(avg_loss)
                net.clear_gradients()

                t2 = time.time()
                train_batch_elapse = t2 - t1
                if batch_id % args.print_step == 0:
                    print("epoch id: %d, batch step: %d,  avg_loss %0.5f acc_top1 %0.5f acc_top5 %0.5f %2.4f sec net_t:%2.4f back_t:%2.4f read_t:%2.4f" % \
                          (eop, batch_id, avg_loss.numpy(), acc_top1.numpy(), acc_top5.numpy(), train_batch_elapse,
                           t_end - t_start, t_end_back - t_start_back,  t1 - t_last))
                batch_id += 1
                t_last = time.time()
                if batch_id > args.train_step:
                    if to_static:
                        fluid.dygraph.jit.save(net, args.model_save_prefix)
                    else:
                        fluid.dygraph.save_dygraph(net.state_dict(),
                                                   args.dy_state_dict_save_path)
                    break

    return np.array(loss_data)


def predict_static(args, data):
    paddle.enable_static()
    exe = fluid.Executor(args.place)
    # load inference model
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(
         args.model_save_dir,
         executor=exe,
         model_filename=args.model_filename,
         params_filename=args.params_filename)

    pred_res = exe.run(inference_program,
                       feed={feed_target_names[0]: data},
                       fetch_list=fetch_targets)
    return pred_res[0]


def predict_dygraph(args, data):
    program_translator.enable(False)
    with fluid.dygraph.guard(args.place):
        if args.model == "MobileNetV1":
            model = MobileNetV1(class_dim=args.class_dim, scale=1.0)
        elif args.model == "MobileNetV2":
            model = MobileNetV2(class_dim=args.class_dim, scale=1.0)
        # load dygraph trained parameters
        model_dict, _ = fluid.load_dygraph(args.dy_state_dict_save_path)
        model.set_dict(model_dict)
        model.eval()

        pred_res = model(fluid.dygraph.to_variable(data))

        return pred_res.numpy()


def predict_dygraph_jit(args, data):
    with fluid.dygraph.guard(args.place):
        model = fluid.dygraph.jit.load(args.model_save_prefix)
        model.eval()

        pred_res = model(data)

        return pred_res.numpy()


def predict_analysis_inference(args, data):
    output = PredictorTools(args.model_save_dir, args.model_filename,
                            args.params_filename, [data])
    out = output()
    return out


class TestMobileNet(unittest.TestCase):
    def setUp(self):
        self.args = Args()

    def train(self, model_name, to_static):
        self.args.model = model_name
        self.args.model_save_prefix = "./inference/" + model_name
        self.args.model_filename = model_name + INFER_MODEL_SUFFIX
        self.args.params_filename = model_name + INFER_PARAMS_SUFFIX
        self.args.dy_state_dict_save_path = model_name + ".dygraph"
        out = train_mobilenet(self.args, to_static)
        return out

    def assert_same_loss(self, model_name):
        dy_out = self.train(model_name, to_static=False)
        st_out = self.train(model_name, to_static=True)
        self.assertTrue(
            np.allclose(dy_out, st_out),
            msg="dy_out: {}, st_out: {}".format(dy_out, st_out))

    def assert_same_predict(self, model_name):
        self.args.model = model_name
        self.args.model_save_prefix = "./inference/" + model_name
        self.args.model_filename = model_name + INFER_MODEL_SUFFIX
        self.args.params_filename = model_name + INFER_PARAMS_SUFFIX
        self.args.dy_state_dict_save_path = model_name + ".dygraph"
        local_random = np.random.RandomState(SEED)
        image = local_random.random_sample([1, 3, 224, 224]).astype('float32')
        dy_pre = predict_dygraph(self.args, image)
        st_pre = predict_static(self.args, image)
        dy_jit_pre = predict_dygraph_jit(self.args, image)
        predictor_pre = predict_analysis_inference(self.args, image)
        self.assertTrue(
            np.allclose(dy_pre, st_pre),
            msg="dy_pre:\n {}\n, st_pre: \n{}.".format(dy_pre, st_pre))
        self.assertTrue(
            np.allclose(dy_jit_pre, st_pre),
            msg="dy_jit_pre:\n {}\n, st_pre: \n{}.".format(dy_jit_pre, st_pre))
        self.assertTrue(
            np.allclose(
                predictor_pre, st_pre, atol=1e-5),
            msg="inference_pred_res:\n {}\n, st_pre: \n{}.".format(
                predictor_pre, st_pre))

    def test_mobile_net(self):
        # MobileNet-V1
        self.assert_same_loss("MobileNetV1")
        # MobileNet-V2
        self.assert_same_loss("MobileNetV2")

        self.verify_predict()

    def verify_predict(self):
        # MobileNet-V1
        self.assert_same_predict("MobileNetV1")
        # MobileNet-V2
        self.assert_same_predict("MobileNetV2")


if __name__ == '__main__':
    unittest.main()
