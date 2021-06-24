#   copyright (c) 2021 paddlepaddle authors. all rights reserved.
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
import numpy as np
import logging

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.dygraph.container import Sequential
from paddle.nn import ReLU, ReLU6, LeakyReLU, Sigmoid, Softmax, PReLU
from paddle.nn import Linear, Conv2D, Softmax, BatchNorm2D, MaxPool2D

from paddle.fluid.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


def fix_model_dict(model):
    fixed_state = {}
    for name, param in model.named_parameters():
        p_shape = param.numpy().shape
        p_value = param.numpy()
        if name.endswith("bias"):
            value = np.zeros_like(p_value).astype('float32')
        else:
            value = np.random.normal(
                loc=0.0, scale=0.01,
                size=np.product(p_shape)).reshape(p_shape).astype('float32')
        fixed_state[name] = value
    model.set_dict(fixed_state)
    return model


def train_lenet(lenet, reader, optimizer):
    loss_list = []
    lenet.train()

    for batch_id, data in enumerate(reader()):
        x_data = np.array([x[0].reshape(1, 28, 28)
                           for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)

        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)

        out = lenet(img)
        loss = fluid.layers.cross_entropy(out, label)
        avg_loss = fluid.layers.mean(loss)
        avg_loss.backward()

        optimizer.minimize(avg_loss)
        lenet.clear_gradients()

        if batch_id % 100 == 0:
            loss_list.append(avg_loss.numpy()[0])
            _logger.info('{}: {}'.format('loss', avg_loss.numpy()))

    return loss_list


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
        self.add = paddle.nn.quant.add()

    def forward(self, inputs):
        x = self.features(inputs)

        x = fluid.layers.flatten(x, 1)
        x = self.add(x, paddle.to_tensor(0.0))  # For CI
        x = self.fc(x)
        return x


class ImperativeLenetWithSkipQuant(fluid.dygraph.Layer):
    def __init__(self, num_classes=10):
        super(ImperativeLenetWithSkipQuant, self).__init__()

        conv2d_w1_attr = fluid.ParamAttr(name="conv2d_w_1")
        conv2d_w2_attr = fluid.ParamAttr(name="conv2d_w_2")
        fc_w1_attr = fluid.ParamAttr(name="fc_w_1")
        fc_w2_attr = fluid.ParamAttr(name="fc_w_2")
        fc_w3_attr = fluid.ParamAttr(name="fc_w_3")
        conv2d_b1_attr = fluid.ParamAttr(name="conv2d_b_1")
        conv2d_b2_attr = fluid.ParamAttr(name="conv2d_b_2")
        fc_b1_attr = fluid.ParamAttr(name="fc_b_1")
        fc_b2_attr = fluid.ParamAttr(name="fc_b_2")
        fc_b3_attr = fluid.ParamAttr(name="fc_b_3")
        self.conv2d_0 = Conv2D(
            in_channels=1,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=conv2d_w1_attr,
            bias_attr=conv2d_b1_attr)
        self.conv2d_0.skip_quant = True

        self.batch_norm_0 = BatchNorm2D(6)
        self.relu_0 = ReLU()
        self.pool2d_0 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2d_1 = Conv2D(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            weight_attr=conv2d_w2_attr,
            bias_attr=conv2d_b2_attr)
        self.conv2d_1.skip_quant = False

        self.batch_norm_1 = BatchNorm2D(16)
        self.relu6_0 = ReLU6()
        self.pool2d_1 = MaxPool2D(kernel_size=2, stride=2)
        self.linear_0 = Linear(
            in_features=400,
            out_features=120,
            weight_attr=fc_w1_attr,
            bias_attr=fc_b1_attr)
        self.linear_0.skip_quant = True

        self.leaky_relu_0 = LeakyReLU()
        self.linear_1 = Linear(
            in_features=120,
            out_features=84,
            weight_attr=fc_w2_attr,
            bias_attr=fc_b2_attr)
        self.linear_1.skip_quant = False

        self.sigmoid_0 = Sigmoid()
        self.linear_2 = Linear(
            in_features=84,
            out_features=num_classes,
            weight_attr=fc_w3_attr,
            bias_attr=fc_b3_attr)
        self.linear_2.skip_quant = False
        self.softmax_0 = Softmax()

    def forward(self, inputs):
        x = self.conv2d_0(inputs)
        x = self.batch_norm_0(x)
        x = self.relu_0(x)
        x = self.pool2d_0(x)
        x = self.conv2d_1(x)
        x = self.batch_norm_1(x)
        x = self.relu6_0(x)
        x = self.pool2d_1(x)

        x = fluid.layers.flatten(x, 1)

        x = self.linear_0(x)
        x = self.leaky_relu_0(x)
        x = self.linear_1(x)
        x = self.sigmoid_0(x)
        x = self.linear_2(x)
        x = self.softmax_0(x)

        return x
