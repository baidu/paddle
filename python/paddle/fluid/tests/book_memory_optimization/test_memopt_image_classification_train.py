#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import sys

import paddle
import paddle.fluid as fluid
import math
import sys

# need to fix random seed and training data to compare the loss
# value accurately calculated by the default and the memory optimization
# version.
fluid.default_startup_program().random_seed = 111


def resnet_cifar10(input, depth=32):
    def conv_bn_layer(input, ch_out, filter_size, stride, padding, act='relu'):
        tmp = fluid.layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=False)
        return fluid.layers.batch_norm(input=tmp, act=act)

    def shortcut(input, ch_in, ch_out, stride):
        if ch_in != ch_out:
            return conv_bn_layer(input, ch_out, 1, stride, 0, None)
        else:
            return input

    def basicblock(input, ch_in, ch_out, stride):
        tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None)
        short = shortcut(input, ch_in, ch_out, stride)
        return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')

    def layer_warp(block_func, input, ch_in, ch_out, count, stride):
        tmp = block_func(input, ch_in, ch_out, stride)
        for i in range(1, count):
            tmp = block_func(tmp, ch_out, ch_out, 1)
        return tmp

    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    conv1 = conv_bn_layer(
        input=input, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    return pool


def vgg16_bn_drop(input):
    def conv_block(input, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=4096, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=4096, act=None)
    return fc2


classdim = 10
data_shape = [3, 32, 32]

images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

net_type = "vgg"
if len(sys.argv) >= 2:
    net_type = sys.argv[1]

if net_type == "vgg":
    print("train vgg net")
    net = vgg16_bn_drop(images)
elif net_type == "resnet":
    print("train resnet")
    net = resnet_cifar10(images, 32)
else:
    raise ValueError("%s network is not supported" % net_type)

predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
cost = fluid.layers.cross_entropy(input=predict, label=label)
avg_cost = fluid.layers.mean(cost)

optimizer = fluid.optimizer.Adam(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)

batch_size = fluid.layers.create_tensor(dtype='int64')
batch_acc = fluid.layers.accuracy(input=predict, label=label, total=batch_size)

# fluid.memory_optimize(fluid.default_main_program(), level=0)
fluid.release_memory(fluid.default_main_program())

BATCH_SIZE = 16
PASS_NUM = 1

# fix the order of training data
train_reader = paddle.batch(
    paddle.dataset.cifar.train10(), batch_size=BATCH_SIZE)

# train_reader = paddle.batch(
#     paddle.reader.shuffle(
#         paddle.dataset.cifar.train10(), buf_size=128 * 10),
#     batch_size=BATCH_SIZE)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(place=place, feed_list=[images, label])
exe.run(fluid.default_startup_program())

i = 0

accuracy = fluid.average.WeightedAverage()
for pass_id in range(PASS_NUM):
    accuracy.reset()
    for data in train_reader():
        loss, acc, weight = exe.run(
            fluid.default_main_program(),
            feed=feeder.feed(data),
            fetch_list=[avg_cost, batch_acc, batch_size])
        accuracy.add(value=acc, weight=weight)
        pass_acc = accuracy.eval()
        print(("loss:" + str(loss) + " acc:" + str(acc) + " pass_acc:" +
               str(pass_acc)))
        # this model is slow, so if we can train two mini batch, we think it works properly.
        if i > 0:
            exit(0)
        if math.isnan(float(loss)):
            sys.exit("got NaN loss, training failed.")
        i += 1
exit(1)
