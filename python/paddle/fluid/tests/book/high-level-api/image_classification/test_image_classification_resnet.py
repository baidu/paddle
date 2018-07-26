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

import paddle
import paddle.fluid as fluid
import numpy
import cifar10_small_test_set


def resnet_cifar10(input, depth=32):
    def conv_bn_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      bias_attr=False):
        tmp = fluid.layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=bias_attr)
        return fluid.layers.batch_norm(input=tmp, act=act)

    def shortcut(input, ch_in, ch_out, stride):
        if ch_in != ch_out:
            return conv_bn_layer(input, ch_out, 1, stride, 0, None)
        else:
            return input

    def basicblock(input, ch_in, ch_out, stride):
        tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
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
    predict = fluid.layers.fc(input=pool, size=10, act='softmax')
    return predict


def inference_network():
    data_shape = [3, 32, 32]
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    predict = resnet_cifar10(images, 32)
    return predict


def train_network():
    predict = inference_network()
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, accuracy]


def optimizer_func():
    return fluid.optimizer.Adam(learning_rate=0.001)


def train(use_cuda, train_program, params_dirname):
    BATCH_SIZE = 128
    EPOCH_NUM = 1

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            cifar10_small_test_set.train10(batch_size=10), buf_size=128 * 10),
        batch_size=BATCH_SIZE,
        drop_last=False)

    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE, drop_last=False)

    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):
            avg_cost, accuracy = trainer.test(
                reader=test_reader, feed_order=['pixel', 'label'])

            print(('Loss {0:2.2}, Acc {1:2.2}'.format(avg_cost, accuracy)))

            if accuracy > 0.01:  # Low threshold for speeding up CI
                if params_dirname is not None:
                    trainer.save_params(params_dirname)
                return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    trainer = fluid.Trainer(
        train_func=train_program, optimizer_func=optimizer_func, place=place)

    trainer.train(
        reader=train_reader,
        num_epochs=EPOCH_NUM,
        event_handler=event_handler,
        feed_order=['pixel', 'label'])


def infer(use_cuda, inference_program, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inferencer = fluid.Inferencer(
        infer_func=inference_program, param_path=params_dirname, place=place)

    # The input's dimension of conv should be 4-D or 5-D.
    # Use normilized image pixels as input data, which should be in the range
    # [0, 1.0].
    tensor_img = numpy.random.rand(1, 3, 32, 32).astype("float32")
    results = inferencer.infer({'pixel': tensor_img})

    print(("infer results: ", results))


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    save_path = "image_classification_resnet.inference.model"

    train(
        use_cuda=use_cuda,
        train_program=train_network,
        params_dirname=save_path)

    infer(
        use_cuda=use_cuda,
        inference_program=inference_network,
        params_dirname=save_path)


if __name__ == '__main__':
    for use_cuda in (False, True):
        main(use_cuda=use_cuda)
