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

import numpy as np
import paddle
import paddle.fluid as fluid

BATCH_SIZE = 128
CLIP_MAX = 2e-6
CLIP_MIN = -1e-6

prog = fluid.framework.Program()

with fluid.program_guard(main_program=prog):
    image = fluid.layers.data(name='x', shape=[784], dtype='float32')

    hidden1 = fluid.layers.fc(input=image, size=128, act='relu')
    hidden2 = fluid.layers.fc(input=hidden1, size=64, act='relu')
    predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')

    label = fluid.layers.data(name='y', shape=[1], dtype='int64')

    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)

prog_clip = prog.clone()
prog_clip.block(0).var(hidden1.name)._set_error_clip(
    fluid.clip.ErrorClipByValue(
        max=CLIP_MAX, min=CLIP_MIN))

avg_cost_clip = prog_clip.block(0).var(avg_cost.name)
fluid.backward.append_backward(loss=avg_cost)
fluid.backward.append_backward(
    loss=avg_cost_clip, callbacks=[fluid.clip.error_clip_callback])

hidden1_grad = prog.block(0).var(hidden1.name + "@GRAD")
hidden1_grad_clip = prog_clip.block(0).var(hidden1.name + "@GRAD")

hidden2_grad = prog.block(0).var(hidden2.name + "@GRAD")
hidden2_grad_clip = prog_clip.block(0).var(hidden2.name + "@GRAD")

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=8192),
    batch_size=BATCH_SIZE)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(feed_list=[image, label], place=place)
exe.run(fluid.default_startup_program())

count = 0
for data in train_reader():
    count += 1
    if count > 5:
        break
    out1, out2 = exe.run(prog,
                         feed=feeder.feed(data),
                         fetch_list=[hidden1_grad, hidden2_grad])
    out1_clip, out2_clip = exe.run(
        prog_clip,
        feed=feeder.feed(data),
        fetch_list=[hidden1_grad_clip, hidden2_grad_clip])
    if not ((out1.clip(
            min=CLIP_MIN, max=CLIP_MAX) == out1_clip).all() and
            (out2 == out2_clip).all()):
        exit(1)

exit(0)
