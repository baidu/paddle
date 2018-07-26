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

import paddle.fluid as fluid
import numpy
import sys

TRAIN_FILES = ['train.recordio']
TEST_FILES = ['test.recordio']

DICT_DIM = 89528

# embedding dim
emb_dim = 128

# hidden dim
hid_dim = 128

# hidden dim2
hid_dim2 = 96

# class num
class_dim = 2


def network_cfg(is_train, pass_num=100):
    with fluid.unique_name.guard():
        train_file_obj = fluid.layers.open_files(
            filenames=TRAIN_FILES,
            pass_num=pass_num,
            shapes=[[-1, 1], [-1, 1]],
            lod_levels=[1, 0],
            dtypes=['int64', 'int64'])

        test_file_obj = fluid.layers.open_files(
            filenames=TEST_FILES,
            pass_num=1,
            shapes=[[-1, 1], [-1, 1]],
            lod_levels=[1, 0],
            dtypes=['int64', 'int64'])

        if is_train:
            file_obj = fluid.layers.shuffle(train_file_obj, buffer_size=1000)
        else:
            file_obj = test_file_obj

        file_obj = fluid.layers.double_buffer(
            file_obj,
            name="train_double_buffer" if is_train else 'test_double_buffer')

        data, label = fluid.layers.read_file(file_obj)

        emb = fluid.layers.embedding(input=data, size=[DICT_DIM, emb_dim])

        # sequence conv with window size = 3
        win_size = 3
        conv_3 = fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=hid_dim,
            filter_size=win_size,
            act="tanh",
            pool_type="max")

        # fc layer after conv
        fc_1 = fluid.layers.fc(input=[conv_3], size=hid_dim2)

        # probability of each class
        prediction = fluid.layers.fc(input=[fc_1],
                                     size=class_dim,
                                     act="softmax")
        # cross entropy loss
        cost = fluid.layers.cross_entropy(input=prediction, label=label)

        # mean loss
        avg_cost = fluid.layers.mean(x=cost)
        acc = fluid.layers.accuracy(input=prediction, label=label)

        if is_train:
            # SGD optimizer
            sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=0.01)
            sgd_optimizer.minimize(avg_cost)

        return {
            'loss': avg_cost,
            'log': [avg_cost, acc],
            'file': train_file_obj if is_train else test_file_obj
        }


def main():
    train = fluid.Program()
    startup = fluid.Program()

    with fluid.program_guard(train, startup):
        train_args = network_cfg(is_train=True)

    test = fluid.Program()

    with fluid.program_guard(test, fluid.Program()):
        test_args = network_cfg(is_train=False)

    # startup
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place=place)
    exe.run(startup)

    train_exe = fluid.ParallelExecutor(
        use_cuda=True, loss_name=train_args['loss'].name, main_program=train)

    fetch_var_list = [var.name for var in train_args['log']]
    for i in range(sys.maxsize):
        result = list(
            map(numpy.array,
                train_exe.run(fetch_list=fetch_var_list
                              if i % 1000 == 0 else [])))
        if len(result) != 0:
            print(('Train: ', result))

        if i % 1000 == 0:
            test_exe = fluid.ParallelExecutor(
                use_cuda=True, main_program=test, share_vars_from=train_exe)
            loss = []
            acc = []
            try:
                while True:
                    loss_np, acc_np = list(
                        map(numpy.array,
                            test_exe.run(fetch_list=fetch_var_list)))
                    loss.append(loss_np[0])
                    acc.append(acc_np[0])
            except:
                test_args['file'].reset()
                print(('TEST: ', numpy.mean(loss), numpy.mean(acc)))


if __name__ == '__main__':
    main()
