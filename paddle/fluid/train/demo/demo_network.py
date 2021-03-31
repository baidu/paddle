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
import paddle.fluid.framework as framework

def least_square_graph(with_optimize):
    # construct a least square Paddle 1.4, note the batch size is used for
    x = fluid.layers.data(name='x', shape=[None, 13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[None, 1], dtype='float32')

    # fully connected to ouptut [batch_size , size] tensor
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

    if with_optimize:
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        sgd_optimizer.minimize(avg_cost)
    else:
        fluid.backward.append_backward(avg_cost)
    return avg_cost


def train():
    batch_size = 20
    enable_ce = True

    if enable_ce:
        train_reader = paddle.batch(
            paddle.reader.shuffle(paddle.dataset.uci_housing.train(), buf_size=500),
            batch_size=batch_size
        )
        valid_reader = paddle.batch(
            paddle.reader.shuffle(paddle.dataset.uci_housing.test(), buf_size=500),
            batch_size=batch_size
        )
    else:
        raise Exception("Not Implemented Yet!")

    startup_program = framework.Program()
    train_program = framework.Program() #after fluid 1.5 you can also use train_program.clone to obtain validation program
    place = fluid.CPUPlace()

    if enable_ce:
        startup_program.random_seed = 90
        train_program.random_seed = 90

    with framework.program_guard(train_program, startup_program):
        avg_cost = least_square_graph(with_optimize=True)
        validation_program = train_program.clone(for_test=True)

        train_executor = fluid.Executor(place)
        validate_executor = fluid.Executor(place)

        # main train loop
        feeder = fluid.DataFeeder(feed_list=['x', 'y'], place=place)

        train_executor.run(startup_program)

        step = 0
        num_epochs = 100
        # train and validate
        for pass_id in range(num_epochs):
            if (pass_id+1) % 10 == 0:
                validation_loss = 0.
                count = 0
                for data in valid_reader():
                    avg_loss_val = validate_executor.run(validation_program, feed=feeder.feed(data),
                                                         fetch_list=[avg_cost])
                    validation_loss += avg_loss_val[0]
                    count += 1

                # check avg_loss_val
                validation_loss /= count
                if validation_loss < 10.0:
                    break
                else:
                    pass #print("[Validation LR solver, epoch %d, cost %f]" % (pass_id, validation_loss))
            train_loss = 0.
            for data in train_reader():
                avg_loss_val = train_executor.run(train_program, feed=feeder.feed(data), fetch_list=[avg_cost])
                if step % 10 == 0:
                    pass #print("[Training LR solver, epoch %d, step %d, cost %f]" % (pass_id, step, avg_loss_val[0]))

                train_loss = avg_loss_val[0]
                step += 1

            if (step+1) % 20 == 0:
                print("[Training LR solver, epoch %d, step %d, cost %f, validation err %f]" % (pass_id, step, train_loss, validation_loss))
            # save parameters

def save_program_desc(network_func):
    startup_program = framework.Program()
    train_program = framework.Program()

    with framework.program_guard(train_program, startup_program):
        # build graph here
        network_func(with_optimize=True)

    with open("startup_program", "wb") as f:
        f.write(startup_program.desc.serialize_to_string())
    with open("main_program", "wb") as f:
        f.write(train_program.desc.serialize_to_string())

paddle.enable_static()
save_program_desc(least_square_graph)
train()
