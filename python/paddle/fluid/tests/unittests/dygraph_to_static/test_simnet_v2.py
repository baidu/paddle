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

import argparse
import numpy as np
import paddle
import random
import unittest

from simnet_dygraph_model_v2 import BOW, HingeLoss

SEED = 102
random.seed(SEED)


def create_conf_dict():
    conf_dict = {}
    conf_dict["task_mode"] = "pairwise"
    conf_dict["net"] = {"emb_dim": 128, "bow_dim": 128, "hidden_dim": 128}
    conf_dict["loss"] = {"margin": 0.1}
    return conf_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Total examples' number in batch for training.")
    parser.add_argument(
        "--seq_len", type=int, default=32, help="The length of each sentence.")
    parser.add_argument(
        "--epoch", type=int, default=1, help="The number of training epoch.")
    parser.add_argument(
        "--fake_sample_size",
        type=int,
        default=128,
        help="The number of samples of fake data.")
    args = parser.parse_args([])
    return args


args = parse_args()


def fake_vocabulary():
    vocab = {}
    vocab["<unk>"] = 0
    for i in range(26):
        c = chr(ord('a') + i)
        vocab[c] = i + 1
    return vocab


vocab = fake_vocabulary()


class FakeReaderProcessor(object):
    def __init__(self, args, vocab):
        self.vocab = vocab
        self.seq_len = args.seq_len
        self.sample_size = args.fake_sample_size
        self.data_samples = []
        for i in range(self.sample_size):
            query = [random.randint(0, 26) for i in range(self.seq_len)]
            pos_title = query[:]
            neg_title = [26 - q for q in query]
            self.data_samples.append(
                np.array([query, pos_title, neg_title]).astype(np.int64))

    def get_reader(self, mode, epoch=0):
        def reader_with_pairwise():
            if mode == "train":
                for i in range(self.sample_size):
                    yield self.data_samples[i]

        return reader_with_pairwise


simnet_process = FakeReaderProcessor(args, vocab)


def train(conf_dict, to_static):
    """
    train process
    """
    program_translator = paddle.jit.ProgramTranslator()
    program_translator.enable(to_static)

    # Get device
    if paddle.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    else:
        place = paddle.CPUPlace()

    paddle.disable_static(place)
    paddle.manual_seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)

    conf_dict['dict_size'] = len(vocab)
    conf_dict['seq_len'] = args.seq_len

    net = BOW(conf_dict)
    loss = HingeLoss(conf_dict)
    optimizer = paddle.optimizer.Adam(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        parameters=net.parameters())

    metric = paddle.metric.Auc(name="auc")

    global_step = 0
    losses = []

    train_loader = paddle.io.DataLoader.from_generator(
        capacity=16, return_list=True, iterable=True, use_double_buffer=True)
    get_train_examples = simnet_process.get_reader("train", epoch=args.epoch)
    train_loader.set_sample_list_generator(
        paddle.batch(
            get_train_examples, batch_size=args.batch_size), place)

    for left, pos_right, neg_right in train_loader():
        left = paddle.reshape(left, shape=[-1, 1])
        pos_right = paddle.reshape(pos_right, shape=[-1, 1])
        neg_right = paddle.reshape(neg_right, shape=[-1, 1])
        net.train()
        global_step += 1
        left_feat, pos_score = net(left, pos_right)
        pred = pos_score
        _, neg_score = net(left, neg_right)
        avg_cost = loss.compute(pos_score, neg_score)
        losses.append(np.mean(avg_cost.numpy()))
        avg_cost.backward()
        optimizer.minimize(avg_cost)
        net.clear_gradients()
    paddle.enable_static()
    return losses


class TestSimnet(unittest.TestCase):
    def test_dygraph_static_same_loss(self):
        if paddle.is_compiled_with_cuda():
            paddle.fluid.set_flags({"FLAGS_cudnn_deterministic": True})
        conf_dict = create_conf_dict()
        dygraph_loss = train(conf_dict, to_static=False)
        static_loss = train(conf_dict, to_static=True)

        self.assertEqual(len(dygraph_loss), len(static_loss))
        for i in range(len(dygraph_loss)):
            self.assertAlmostEqual(dygraph_loss[i], static_loss[i])


if __name__ == '__main__':
    unittest.main()
