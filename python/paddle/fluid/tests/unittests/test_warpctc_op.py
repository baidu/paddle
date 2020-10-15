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

from __future__ import print_function

import sys
import unittest
import numpy as np
from op_test import OpTest
from test_softmax_op import stable_softmax
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import paddle
import paddle.nn.functional as F

paddle.enable_static()

CUDA_BLOCK_SIZE = 32


class CTCForward(object):
    def __init__(self, softmax, softmax_lod, labels, labels_lod, num_classes,
                 batch_size, blank, norm_by_times):
        self.softmax = softmax
        self.softmax_lod = softmax_lod
        self.labels = labels
        self.labels_lod = labels_lod
        self.blank = blank
        self.norm_by_times = norm_by_times

        self.level = 0
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.loss = np.zeros([self.batch_size, 1], dtype=softmax.dtype)
        self.gradient = np.zeros(self.softmax.shape, dtype=softmax.dtype)

        # float64
        self.EXP_MAX = sys.float_info.max
        self.EXP_MIN = sys.float_info.min
        self.LOG_ZERO = np.log(self.EXP_MIN)
        self.LOG_INFINITY = np.log(self.EXP_MAX)

    def safe_exp(self, x):
        if x <= self.LOG_ZERO:
            return 0.0
        if x >= self.LOG_INFINITY:
            return self.EXP_MAX
        return np.exp(x)

    def safe_log(self, x):
        if x <= self.EXP_MIN:
            return self.LOG_ZERO
        return np.log(x)

    # x = lna and y = lnb are in log scale, ln(a / b) = lna - lnb
    def log_div(self, x, y):
        res = x - y
        if res <= self.LOG_ZERO:
            return self.LOG_ZERO
        if res >= self.LOG_INFINITY:
            return self.LOG_INFINITY
        return res

    # x = lna and y = lnb are in log scale, ln(a * b) = lna + lnb
    def log_mul(self, x, y):
        res = x + y
        if res <= self.LOG_ZERO:
            return self.LOG_ZERO
        if res >= self.LOG_INFINITY:
            return self.LOG_INFINITY
        return res

    # x = lna and y = lnb are in log scale,
    # ln(a + b) = lna + ln(1 + exp(lnb - lna)), where b > a
    def log_add(self, x, y):
        if x < y:
            t = y
            y = x
            x = t
        return x + self.safe_log(1 + self.safe_exp(y - x))

    def segment_range(self, time, total_times, total_segments):
        start = max(0, total_segments - (2 * (total_times - time)))
        end = min(total_segments, 2 * (time + 1))
        return start, end

    def forward_a_sequence(self, softmax_a_sequence, labels_a_sequence):
        total_times = softmax_a_sequence.shape[0]
        total_segments = labels_a_sequence.shape[0] * 2 + 1

        required_times = labels_a_sequence.shape[0]
        old_label = -1
        for i in range(labels_a_sequence.shape[0]):
            # two contingous labels with the same value
            if labels_a_sequence[i, 0] == old_label:
                required_times = required_times + 1
            old_label = labels_a_sequence[i, 0]

        if total_times < required_times:
            return 0

        # calculate the forward and backward variables,
        # reference Chapter 7.3 of "Alex Grave, Supervised Sequence
        # Labelling with Recurrent Neural Networks"
        log_acts = np.zeros(
            [total_times, self.num_classes], dtype=softmax_a_sequence.dtype)
        for i in range(total_times):
            for j in range(self.num_classes):
                log_acts[i, j] = self.safe_log(softmax_a_sequence[i, j])

        # calculate the forward variables
        forward_vars = np.zeros(
            [total_times, total_segments], dtype=softmax_a_sequence.dtype)
        for i in range(total_times):
            for j in range(total_segments):
                forward_vars[i, j] = self.LOG_ZERO

        for i in range(total_times):
            # dp initialization at t0
            if i == 0:
                forward_vars[i, 0] = log_acts[0, self.blank]
                if total_segments > 1:
                    forward_vars[i, 1] = log_acts[0, labels_a_sequence[i, 0]]
                continue

            # dp from t1
            start, end = self.segment_range(i, total_times, total_segments)
            for k in range(end - start):
                j = k + start
                if j & 1 == 1:
                    label_idx = j // 2
                    label_val = labels_a_sequence[label_idx, 0]
                    fv = self.log_add(forward_vars[i - 1, j],
                                      forward_vars[i - 1, j - 1])
                    if j > 1 and label_val != labels_a_sequence[label_idx - 1,
                                                                0]:
                        fv = self.log_add(fv, forward_vars[i - 1, j - 2])
                    fv = self.log_mul(fv, log_acts[i, label_val])
                else:
                    fv = forward_vars[i - 1, j]
                    if j > 0:
                        fv = self.log_add(fv, forward_vars[i - 1, j - 1])
                    fv = self.log_mul(fv, log_acts[i, self.blank])
                forward_vars[i, j] = fv

        # sum the last two value as log_prob
        log_prob = forward_vars[total_times - 1, total_segments - 1]
        if total_segments > 1:
            log_prob = self.log_add(
                log_prob, forward_vars[total_times - 1, total_segments - 2])

        return -log_prob

    def forward(self):
        softmax_offset = 0
        labels_offset = 0
        for i in range(self.batch_size):
            if self.labels.shape[1] == 1:
                softmax_start_i = softmax_offset
                softmax_end_i = softmax_offset + self.softmax_lod[self.level][i]
                labels_start_i = labels_offset
                labels_end_i = labels_offset + self.labels_lod[self.level][i]

                softmax_a_sequence = self.softmax[softmax_start_i:
                                                  softmax_end_i, :]
                labels_a_sequence = self.labels[labels_start_i:labels_end_i, :]
                self.loss[i] = self.forward_a_sequence(softmax_a_sequence,
                                                       labels_a_sequence)
                softmax_offset += self.softmax_lod[self.level][i]
                labels_offset += self.labels_lod[self.level][i]
            else:
                softmax_a_sequence = self.softmax[:self.softmax_lod[i], i, :]
                labels_a_sequence = self.labels[:self.labels_lod[i], :]
                self.loss[i] = self.forward_a_sequence(softmax_a_sequence,
                                                       labels_a_sequence)

        return self.loss


class TestWarpCTCOp(OpTest):
    def config(self):
        self.batch_size = 4
        self.num_classes = 12
        self.logits_lod = [[4, 1, 3, 3]]
        self.labels_lod = [[3, 1, 4, 4]]
        self.blank = self.num_classes - 1
        self.norm_by_times = False

    def setUp(self):
        self.op_type = "warpctc"
        self.config()

        logits = np.random.uniform(
            0.1, 1.0,
            [sum(self.logits_lod[0]), self.num_classes]).astype("float32")
        softmax = np.apply_along_axis(stable_softmax, 1, logits)
        # labels should not be blank
        labels = np.random.randint(
            0,
            self.num_classes - 1, [sum(self.labels_lod[0]), 1],
            dtype="int32")

        ctc = CTCForward(softmax, self.logits_lod, labels, self.labels_lod,
                         self.num_classes, self.batch_size, self.blank,
                         self.norm_by_times)
        loss = ctc.forward()

        max_sequence_length = 0
        for i in range(self.batch_size):
            max_sequence_length = max(max_sequence_length,
                                      self.logits_lod[0][i])
        self.gradient = np.zeros(
            [max_sequence_length, self.batch_size, self.num_classes],
            dtype=logits.dtype)

        self.inputs = {
            "Logits": (logits, self.logits_lod),
            "Label": (labels, self.labels_lod)
        }
        self.outputs = {"Loss": loss}
        self.attrs = {
            "blank": self.blank,
            "norm_by_times": self.norm_by_times,
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.outputs['WarpCTCGrad'] = self.gradient
        self.check_grad(
            ["Logits"], "Loss", max_relative_error=0.007, check_dygraph=False)


class TestWarpCTCOpCase1(TestWarpCTCOp):
    def config(self):
        self.batch_size = 4
        self.num_classes = CUDA_BLOCK_SIZE + 2
        self.logits_lod = [[4, 1, 3, 3]]
        self.labels_lod = [[3, 1, 4, 4]]
        self.blank = self.num_classes - 1
        self.norm_by_times = False


class TestWarpCTCOpWithPadding(OpTest):
    def config(self):
        self.batch_size = 4
        self.num_classes = 8
        self.logits_lod = [[4, 1, 3, 3]]
        self.labels_lod = [[3, 1, 4, 4]]
        self.logits_length = np.array([4, 1, 3, 3], dtype=np.int64)
        self.labels_length = np.array([3, 1, 4, 4], dtype=np.int64)
        self.blank = self.num_classes - 1
        self.norm_by_times = False

    def setUp(self):
        self.op_type = "warpctc"
        self.config()

        logits = np.random.uniform(
            0.1, 1.0,
            [sum(self.logits_length), self.num_classes]).astype("float32")
        softmax = np.apply_along_axis(stable_softmax, 1, logits)
        # labels should not be blank
        labels = np.random.randint(
            0,
            self.num_classes - 1, [sum(self.labels_length), 1],
            dtype="int32")

        ctc = CTCForward(softmax, self.logits_lod, labels, self.labels_lod,
                         self.num_classes, self.batch_size, self.blank,
                         self.norm_by_times)
        loss = ctc.forward()

        max_sequence_length = 0
        for i in range(self.batch_size):
            max_sequence_length = max(max_sequence_length,
                                      self.logits_length[i])
        # reshape logits to T*N*S
        new_logits = np.zeros(
            [max_sequence_length, self.batch_size, self.num_classes],
            dtype=logits.dtype)

        cur = 0
        for batch_id in range(self.batch_size):
            for i in range(self.logits_length[batch_id]):
                for j in range(self.num_classes):
                    new_logits[i, batch_id, j] = logits[cur + i, j]
            cur = cur + self.logits_length[batch_id]

        # reshape labels to N*S
        max_target_seq_length = 0
        for i in range(self.batch_size):
            max_target_seq_length = max(max_target_seq_length,
                                        self.labels_length[i])
        new_labels = np.zeros(
            [self.batch_size, max_target_seq_length], dtype="int32")

        cur = 0
        for batch_id in range(self.batch_size):
            for i in range(self.labels_length[batch_id]):
                new_labels[batch_id, i] = labels[cur + i]
            cur = cur + self.labels_length[batch_id]

        self.gradient = np.zeros(
            [max_sequence_length, self.batch_size, self.num_classes],
            dtype=logits.dtype)

        self.inputs = {
            "Logits": new_logits,
            "Label": new_labels,
            "LogitsLength": self.logits_length,
            "LabelLength": self.labels_length
        }
        self.outputs = {"Loss": loss}
        self.attrs = {
            "blank": self.blank,
            "norm_by_times": self.norm_by_times,
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.outputs['WarpCTCGrad'] = self.gradient
        self.check_grad(
            ["Logits"], "Loss", max_relative_error=0.007, check_dygraph=False)


class TestWarpCTCOpWithPaddingCase1(TestWarpCTCOpWithPadding):
    def config(self):
        self.batch_size = 4
        self.num_classes = CUDA_BLOCK_SIZE + 2
        self.logits_lod = [[4, 1, 3, 3]]
        self.labels_lod = [[3, 1, 4, 4]]
        self.logits_length = np.array([4, 1, 3, 3], dtype=np.int64)
        self.labels_length = np.array([3, 1, 4, 4], dtype=np.int64)
        self.blank = self.num_classes - 1
        self.norm_by_times = False


class TestWarpCTCOpFp64(OpTest):
    def config(self):
        self.batch_size = 4
        self.num_classes = 8
        self.logits_lod = [[4, 1, 5, 5]]
        self.labels_lod = [[3, 1, 4, 2]]
        self.logits_length = np.array([4, 1, 5, 5], dtype=np.int64)
        self.labels_length = np.array([3, 1, 4, 2], dtype=np.int64)
        self.blank = self.num_classes - 1
        self.norm_by_times = False

    def setUp(self):
        self.op_type = "warpctc"
        self.config()

        logits = np.random.uniform(
            0.1, 1.0,
            [sum(self.logits_length), self.num_classes]).astype("float64")
        softmax = np.apply_along_axis(stable_softmax, 1, logits)
        # labels should not be blank
        labels = np.random.randint(
            0,
            self.num_classes - 1, [sum(self.labels_length), 1],
            dtype="int32")

        ctc = CTCForward(softmax, self.logits_lod, labels, self.labels_lod,
                         self.num_classes, self.batch_size, self.blank,
                         self.norm_by_times)
        loss = ctc.forward()

        max_sequence_length = 0
        for i in range(self.batch_size):
            max_sequence_length = max(max_sequence_length,
                                      self.logits_length[i])
        # reshape logits to T*N*S
        new_logits = np.zeros(
            [max_sequence_length, self.batch_size, self.num_classes],
            dtype=logits.dtype)

        cur = 0
        for batch_id in range(self.batch_size):
            for i in range(self.logits_length[batch_id]):
                for j in range(self.num_classes):
                    new_logits[i, batch_id, j] = logits[cur + i, j]
            cur = cur + self.logits_length[batch_id]

        # reshape labels to N*S
        max_target_seq_length = 0
        for i in range(self.batch_size):
            max_target_seq_length = max(max_target_seq_length,
                                        self.labels_length[i])
        new_labels = np.zeros(
            [self.batch_size, max_target_seq_length], dtype="int32")

        cur = 0
        for batch_id in range(self.batch_size):
            for i in range(self.labels_length[batch_id]):
                new_labels[batch_id, i] = labels[cur + i]
            cur = cur + self.labels_length[batch_id]

        self.gradient = np.zeros(
            [max_sequence_length, self.batch_size, self.num_classes],
            dtype=logits.dtype)

        self.inputs = {
            "Logits": new_logits,
            "Label": new_labels,
            "LogitsLength": self.logits_length,
            "LabelLength": self.labels_length
        }
        self.outputs = {"Loss": loss}
        self.attrs = {
            "blank": self.blank,
            "norm_by_times": self.norm_by_times,
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.outputs['WarpCTCGrad'] = self.gradient
        self.check_grad(["Logits"], "Loss")


class TestWarpCTCOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            logits = fluid.data(
                name='logits', shape=[5, 16, 6], dtype='float32')
            logits_length = fluid.data(
                name='logits_length', shape=[None], dtype='int64')
            label = fluid.data(name='label', shape=[16, 3], dtype='int32')
            label_length = fluid.data(
                name='labels_length', shape=[None], dtype='int64')

            def test_logits_Variable():
                logits_data = np.random.rand(5, 16, 6).astype(logits.dtype)
                fluid.layers.warpctc(
                    input=logits_data,
                    label=label,
                    input_length=logits_length,
                    label_length=label_length)

            self.assertRaises(TypeError, test_logits_Variable)

            def test_label_Variable():
                label_data = np.random.randint(0, 5, [5, 1]).astype("int32")
                fluid.layers.warpctc(
                    input=logits,
                    label=label_data,
                    input_length=logits_length,
                    label_length=label_length)

            self.assertRaises(TypeError, test_label_Variable)

            def test_logits_len_Variable():
                logits_length_data = np.array([5] * 16).astype("int64")
                fluid.layers.warpctc(
                    input=logits,
                    label=label,
                    input_length=logits_length_data,
                    label_length=label_length)

            self.assertRaises(TypeError, test_logits_len_Variable)

            def test_label_len_Variable():
                label_length_data = np.array([3] * 16).astype("int64")
                fluid.layers.warpctc(
                    input=logits,
                    label=label,
                    input_length=logits_length,
                    label_length=label_length_data)

            self.assertRaises(TypeError, test_label_len_Variable)

    def test_dygraph_errors(self):
        def test_dygraph_with_lod():

            logits = np.random.uniform(0.1, 1.0, [20, 15]).astype("float32")
            # labels should not be blank
            labels = np.random.randint(0, 15 - 1, [15, 1], dtype="int32")
            softmax = paddle.to_tensor(logits)
            labels = paddle.to_tensor(labels)

            fluid.layers.warpctc(input=softmax, label=labels)

        paddle.disable_static()
        self.assertRaises(ValueError, test_dygraph_with_lod)
        paddle.enable_static()


class TestCTCLossAPICase(unittest.TestCase):
    def test_functinal_api(self):
        self.batch_size = 4
        self.num_classes = CUDA_BLOCK_SIZE + 2
        self.logits_length = np.array([4, 1, 3, 3], dtype=np.int64)
        self.labels_length = np.array([3, 1, 4, 4], dtype=np.int64)
        self.blank = self.num_classes - 1
        self.norm_by_times = False

        logits = np.random.uniform(0.1, 1.0, [
            max(self.logits_length), self.batch_size, self.num_classes
        ]).astype("float32")
        softmax = np.apply_along_axis(stable_softmax, -1, logits)
        # labels should not be blank
        labels = np.random.randint(
            0,
            self.num_classes - 1, [self.batch_size, max(self.labels_length)],
            dtype="int32")

        ctc = CTCForward(softmax, self.logits_length, labels,
                         self.labels_length, self.num_classes, self.batch_size,
                         self.blank, self.norm_by_times)
        loss_np = ctc.forward()

        paddle.disable_static()
        softmax = paddle.to_tensor(logits)
        labels = paddle.to_tensor(labels)
        logits_length = paddle.to_tensor(self.logits_length)
        labels_length = paddle.to_tensor(self.labels_length)
        loss_pd_mean = F.ctc_loss(
            softmax,
            labels,
            logits_length,
            labels_length,
            blank=self.blank,
            reduction='mean')
        loss_pd_mean = loss_pd_mean.numpy()

        loss_pd_sum = F.ctc_loss(
            softmax,
            labels,
            logits_length,
            labels_length,
            blank=self.blank,
            reduction='sum')
        loss_pd_sum = loss_pd_sum.numpy()
        paddle.enable_static()
        loss_np = np.squeeze(loss_np, axis=-1)
        loss_np_mean = (loss_np / labels_length.numpy()).mean()
        loss_np_sum = loss_np.sum()

        self.assertTrue(np.allclose(loss_pd_mean, loss_np_mean, atol=1))
        self.assertTrue(np.allclose(loss_pd_sum, loss_np_sum, atol=1))

    def test_class_api(self):
        self.batch_size = 3
        self.num_classes = 15
        self.logits_length = np.array([3, 3, 3], dtype=np.int64)
        self.labels_length = np.array([0, 1, 2], dtype=np.int64)
        self.blank = 0
        self.norm_by_times = False

        logits = np.random.uniform(0.1, 1.0, [
            max(self.logits_length), self.batch_size, self.num_classes
        ]).astype("float32")
        softmax = np.apply_along_axis(stable_softmax, -1, logits)
        # labels should not be blank
        labels = np.random.randint(
            1,
            self.num_classes, [self.batch_size, max(self.labels_length)],
            dtype="int32")

        ctc = CTCForward(softmax, self.logits_length, labels,
                         self.labels_length, self.num_classes, self.batch_size,
                         self.blank, self.norm_by_times)
        loss_np = ctc.forward()

        paddle.disable_static()
        softmax = paddle.to_tensor(logits)
        labels = paddle.to_tensor(labels)
        logits_length = paddle.to_tensor(self.logits_length)
        labels_length = paddle.to_tensor(self.labels_length)

        loss_pd = paddle.nn.CTCLoss(self.blank, 'none')(
            softmax, labels, logits_length, labels_length)
        loss_pd = loss_pd.numpy()
        paddle.enable_static()
        loss_np = np.squeeze(loss_np, axis=-1)

        self.assertTrue(np.allclose(loss_pd, loss_np, atol=1))


if __name__ == "__main__":
    unittest.main()
