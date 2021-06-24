#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
import struct
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
from paddle.fluid.tests.unittests.test_fusion_lstm_op import TestFusionLSTMOp, fc, ACTIVATION, fusion_lstm
from paddle.fluid.tests.unittests.test_fusion_gru_op import fusion_gru


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestFusionLSTMBF16ONEDNNOp(OpTest):
    def set_confs(self):
        pass

    def test_check_output(self):
        for use_seq in {True, False}:
            self.attrs['use_seq'] = use_seq
            self.check_output(
                check_dygraph=False, no_check_set=["Cell"], atol=2e-2)

    def setUp(self):
        self.op_type = 'fusion_lstm'
        self.lod = [[2, 3, 5, 4]]
        self.M = 8
        self.D = 16
        self.has_initial_state = False
        self.use_peepholes = False
        self.is_reverse = False
        self._cpu_only = True
        self.act_gate = 'sigmoid'
        self.act_cell = 'tanh'
        self.act_cand = 'tanh'
        self.use_mkldnn = True
        self.mkldnn_data_type = "bfloat16"
        self.force_fp32_output = False
        self.weights_dtype = 'fp32'
        self.set_confs()

        T = sum(self.lod[0])
        bs = len(self.lod[0])

        # fp32 X input for reference implementation and
        # corressponding bf16 data as input to LSTM oneDNN bf16 kernel
        x = np.random.normal(size=(T, self.M)).astype('float32')

        x_bf16 = convert_float_to_uint16(x)

        if self.has_initial_state:
            h0 = np.random.normal(size=(bs, self.D)).astype('float32')
            c0 = np.random.normal(size=(bs, self.D)).astype('float32')
        else:
            h0 = np.zeros((bs, self.D)).astype('float32')
            c0 = np.zeros((bs, self.D)).astype('float32')

        wh = np.random.normal(size=(self.D, 4 * self.D)).astype('float32')

        h0_bf16 = convert_float_to_uint16(h0)

        if self.use_peepholes:
            b = np.random.normal(size=(1, 7 * self.D)).astype('float32')
        else:
            b = np.random.normal(size=(1, 4 * self.D)).astype('float32')
        w_b = np.copy(b[:, 0:4 * self.D])
        w_c = b[:, 4 * self.D:] if self.use_peepholes else None

        wx = np.random.normal(size=(self.M, 4 * self.D)).astype('float32')

        wx_bf16 = convert_float_to_uint16(wx)
        wh_bf16 = convert_float_to_uint16(wh)

        bx = np.random.normal(size=(1, 4 * self.D)).astype('float32')
        b[0, 0:4 * self.D] += bx[0, :]

        hidden, c = fusion_lstm(x, self.lod, wx, bx, h0, c0, wh, w_b, w_c,
                                self.is_reverse, ACTIVATION[self.act_gate],
                                ACTIVATION[self.act_cell],
                                ACTIVATION[self.act_cand])

        hidden = hidden.astype('float32')
        hidden_bf16 = convert_float_to_uint16(hidden)

        if self.weights_dtype == 'bf16':
            self.inputs = {
                'X': (x_bf16, self.lod),
                'WeightX': wx_bf16,
                'WeightH': wh_bf16,
                'Bias': b
            }
        elif self.weights_dtype == 'fp32':
            self.inputs = {
                'X': (x_bf16, self.lod),
                'WeightX': wx,
                'WeightH': wh,
                'Bias': b
            }

        if self.has_initial_state:
            if self.weights_dtype == 'bf16':
                self.inputs['H0'] = h0_bf16
            elif self.weights_dtype == 'fp32':
                self.inputs['H0'] = h0

            self.inputs['C0'] = c0

        self.outputs = {
            'Hidden': (hidden, self.lod),
            'Cell': (c, self.lod),
        }

        self.attrs = {
            'use_peepholes': self.use_peepholes,
            'is_reverse': self.is_reverse,
            'gate_activation': self.act_gate,
            'cell_activation': self.act_cell,
            'candidate_activation': self.act_cand,
            'force_fp32_output': self.force_fp32_output,
            'use_mkldnn': self.use_mkldnn,
            'mkldnn_data_type': self.mkldnn_data_type,
        }


class TestFusionLSTMBF16ONEDNNPeepholesOp(TestFusionLSTMBF16ONEDNNOp):
    def set_confs(self):
        self.use_peepholes = True


class TestFusionLSTMBF16ONEDNNInitializedStateOp(TestFusionLSTMBF16ONEDNNOp):
    def set_confs(self):
        self.has_initial_state = True


class TestFusionLSTMBF16ONEDNNReverseOp(TestFusionLSTMBF16ONEDNNOp):
    def set_confs(self):
        self.is_reverse = True


class TestFusionLSTMBF16ONEDNNBF16WeightsOp(TestFusionLSTMBF16ONEDNNOp):
    def set_confs(self):
        self.weights_dtype = 'bf16'


if __name__ == "__main__":
    from paddle import enable_static
    enable_static()
    unittest.main()
