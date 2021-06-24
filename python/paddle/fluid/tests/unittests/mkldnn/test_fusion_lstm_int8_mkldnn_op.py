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

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.test_fusion_lstm_op import fc, ACTIVATION, fusion_lstm


class TestFusionLSTMINT8MKLDNNOp(OpTest):
    def set_confs(self):
        pass

    def setUp(self):
        self.op_type = "fusion_lstm"
        self.lod = [[2, 3, 5, 4]]
        self.IC = 3
        self.OC = 5
        self.is_reverse = False
        self.has_initial_state = False
        self.act_cell = 'tanh'
        self.act_gate = 'sigmoid'
        self.act_cand = 'tanh'
        self.use_peepholes = False  # LSTM u8 doesn't support peepholes
        self.use_mkldnn = True
        self.mkldnn_data_type = "int8"
        self.force_fp32_output = False
        self.error_margin = 1e-5
        self.set_confs()

        # RNN dimensions
        T = sum(self.lod[0])
        N = len(self.lod[0])

        # Input data
        x_f32 = np.random.rand(T, self.IC).astype('float32') * 2 - 1
        scale_data = 63.0
        shift_data = 64.0
        x_u8 = np.rint(x_f32 * scale_data + shift_data).astype(np.uint8)

        # WeightX/WeightH data
        wx = np.random.rand(self.IC, 4 * self.OC).astype('float32') * 2 - 1
        wh = np.random.rand(self.OC, 4 * self.OC).astype('float32') * 2 - 1

        # Calculating weight scales
        # scales = 127 / max(abs(channel_wise(weightsX + weightsH)))
        s8_max = 127.0

        scale_weights = s8_max / np.max(
            np.abs(np.concatenate(
                [wx[:, :], wh[:, :]], axis=0)), axis=0)

        scale_weights = scale_weights.astype('float')

        if self.use_peepholes:
            b = np.random.rand(1, 7 * self.OC).astype('float32')
        else:
            b = np.random.rand(1, 4 * self.OC).astype('float32')
        w_b = np.copy(b[:, 0:4 * self.OC])
        w_c = b[:, 4 * self.OC:] if self.use_peepholes else None

        bx = np.random.normal(size=(1, 4 * self.OC)).astype('float32')
        b[0, 0:4 * self.OC] += bx[0, :]

        if self.has_initial_state:
            h0 = np.random.rand(N, self.OC).astype('float32')
            c0 = np.random.rand(N, self.OC).astype('float32')
        else:
            h0 = np.zeros((N, self.OC)).astype('float32')
            c0 = np.zeros((N, self.OC)).astype('float32')

        hidden_f32, c = fusion_lstm(
            x_f32, self.lod, wx, bx, h0, c0, wh, w_b, w_c, self.is_reverse,
            ACTIVATION[self.act_gate], ACTIVATION[self.act_cell],
            ACTIVATION[self.act_cand])

        self.inputs = {
            'X': (x_u8, self.lod),
            'WeightX': wx,
            'WeightH': wh,
            'Bias': b
        }

        if self.has_initial_state:
            self.inputs['H0'] = h0
            self.inputs['C0'] = c0

        if self.force_fp32_output:
            self.error_margin = 1e-1
            self.outputs = {
                'Hidden': (hidden_f32, self.lod),
                'Cell': (c, self.lod)
            }
        else:
            self.error_margin = 2
            hidden_u8 = np.rint(hidden_f32 * scale_data + shift_data).astype(
                np.uint8)
            self.outputs = {
                'Hidden': (hidden_u8, self.lod),
                'Cell': (c, self.lod)
            }

        self.attrs = {
            'gate_activation': self.act_gate,
            'cell_activation': self.act_cell,
            'candidate_activation': self.act_cand,
            'is_reverse': self.is_reverse,
            'use_peepholes': self.use_peepholes,
            'use_mkldnn': self.use_mkldnn,
            'mkldnn_data_type': self.mkldnn_data_type,
            'force_fp32_output': self.force_fp32_output,
            'Scale_data': scale_data,
            'Shift_data': shift_data,
            'Scale_weights': scale_weights
        }

    def test_check_output(self):
        for use_seq in {True, False}:
            self.attrs['use_seq'] = use_seq
            self.check_output(
                check_dygraph=False,
                no_check_set=["Cell"],
                atol=self.error_margin)


class TestFusionLSTMINT8MKLDNNOp2(TestFusionLSTMINT8MKLDNNOp):
    def set_confs(self):
        self.force_fp32_output = True


class TestFusionLSTMINT8MKLDNNOp4(TestFusionLSTMINT8MKLDNNOp):
    def set_confs(self):
        self.is_reverse = True


class TestFusionLSTMINT8MKLDNNOp5(TestFusionLSTMINT8MKLDNNOp):
    def set_confs(self):
        self.has_initial_state = True


if __name__ == "__main__":
    from paddle import enable_static
    enable_static()
    unittest.main()
