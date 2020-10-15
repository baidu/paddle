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
from paddle.fluid.tests.unittests.test_fusion_gru_op import fusion_gru
from paddle.fluid.tests.unittests.test_fusion_lstm_op import fc, ACTIVATION


class TestFusionGRUINT8MKLDNNOp(OpTest):
    def set_confs(self):
        pass

    def setUp(self):
        self.op_type = "fusion_gru"
        self.lod = [[2, 4, 3]]
        self.IC = 3
        self.OC = 5
        self.is_reverse = False
        self.with_h0 = False
        self.with_bias = True
        self.act_state = 'tanh'
        self.act_gate = 'sigmoid'
        self.origin_mode = True
        self.use_mkldnn = True
        self.force_fp32_output = True
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
        #  x_u8 = (x_f32 * scale_data + shift_data).astype(np.uint8)

        # WeightX/WeightH data
        wx = np.random.rand(self.IC, 3 * self.OC).astype('float32') * 2 - 1
        wh = np.random.rand(self.OC, 3 * self.OC).astype('float32') * 2 - 1

        # Calculating weight scales
        # scales = 63 / max(abs(channel_wise(weightsX + weightsH)))
        # WeightX data shape in PP: [IC, 3 * OC]
        # WeightH data shape in PP: [OC, 2 * OC] + [OC, OC]
        # Scales shape in oneDNN:   [3, OC]
        s8_max = 127.0
        scale_ur = s8_max / np.max(np.abs(
            np.concatenate(
                [
                    wx[:, :2 * self.OC], wh.flatten()[:2 * self.OC * self.OC]
                    .reshape(self.OC, 2 * self.OC)
                ],
                axis=0)),
                                   axis=0)
        scale_o = s8_max / np.max(np.abs(
            np.concatenate(
                [
                    wx[:, 2 * self.OC:], wh.flatten()[2 * self.OC * self.OC:]
                    .reshape(self.OC, self.OC)
                ],
                axis=0)),
                                  axis=0)

        scale_weights = np.concatenate([scale_ur, scale_o]).astype('float')

        bias = np.random.rand(
            1, 3 * self.OC).astype('float32') if self.with_bias else np.zeros(
                (1, 3 * self.OC), dtype='float32')
        h0 = np.random.rand(
            N, self.OC).astype('float32') if self.with_h0 else np.zeros(
                (N, self.OC), dtype='float32')

        _, _, _, hidden_f32 = fusion_gru(x_f32, self.lod, h0, wx, wh, bias,
                                         self.is_reverse, self.origin_mode,
                                         ACTIVATION[self.act_state],
                                         ACTIVATION[self.act_gate])

        self.inputs = {'X': (x_u8, self.lod), 'WeightX': wx, 'WeightH': wh}

        if self.with_bias:
            self.inputs['Bias'] = bias

        if self.with_h0:
            self.inputs['H0'] = h0

        if self.force_fp32_output:
            self.error_margin = 1e-1
            self.outputs = {'Hidden': (hidden_f32, self.lod)}
        else:
            self.error_margin = 1
            hidden_u8 = np.rint(hidden_f32 * scale_data + shift_data).astype(
                np.uint8)
            #  hidden_u8 = (hidden_f32 * scale_data + shift_data).astype(np.uint8)
            self.outputs = {'Hidden': (hidden_u8, self.lod)}

        self.attrs = {
            'activation': self.act_state,
            'gate_activation': self.act_gate,
            'is_reverse': self.is_reverse,
            'origin_mode': self.origin_mode,
            'use_mkldnn': self.use_mkldnn,
            'force_fp32_output': self.force_fp32_output,
            'Scale_data': scale_data,
            'Shift_data': shift_data,
            'Scale_weights': scale_weights
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False, atol=self.error_margin)


class TestFusionGRUINT8MKLDNNOp2(TestFusionGRUINT8MKLDNNOp):
    def set_confs(self):
        self.force_fp32_output = False


class TestFusionGRUINT8MKLDNNOp3(TestFusionGRUINT8MKLDNNOp):
    def set_confs(self):
        self.origin_mode = False


class TestFusionGRUINT8MKLDNNOp4(TestFusionGRUINT8MKLDNNOp):
    def set_confs(self):
        self.with_bias = False


class TestFusionGRUINT8MKLDNNOp5(TestFusionGRUINT8MKLDNNOp):
    def set_confs(self):
        self.with_h0 = False


if __name__ == "__main__":
    unittest.main()
