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
from paddle.fluid.tests.unittests.op_test import (
    OpTest, convert_float_to_uint16, convert_uint16_to_float,
    skip_check_grad_ci)
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle import enable_static


def _lookup(weights, ids, flat_ids, op_version="lookup_table"):
    w_shape = weights.shape
    out_shape = list(ids.shape[:-1]) if op_version is "lookup_table" else list(
        ids.shape)
    out_shape.append(w_shape[-1])
    out = weights[flat_ids].reshape(out_shape)
    return out


def _get_grad(weights, ids, flat_ids, op_version="lookup_table"):
    w_shape = weights.shape
    w_grad = np.zeros((w_shape), dtype=weights.dtype)
    out_shape = list(ids.shape[:-1]) if op_version is "lookup_table" else list(
        ids.shape)
    out_grad_shape = (np.prod(out_shape), w_shape[-1])
    out_grad = weights[flat_ids].reshape(out_grad_shape)
    for i, idx in enumerate(flat_ids):
        w_grad[idx, :] += out_grad[i]
    return w_grad


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestLookupTableBF16Op(OpTest):
    def init_test(self):
        self.op_type = "lookup_table"
        self.ids_shape = (4, 1)

    def setUp(self):
        self.init_test()
        self.dtype = np.uint16

        table = np.random.random((17, 31)).astype("float32")
        self.ids = np.random.randint(0, 17, self.ids_shape).astype("int64")
        self.flat_ids = self.ids.flatten()

        self.w_bf16 = convert_float_to_uint16(table)
        self.out_bf16 = _lookup(self.w_bf16, self.ids, self.flat_ids,
                                self.op_type)
        self.out_fp32 = _lookup(table, self.ids, self.flat_ids, self.op_type)
        self.w_grad_fp32 = _get_grad(table, self.ids, self.flat_ids,
                                     self.op_type)

        self.inputs = {'W': self.w_bf16, 'Ids': self.ids}
        self.outputs = {'Out': self.out_fp32}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), check_dygraph=False)

    def test_check_grad(self):
        self.check_grad_with_place(
            core.CPUPlace(), ['W'],
            'Out',
            no_grad_set=set('Ids'),
            check_dygraph=False,
            max_relative_error=1.5e-2,
            user_defined_grads=[self.w_grad_fp32],
            user_defined_grad_outputs=[self.out_bf16])


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestLookupTableBF16OpIds4D(TestLookupTableBF16Op):
    def init_test(self):
        self.op_type = "lookup_table"
        self.ids_shape = (2, 4, 5, 1)


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestLookupTableBF16OpWIsSelectedRows(unittest.TestCase):
    def init_test(self):
        self.op_type = "lookup_table"
        self.ids_shape = (10, 1)

    def setUp(self):
        self.init_test()
        self.ids = np.random.randint(
            low=0, high=15, size=self.ids_shape).astype("int64")
        self.flat_ids = self.ids.flatten()
        self.w_fp32 = np.random.random((15, 32)).astype("float32")
        self.w_bf16 = convert_float_to_uint16(self.w_fp32)
        self.scope = core.Scope()
        self.place = core.CPUPlace()

    def prepare_w(self):
        rows = [a for a in range(self.w_bf16.shape[0])]
        row_numel = self.w_bf16.shape[1]

        w_selected_rows = self.scope.var('W').get_selected_rows()
        w_selected_rows.set_height(len(rows))
        w_selected_rows.set_rows(rows)
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(self.w_bf16, self.place)

    def prepare_ids(self):
        ids_tensor = self.scope.var('Ids').get_tensor()
        ids_tensor.set(self.ids, self.place)

    def _check_output(self, reference, result_array):
        result_array_fp32 = convert_uint16_to_float(result_array)
        np.testing.assert_allclose(result_array_fp32, reference, rtol=1.5e-2)

    def test_check_output(self):
        self.prepare_ids()
        self.prepare_w()
        out_tensor = self.scope.var('Out').get_tensor()

        # create and run lookup_table operator
        lookup_table = Operator(self.op_type, W='W', Ids='Ids', Out='Out')
        lookup_table.run(self.scope, self.place)

        # get result from Out
        result_array = np.array(out_tensor)
        ref = _lookup(self.w_fp32, self.ids, self.flat_ids, self.op_type)
        self._check_output(ref, result_array)


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestLookupTableBF16OpWIsSelectedRows4DIds(
        TestLookupTableBF16OpWIsSelectedRows):
    def init_test(self):
        self.op_type = "lookup_table"
        self.ids_shape = (3, 4, 5, 1)

    def setUp(self):
        super(TestLookupTableBF16OpWIsSelectedRows4DIds, self).setUp()
        self.flat_ids = self.ids.flatten()


@skip_check_grad_ci(
    reason="Since paddings are not trainable and fixed in forward,"
    "the gradient of paddings makes no sense and we don't "
    "test the gradient here.")
@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestLookupTableBF16OpWithPadding(TestLookupTableBF16Op):
    def test_check_output(self):
        ids = np.squeeze(self.inputs['Ids'])
        padding_idx = np.random.choice(ids, 1)[0]
        self.outputs['Out'][ids == padding_idx] = np.zeros(31)
        self.attrs = {'padding_idx': int(padding_idx)}
        self.check_output_with_place(core.CPUPlace(), check_dygraph=False)


@skip_check_grad_ci(
    reason="Since paddings are not trainable and fixed in forward,"
    "the gradient of paddings makes no sense and we don't "
    "test the gradient here.")
@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestLookupTableBF16OpIds4DPadding(TestLookupTableBF16OpIds4D):
    def test_check_output(self):
        ids = self.inputs['Ids']
        flatten_idx = ids.flatten()
        padding_idx = np.random.choice(flatten_idx, 1)[0]
        self.outputs['Out'][np.squeeze(ids == padding_idx)] = np.zeros(31)
        self.attrs = {'padding_idx': int(padding_idx)}
        self.check_output_with_place(core.CPUPlace(), check_dygraph=False)


class TestEmbeddingLayerBF16ConstantInitializer(unittest.TestCase):
    """
    Test embedding layer api and results for bfloat16
    """

    def set_initializer(self):
        self.initializer = fluid.initializer.Constant(value=self.value)

    def setUp(self):
        self.ids_shape = [4, 1]
        self.w_shape = [10, 64]
        self.ids = np.random.randint(
            low=0, high=9, size=self.ids_shape).astype("int64")
        self.flat_ids = self.ids.flatten()
        self.value = 3.0
        self.w_fp32 = np.full(self.w_shape, self.value)
        self.place = fluid.CPUPlace()
        self.prog = fluid.Program()
        self.startup_prog = fluid.Program()
        self.set_initializer()

        with fluid.program_guard(self.prog, self.startup_prog):
            x = fluid.layers.data(name='x', shape=self.ids_shape, dtype='int64')
            self.emb = fluid.layers.embedding(
                input=x,
                size=self.w_shape,
                param_attr=fluid.ParamAttr(
                    name="emb_weight", initializer=self.initializer),
                is_sparse=False,
                dtype="uint16")  # bfloat16
        exe = fluid.Executor(self.place)
        exe.run(self.startup_prog)
        self.result = exe.run(self.prog,
                              feed={'x': self.ids},
                              fetch_list=['emb_weight', self.emb])

    def test_embedding_weights(self):
        result = convert_uint16_to_float(self.result[0])
        self.assertTrue(np.array_equal(self.w_fp32, result))

    def test_lookup_results(self):
        lookup_result = convert_uint16_to_float(self.result[1])
        lookup_ref = _lookup(self.w_fp32, self.ids, self.flat_ids)
        self.assertTrue(np.array_equal(lookup_result, lookup_ref))


if __name__ == "__main__":
    enable_static()
    unittest.main()
