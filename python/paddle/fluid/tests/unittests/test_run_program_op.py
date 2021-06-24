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

from __future__ import print_function

import contextlib
import unittest
import numpy as np
import six

import paddle
import paddle.fluid as fluid
from paddle import compat as cpt
from paddle.fluid import core, framework, executor

paddle.enable_static()


@contextlib.contextmanager
def program_scope_guard():
    prog = fluid.Program()
    startup_prog = fluid.Program()
    scope = fluid.core.Scope()
    with fluid.scope_guard(scope):
        with fluid.program_guard(prog, startup_prog):
            with fluid.unique_name.guard():
                yield


# NOTE: Because RunProgramOp has a special output of type std::vector<Scope *>, 
# the OpTest cannot be used in RunProgramOp. The variable type cannot be specified
# when creating output variables in OpTest, default type is LoDTensor
# NOTE: the gradient test method in OpTest also cannot be used for RunProgramOp,
# because it hold BlockDesc type attr, OperatorFactory can't parse this attr type
# when create Operator, so here compare gradients with static graph
# NOTE: Here rewrite a simple unittest framework for RunProgramOp
class RunProgramOpTest(unittest.TestCase):
    def build_model(self):
        raise NotImplementedError(
            "RunProgramOp test should implement build_model")

    def check_output(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            # TODO: RunProgramOp is not recommended for use in static mode now
            self.expect_outs = self.run_static_model(place, is_test=True)
            self.check_output_with_place(place)

    def check_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            # TODO: RunProgramOp is not recommended for use in static mode now
            self.expect_grads = self.run_static_model(place, is_test=False)
            self.check_grad_with_place(place)

    def run_static_model(self, place, is_test=True):
        with program_scope_guard():
            startup_program = fluid.default_startup_program()
            main_program = fluid.default_main_program()

            self.build_model()

            exe = fluid.Executor(place)
            exe.run(startup_program)

            if is_test:
                fetch_list = self.output_names['Out']
            else:
                fetch_list = self.get_param_grad_names()

            outs = exe.run(main_program,
                           feed=self.inputs['X'],
                           fetch_list=fetch_list)
            return outs

    def get_program_desc(self):
        with program_scope_guard():
            fwd_op_num = self.build_model()
            return fluid.default_main_program().desc, fwd_op_num

    def prepare_attrs(self):
        return {
            'global_block': self.program_desc.block(0),
            'start_op_index': 0,
            'end_op_index': self.fwd_op_num
        }

    def get_param_grad_names(self):
        grad_names = []
        for var_name in self.inputs['Params']:
            grad_names.append(var_name + core.grad_var_suffix())
        return grad_names

    def check_output_with_place(self, place):
        # Step 1. run op
        actual_outs = self.calc_dygraph_output(place)

        # Step 2. compare output
        for expect_v, actual_v in six.moves.zip(self.expect_outs, actual_outs):
            self.assertTrue(np.allclose(expect_v, actual_v.numpy(), atol=1e-5))

    def check_grad_with_place(self, place):
        # Step 1. calc grads
        actual_grads = self.calc_dygraph_grad(place)

        # Step 2. compare grads
        for expect_v, actual_v in six.moves.zip(self.expect_grads,
                                                actual_grads):
            np.testing.assert_array_almost_equal(expect_v, actual_v)
            self.assertTrue(np.allclose(expect_v, actual_v, atol=1e-5))

    def prepare_dygraph_input(self, place, return_param_list=False):
        def create_var_base(is_input, name, np_value, stop_gradient):
            var = core.VarBase(
                value=np_value, name=name, place=place, zero_copy=True)
            var.stop_gradient = stop_gradient
            return var

        # build inputs
        inputs = {}
        param_list = []
        inputs['X'] = []
        for name, np_value in self.inputs['X'].items():
            var = create_var_base(True, name, np_value, True)
            inputs['X'].append(var)
        inputs['Params'] = []
        for name, np_value in self.inputs['Params'].items():
            var = create_var_base(True, name, np_value, False)
            inputs['Params'].append(var)
            if return_param_list:
                param_list.append(var)

        if return_param_list:
            return inputs, param_list
        return inputs

    def prepare_dygraph_output(self):
        def create_var_base(is_input, name):
            var = framework._varbase_creator(dtype=None, shape=None, name=name)
            var.stop_gradient = False
            return var

        # build outputs
        outputs = {}
        outputs['Out'] = []
        for name in self.output_names['Out']:
            outputs['Out'].append(create_var_base(False, name))

        outputs['OutScope'] = framework._varbase_creator(
            type=core.VarDesc.VarType.STEP_SCOPES,
            name="program_out_scope",
            persistable=True)
        inner_scope = core.Scope()
        outputs['OutScope'].value().set_scope(inner_scope)

        outputs['DOut'] = [create_var_base(False, "Fake_var")]
        return outputs

    def calc_dygraph_output(self, place):
        self.program_desc, self.fwd_op_num = self.get_program_desc()
        self.attrs = self.prepare_attrs()

        with fluid.dygraph.guard(place):
            inputs = self.prepare_dygraph_input(place)
            outputs = self.prepare_dygraph_output()

            framework._dygraph_tracer().trace_op(
                type=self.op_type,
                inputs=inputs,
                outputs=outputs,
                attrs=self.attrs)
            return outputs['Out']

    def calc_dygraph_grad(self, place):
        self.program_desc, self.fwd_op_num = self.get_program_desc()
        self.attrs = self.prepare_attrs()

        with fluid.dygraph.guard(place):
            # Step 1. run forward
            inputs, input_param_list = self.prepare_dygraph_input(place, True)
            outputs = self.prepare_dygraph_output()

            framework._dygraph_tracer().trace_op(
                type=self.op_type,
                inputs=inputs,
                outputs=outputs,
                attrs=self.attrs)

            for param in input_param_list:
                var_type = self._get_grad_vartype(param.name)
                if var_type is None:
                    continue
                param._set_grad_type(var_type)

            # Step 2. run backward
            # NOTE: in unittest, only support single output now
            actual_outs = outputs['Out']
            assert len(actual_outs) == 1
            actual_outs[0].backward()

            # Step 3. prepare grads
            grads = []
            for param in input_param_list:
                grad = param.gradient()
                grads.append(grad)
            return grads

    def _get_grad_vartype(self, name):
        assert self.program_desc is not None
        grad_name = name + core.grad_var_suffix()
        for i in six.moves.range(self.program_desc.num_blocks()):
            block = self.program_desc.block(i)
            var_desc = block.find_var_recursive(cpt.to_bytes(grad_name))
            return var_desc.type() if var_desc is not None else None


class TestRunProgramOpWithFC(RunProgramOpTest):
    def setUp(self):
        self.op_type = "run_program"
        self.dtype = np.float32
        self.input_names = {
            'X': ['img'],
            'Params': ['weight_param', 'bias_param']
        }
        self.output_names = {'Out': ['fc_0.tmp_2']}

        self.inputs = {
            'X': {
                self.input_names['X'][0]: np.random.random((32, 1, 28, 28))
                .astype(self.dtype)
            },
            'Params': {
                self.input_names['Params'][0]: np.random.random(
                    (784, 10)).astype(self.dtype),
                self.input_names['Params'][1]: np.random.random(
                    (32, 10)).astype(self.dtype)
            }
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad()

    def build_model(self):
        # 1. simple model
        img = fluid.data(
            name=self.input_names['X'][0],
            shape=[None, 1, 28, 28],
            dtype='float32')
        weight_attr = fluid.ParamAttr(
            name=self.input_names['Params'][0],
            learning_rate=0.5,
            initializer=fluid.initializer.NumpyArrayInitializer(self.inputs[
                'Params'][self.input_names['Params'][0]]),
            trainable=True)
        bias_attr = fluid.ParamAttr(
            name=self.input_names['Params'][1],
            learning_rate=0.5,
            initializer=fluid.initializer.NumpyArrayInitializer(self.inputs[
                'Params'][self.input_names['Params'][1]]),
            trainable=True)
        pred = fluid.layers.fc(input=img,
                               size=10,
                               param_attr=weight_attr,
                               bias_attr=bias_attr,
                               act='relu')
        # 2. get forward op num
        fwd_op_num = fluid.default_main_program().global_block().desc.op_size()
        # 3. append backward
        grads = fluid.backward.gradients(targets=[pred], inputs=[img])

        return fwd_op_num


class TestRunProgramOpWithEmbedding(RunProgramOpTest):
    def setUp(self):
        self.op_type = "run_program"
        self.dtype = np.float32
        self.input_names = {'X': ['x'], 'Params': ['emb_weight']}
        self.output_names = {'Out': ['reduce_sum_0.tmp_0']}

        self.inputs = {
            'X': {
                'x': np.array([[1, 3, 0, 4, 7]]).astype("int64")
            },
            'Params': {
                'emb_weight': np.random.random(size=(10, 16)).astype("float32")
            }
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        # NOTE: fecth not support SelectedRows, catnot compare 
        # sparse gradients with staic mode, only run dygraph
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            # TODO: RunProgramOp is not recommended for use in static mode now
            self.calc_dygraph_grad(place)

    def build_model(self):
        # 1. simple model
        x = fluid.layers.data(
            name=self.input_names['X'][0], shape=[5], dtype='int64')
        emb = fluid.input.embedding(
            input=x,
            size=[10, 16],
            param_attr=fluid.ParamAttr(
                name="emb_weight",
                learning_rate=10,
                initializer=fluid.initializer.NumpyArrayInitializer(self.inputs[
                    'Params'][self.input_names['Params'][0]])),
            is_sparse=True)
        y = fluid.layers.reduce_sum(emb, dim=-1)
        # 2. get forward op num
        fwd_op_num = fluid.default_main_program().global_block().desc.op_size()
        # 3. append backward
        grads = fluid.backward.gradients(targets=[y], inputs=[x])

        return fwd_op_num


if __name__ == "__main__":
    unittest.main()
