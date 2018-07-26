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
import unittest
import time
import itertools

import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from .op_test import OpTest


class BenchmarkSuite(OpTest):
    def timeit_function(self, callback, iters, *args, **kwargs):
        assert iters != 0, "Iters should >= 1"
        start = time.time()
        for i in range(iters):
            callback(*args, **kwargs)
        elapse = time.time() - start
        return elapse / iters

    def _assert_cpu_gpu_same(self, cpu_outs, gpu_outs, fetch_list, atol):
        for item_cpu_out, item_gpu_out, variable in zip(cpu_outs, gpu_outs,
                                                        fetch_list):
            # the cpu version is baseline, expect gpu version keep same with cpu version.
            expect = item_cpu_out
            expect_t = np.array(item_cpu_out)
            actual = item_gpu_out
            actual_t = np.array(item_gpu_out)
            var_name = variable if isinstance(variable, str) else variable.name
            self.assertTrue(
                np.allclose(
                    actual_t, expect_t, atol=atol),
                "Output (" + var_name + ") has diff" + str(actual_t) + "\n" +
                str(expect_t))
            self.assertListEqual(actual.lod(),
                                 expect.lod(),
                                 "Output (" + var_name + ") has different lod")

    def _get_input_names(self):
        inputs = []
        for name, value in list(self.inputs.items()):
            if isinstance(value, list):
                inputs.extend([sub_name for sub_name, _ in value])
            inputs.append(name)
        return inputs

    def _get_output_names(self):
        outputs = []
        for var_name, var in list(self.outputs.items()):
            if isinstance(var, list):
                for sub_var_name, sub_var in var:
                    outputs.append(sub_var_name)
            else:
                outputs.append(var_name)
        if len(outputs) == 0:
            for out_name, out_dup in Operator.get_op_outputs(self.op_type):
                outputs.append(str(out_name))
        return outputs

    def check_output_stability(self, atol=1e-8):
        places = self._get_places()
        if len(places) < 2:
            return
        cpu_outs, fetch_list = self._calc_output(places[0])
        gpu_outs, _ = self._calc_output(places[1])
        self._assert_cpu_gpu_same(cpu_outs, gpu_outs, fetch_list, atol)

    def timeit_output_with_place(self, place, iters):
        return self.timeit_function(self.calc_output, iters, place)

    def timeit_output(self, iters=100):
        places = self._get_places()
        elapses = []
        for place in places:
            elapses.append(self.timeit_output_with_place(place, iters))
        for place, elapse in zip(places, elapses):
            print(("One pass of ({2}_op) at {0} cost {1}".format(
                str(place), elapse, self.op_type)))

    def timeit_grad_with_place(self, place, iters=100):
        inputs_to_check = self._get_input_names()
        output_names = self._get_output_names()
        return self.timeit_function(
            self._get_gradient,
            iters,
            inputs_to_check,
            place,
            output_names,
            no_grad_set=None)

    def timeit_grad(self, iters=100):
        places = self._get_places()
        elapses = []
        for place in places:
            elapses.append(self.timeit_grad_with_place(place, iters))
        for place, elapse in zip(places, elapses):
            print(("One pass of ({2}_grad_op) at {0} cost {1}".format(
                str(place), elapse, self.op_type)))
