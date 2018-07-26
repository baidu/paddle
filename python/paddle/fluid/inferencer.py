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

import contextlib

from . import core

from . import executor
from . import framework
from . import io
from . import parallel_executor
from . import unique_name
from .trainer import check_and_get_place

__all__ = ['Inferencer', ]


class Inferencer(object):
    """
    Inferencer High Level API.

    Args:
        infer_func (Python func): Infer function that will return predict Variable
        param_path (str): The path where the inference model is saved by fluid.io.save_params
        place (Place): place to do the inference
        parallel (bool): use parallel_executor to run the inference, it will use multi CPU/GPU.

    Examples:
        .. code-block:: python

            def inference_program():
                x = fluid.layers.data(name='x', shape=[13], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                return y_predict

            place = fluid.CPUPlace()
            inferencer = fluid.Inferencer(
                infer_func=inference_program, param_path="/tmp/model", place=place)

    """

    def __init__(self, infer_func, param_path, place=None, parallel=False):
        self.param_path = param_path
        self.scope = core.Scope()
        self.parallel = parallel
        self.place = check_and_get_place(place)

        self.inference_program = framework.Program()
        with framework.program_guard(self.inference_program):
            with unique_name.guard():
                self.predict_var = infer_func()

        with self._prog_and_scope_guard():
            # load params from param_path into scope
            io.load_params(executor.Executor(self.place), param_path)

        if parallel:
            with self._prog_and_scope_guard():
                self.exe = parallel_executor.ParallelExecutor(
                    use_cuda=isinstance(self.place, core.CUDAPlace),
                    loss_name=self.predict_var.name)
        else:
            self.exe = executor.Executor(self.place)

        self.inference_program = self.inference_program.clone(for_test=True)

    def infer(self, inputs, return_numpy=True):
        """
        Do Inference for Inputs

        Args:
            inputs (map): a map of {"input_name": input_var} that will be feed into the inference program
            return_numpy (bool): transform return value into numpy or not

        Returns:
            Tensor or Numpy: the predict value of the inference model for the inputs

        Examples:
            .. code-block:: python

                tensor_x = numpy.random.uniform(0, 10, [batch_size, 13]).astype("float32")
                results = inferencer.infer({'x': tensor_x})
        """
        if not isinstance(inputs, dict):
            raise ValueError(
                "inputs should be a map of {'input_name': input_var}")

        with executor.scope_guard(self.scope):
            results = self.exe.run(self.inference_program,
                                   feed=inputs,
                                   fetch_list=[self.predict_var],
                                   return_numpy=return_numpy)

        return results

    @contextlib.contextmanager
    def _prog_and_scope_guard(self):
        with framework.program_guard(main_program=self.inference_program):
            with executor.scope_guard(self.scope):
                yield
