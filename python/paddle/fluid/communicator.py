#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from .executor import global_scope
"""
Communicator is used for async distribute training in distribute_transpiler mode.
It's a wrapper of a cpp class Communicator and should be used inside fleet API.
"""
from . import core
from .framework import Program

__all__ = ['Communicator', 'AsyncMode']


class AsyncMode:
    ASYNC = 1
    HALF_ASYNC = 2
    GEO_SGD = 3


class Communicator(object):
    def __init__(self, program, mode, **kwargs):
        """
        Communicator is used for async distribute training in distribute_transpiler mode.
        It's a wrapper of a cpp class Communicator and should be used inside fleet API.

        Args:
            program(Program): the trainers program after transpile of distribute_transpiler.
            It's used by communicator to extract the information to do communication.

        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                prog = fluid.Program()
                comm = fluid.communicator.Communicator(prog)
                comm.start()
                comm.stop()
        """
        # set all recv op to not_run mode
        assert isinstance(program, Program)
        for op in program.block(0).ops:
            if op.type == "recv":
                op._set_attr('do_not_run', True)

        if mode == AsyncMode.ASYNC:
            self.communicator_ = core.DistCommunicator(program.desc,
                                                       global_scope(), False)
        elif mode == AsyncMode.HALF_ASYNC:
            self.communicator_ = core.DistCommunicator(program.desc,
                                                       global_scope(), True)
        elif mode == AsyncMode.GEO_SGD:
            push_vars = kwargs["push_vars"]
            trainers = int(kwargs["trainers"])
            push_nums = int(kwargs["push_nums"])

            if trainers <= 0:
                raise ValueError("trainers must gather than 0")
            if push_nums <= 0:
                raise ValueError("geo push delta must gather than 0")

            self.communicator_ = core.DistCommunicator(
                program.desc, global_scope(), push_vars, trainers, push_nums)
        else:
            raise ValueError("unknown MODE for communicator")

    def start(self):
        """
        Start communicator. Should call before training process.

        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                prog = fluid.Program()
                comm = fluid.communicator.Communicator(prog)
                comm.start()
                comm.stop()
        """
        self.communicator_.start()

    def stop(self):
        """
        Stop communicator. Should call after training process.

        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                prog = fluid.Program()
                comm = fluid.communicator.Communicator(prog)
                comm.start()
                comm.stop()
        """
        self.communicator_.stop()

    def is_running(self):
        """
        Get communicator is running or stop.

        Returns:
            bool

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                prog = fluid.Program()
                comm = fluid.communicator.Communicator(prog)
                comm.is_running()
        """
        self.communicator_.is_running()
