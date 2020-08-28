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
"""This is definition of generator class, which is for managing the state of the algorithm that produces pseudo random numbers."""

from . import core

__all__ = ['Generator']

default_rng_seed_val = 34342423252


class Generator(object):
    """Generator class"""

    def __init__(self, device="CPU"):
        """init"""
        self.device = device
        seed_in = default_rng_seed_val
        if self.device == "CPU":
            self.generator = core.Generator()
            # self.generator.manual_seed(seed_in)
        elif self.device == "CUDA":
            self.cuda_generator = core.default_cuda_generator(-1)
            self.cuda_generator.is_init_py_cuda = True
        else:
            raise ValueError(
                "generator class with device %s does not exist, currently only support generator with device 'CPU' "
                % device)

    def get_state(self):
        if self.device == "CUDA":
            return self.cuda_generator.get_state()
        else:
            return self.generator.get_state()

    def set_state(self, state):
        if self.device == "CUDA":
            self.cuda_generator.set_state(state)
        else:
            self.generator.set_state(state)

    def manual_seed(self, seed):
        if self.device == "CUDA":
            self.cuda_generator.manual_seed(seed)
        else:
            self.generator.manual_seed(seed)

    def seed(self):
        if self.device == "CUDA":
            return self.cuda_generator.seed()
        else:
            return self.generator.seed()

    def initial_seed(self):
        if self.device == "CUDA":
            return self.cuda_generator.initial_seed()
        else:
            return self.generator.initial_seed()

    def random(self):
        if self.device == "CUDA":
            return self.cuda_generator.random()
        else:
            return self.generator.random()

    def get_cpu_engine(self):
        if self.device == "CUDA":
            return self.cuda_generator.get_cpu_engine()
        else:
            return self.generator.get_cpu_engine()

    def set_cpu_engine(self, cpu_engine):
        if self.device == "CUDA":
            self.cuda_generator.set_cpu_engine(cpu_engine)
        else:
            self.generator.set_cpu_engine(cpu_engine)
