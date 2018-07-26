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

import collections
import contextlib
import sys

__all__ = ['generate', 'switch', 'guard']


class UniqueNameGenerator(object):
    """
    Generate unique name with prefix.

    Args:
        prefix(str): The generated name prefix. All generated name will be
                     started with this prefix.
    """

    def __init__(self, prefix=None):
        self.ids = collections.defaultdict(int)
        if prefix is None:
            prefix = ""
        self.prefix = prefix

    def __call__(self, key):
        """
        Generate unique names with prefix

        Args:
            key(str): The key of return string.

        Returns(str): A unique string with the prefix
        """
        tmp = self.ids[key]
        self.ids[key] += 1
        return self.prefix + "_".join([key, str(tmp)])


generator = UniqueNameGenerator()


def generate(key):
    return generator(key)


def switch(new_generator=None):
    global generator
    old = generator
    if new_generator is None:
        generator = UniqueNameGenerator()
    else:
        generator = new_generator
    return old


@contextlib.contextmanager
def guard(new_generator=None):
    if isinstance(new_generator, str):
        new_generator = UniqueNameGenerator(new_generator)
    old = switch(new_generator)
    yield
    switch(old)
