# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import os
import sys
import paddle.fluid as fluid
import importlib
from six.moves import cStringIO


def main():
    sys.path.append(os.getcwd())
    some_test_failed = False
    for module_name in sys.argv[1:]:
        buffer = cStringIO()
        main = fluid.Program()
        startup = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.program_guard(main, startup):
            with fluid.scope_guard(scope):
                with fluid.unique_name.guard():
                    test_loader = unittest.TestLoader()
                    module = importlib.import_module(module_name)
                    tests = test_loader.loadTestsFromModule(module)
                    res = unittest.TextTestRunner(stream=buffer).run(tests)
                    if not res.wasSuccessful():
                        some_test_failed = True
                        print(
                            module_name,
                            'failed\n',
                            buffer.getvalue(),
                            file=sys.stderr)

    if some_test_failed:
        exit(1)


if __name__ == '__main__':
    main()
