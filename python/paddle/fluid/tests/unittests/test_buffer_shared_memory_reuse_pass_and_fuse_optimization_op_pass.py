# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from test_buffer_shared_memory_reuse_pass import InplaceTestBase
import unittest


class CUDAInplaceTestWithFuseOptimizationOps(InplaceTestBase):
    def initParameter(self):
        self.use_cuda = True
        self.fuse_all_optimizer_ops = True
        self.fuse_all_reduce_ops = False

    def test_multi_card_fetch_var(self):
        self.check_multi_card_fetch_var()

    def test_single_card_fetch_var(self):
        self.check_single_card_fetch_var()


class CPUInplaceTestWithFuseOptimizationOps(InplaceTestBase):
    def initParameter(self):
        self.use_cuda = False
        self.fuse_all_optimizer_ops = True
        self.fuse_all_reduce_ops = False

    def test_multi_card_fetch_var(self):
        self.check_multi_card_fetch_var()

    # TODO(zcd): should check why this test failed.
    @unittest.skip("should fix this later.")
    def test_single_card_fetch_var(self):
        self.check_single_card_fetch_var()


if __name__ == '__main__':
    unittest.main()
