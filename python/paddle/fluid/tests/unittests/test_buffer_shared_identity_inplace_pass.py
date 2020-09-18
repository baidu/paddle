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

import paddle.fluid as fluid

from test_buffer_shared_memory_reuse_pass import InplaceTestBase
import unittest
from simple_nets import simple_fc_net_with_inputs


def simple_identity_inplace_test_net():
    image = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    image *= 2
    outs = []
    for _ in range(3):
        tmp = fluid.layers.reshape(image, [-1, 28, 28, 1])
        tmp = fluid.layers.squeeze(tmp, axes=[3])
        tmp = fluid.layers.relu(tmp)
        outs.append(tmp)

    new_image = fluid.layers.sum(outs)
    return simple_fc_net_with_inputs(new_image, label)


class IdentityInplaceTestBase(InplaceTestBase):
    def net(self):
        return simple_identity_inplace_test_net


class CUDAInplaceTest(IdentityInplaceTestBase):
    def initParameter(self):
        self.use_cuda = True
        self.fuse_all_optimizer_ops = False

    def test_multi_card_fetch_var(self):
        self.check_multi_card_fetch_var()

    def test_single_card_fetch_var(self):
        self.check_single_card_fetch_var()


class CPUInplaceTest(IdentityInplaceTestBase):
    def initParameter(self):
        self.use_cuda = False
        self.fuse_all_optimizer_ops = False

    def test_multi_card_fetch_var(self):
        self.check_multi_card_fetch_var()

    def test_single_card_fetch_var(self):
        self.check_single_card_fetch_var()


if __name__ == '__main__':
    unittest.main()
