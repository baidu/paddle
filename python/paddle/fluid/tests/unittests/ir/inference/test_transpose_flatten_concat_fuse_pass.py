# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker


class TransposeFlattenConcatFusePassTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data1 = fluid.data(name="data1", shape=[5, 5, 5], dtype="float32")
            data2 = fluid.data(name="data2", shape=[5, 5, 5], dtype="float32")
            trans1 = fluid.layers.transpose(data1, perm=[2, 1, 0])
            trans2 = fluid.layers.transpose(data2, perm=[2, 1, 0])
            flatt1 = fluid.layers.flatten(trans1)
            flatt2 = fluid.layers.flatten(trans2)
            concat_out = fluid.layers.concat([flatt1, flatt2])
            # There is no parameters for above structure. 
            # Hence, append a batch_norm to avoid failure caused by load_combined. 
            out = fluid.layers.batch_norm(concat_out, is_test=True)

        self.feeds = {
            "data1": np.random.random([5, 5, 5]).astype("float32"),
            "data2": np.random.random([5, 5, 5]).astype("float32")
        }
        self.fetch_list = [out]

    def test_check_output(self):
        # There is no cpu pass for transpose_flatten_concat_fuse
        if core.is_compiled_with_cuda():
            use_gpu = False if core.is_compiled_with_rocm() else True
            self.check_output_with_option(use_gpu)

        PassVersionChecker.IsCompatible('transpose_flatten_concat_fuse_pass')


class TransposeFlattenConcatFusePassWithAxisTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data1 = fluid.data(name="data1", shape=[5, 5, 5], dtype="float32")
            data2 = fluid.data(name="data2", shape=[5, 5, 5], dtype="float32")
            trans1 = fluid.layers.transpose(data1, perm=[2, 1, 0])
            trans2 = fluid.layers.transpose(data2, perm=[2, 1, 0])
            flatt1 = fluid.layers.flatten(trans1, axis=2)
            flatt2 = fluid.layers.flatten(trans2, axis=2)
            concat_out = fluid.layers.concat([flatt1, flatt2], axis=1)
            # There is no parameters for above structure. 
            # Hence, append a batch_norm to avoid failure caused by load_combined. 
            out = fluid.layers.batch_norm(concat_out, is_test=True)

        self.feeds = {
            "data1": np.random.random([5, 5, 5]).astype("float32"),
            "data2": np.random.random([5, 5, 5]).astype("float32")
        }
        self.fetch_list = [out]

    def test_check_output(self):
        # There is no cpu pass for transpose_flatten_concat_fuse
        if core.is_compiled_with_cuda():
            use_gpu = False if core.is_compiled_with_rocm() else True
            self.check_output_with_option(use_gpu)

        self.assertTrue(
            PassVersionChecker.IsCompatible(
                'transpose_flatten_concat_fuse_pass'))


if __name__ == "__main__":
    unittest.main()
