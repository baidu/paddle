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
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.collective import CollectiveOptimizer, fleet
from paddle.fluid.incubate.checkpoint.auto_checkpoint import ExeTrainStatus
from paddle.fluid.incubate.checkpoint.checkpoint_saver import CheckpointSaver
import os
import sys

from paddle.fluid.incubate.fleet.utils.fs import LocalFS, HDFSClient
from paddle.fluid.incubate.checkpoint.checkpoint_saver import CheckpointSaver


class CheckpointerSaverTest(unittest.TestCase):
    def test(self):
        fs = HDFSClient("/usr/local/hadoop-2.7.7", None)
        dir_path = "./checkpointsaver_test"
        fs.delete(dir_path)

        s = CheckpointSaver(fs)

        fs.mkdirs("{}/exe.exe".format(dir_path))
        fs.mkdirs("{}/exe.1".format(dir_path))
        fs.mkdirs("{}/exe".format(dir_path))

        a = s.get_checkpoint_no(dir_path)
        self.assertEqual(len(a), 0)

        fs.mkdirs("{}/__paddle_checkpoint__.0".format(dir_path))
        fs.mkdirs("{}/__paddle_checkpoint__.exe".format(dir_path))

        a = s.get_checkpoint_no(dir_path)
        self.assertEqual(len(a), 1)

        s.clean_redundant_checkpoints(dir_path)
        s.clean_redundant_checkpoints(dir_path)

        fs.delete(dir_path)


if __name__ == '__main__':
    unittest.main()
