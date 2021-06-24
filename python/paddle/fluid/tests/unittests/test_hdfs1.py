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

from paddle.fluid.tests.unittests.hdfs_test_utils import FSTestBase
import unittest
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.collective import CollectiveOptimizer, fleet
import os
import sys

from paddle.distributed.fleet.utils.fs import LocalFS, HDFSClient, FSTimeOut, FSFileExistsError, FSFileNotExistsError

java_home = os.environ["JAVA_HOME"]


class FSTest1(FSTestBase):
    def test_timeout(self):
        fs = HDFSClient(
            "/usr/local/hadoop-2.7.7/",
            None,
            time_out=6 * 1000,
            sleep_inter=100)
        src = "hdfs_test_timeout"
        dst = "new_hdfs_test_timeout"
        fs.delete(dst)
        fs.mkdirs(src)
        fs.mkdirs(dst)
        fs.mkdirs(dst + "/" + src)
        output = ""
        cmd = "{} -mv {} {}".format(fs._base_cmd, src, dst)
        try:
            fs.mv(src, dst, test_exists=False)
            self.assertFalse(1, "can't execute cmd:{} output:{}".format(cmd,
                                                                        output))
        except FSTimeOut as e:
            print("execute mv {} to {} timeout".format(src, dst))

        ret, output = fluid.core.shell_execute_cmd(cmd, 6 * 1000, 2 * 1000)
        self.assertNotEqual(ret, 0)
        print("second mv ret:{} output:{}".format(ret, output))

    def test_is_dir(self):
        fs = HDFSClient(
            "/usr/local/hadoop-2.7.7/",
            None,
            time_out=6 * 1000,
            sleep_inter=100)
        self.assertFalse(fs.is_dir("./test_hdfs.py"))
        s = """
java.io.IOException: Input/output error
 responseErrorMsg : failed to getFileStatus, errorCode: 3, path: /user/PUBLIC_KM_Data/wangxi16/data/serving_model, lparam: d868f6bb6822c621, errorMessage: inner error
	at org.apache.hadoop.util.FileSystemUtil.throwException(FileSystemUtil.java:164)
	at org.apache.hadoop.util.FileSystemUtil.dealWithResponse(FileSystemUtil.java:118)
	at org.apache.hadoop.lite.client.LiteClientImpl.getFileStatus(LiteClientImpl.java:696)
	at org.apache.hadoop.fs.LibDFileSystemImpl.getFileStatus(LibDFileSystemImpl.java:297)
	at org.apache.hadoop.fs.LiteFileSystem.getFileStatus(LiteFileSystem.java:514)
	at org.apache.hadoop.fs.FsShell.test(FsShell.java:1092)
	at org.apache.hadoop.fs.FsShell.run(FsShell.java:2285)
	at org.apache.hadoop.util.ToolRunner.run(ToolRunner.java:65)
	at org.apache.hadoop.util.ToolRunner.run(ToolRunner.java:79)
	at org.apache.hadoop.fs.FsShell.main(FsShell.java:2353)
        """

        print("split lines:", s.splitlines())
        self.assertTrue(fs._test_match(s.splitlines()) != None)

    def test_config(self):
        config = {"fs.default.name": "hdfs://xxx", "hadoop.job.ugi": "ugi"}
        fs = HDFSClient(
            "/usr/local/hadoop-2.7.7/",
            config,
            time_out=6 * 1000,
            sleep_inter=100)

    def test_exists(self):
        fs = HDFSClient(
            "/usr/local/hadoop-2.7.7/",
            None,
            time_out=6 * 1000,
            sleep_inter=100)
        self.assertFalse(fs.is_exist(os.path.abspath("./xxxx")))
        self.assertFalse(fs.is_dir(os.path.abspath("./xxxx")))
        self.assertTrue(fs.is_dir(os.path.abspath("./xxx/..")))
        dirs, files = fs.ls_dir(os.path.abspath("./test_hdfs1.py"))
        self.assertTrue(dirs == [])
        self.assertTrue(len(files) == 1)
        dirs, files = fs.ls_dir(os.path.abspath("./xxx/.."))


if __name__ == '__main__':
    unittest.main()
