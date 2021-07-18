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
import os
import sys

from paddle.fluid.incubate.fleet.utils.fs import LocalFS, HDFSClient, FSTimeOut, FSFileExistsError, FSFileNotExistsError

java_home = os.environ["JAVA_HOME"]


class FSTest(unittest.TestCase):
    def _test_dirs(self, fs):
        dir_path = os.path.abspath("./test_dir")
        fs.delete(dir_path)
        self.assertTrue(not fs.is_exist(dir_path))

        fs.mkdirs(dir_path)
        self.assertTrue(fs.is_exist(dir_path))
        self.assertTrue(not fs.is_file(dir_path))
        self.assertTrue(fs.is_dir(dir_path))

        new_dir_path = os.path.abspath("./new_test_dir")
        fs.delete(new_dir_path)
        try:
            fs.mv(new_dir_path, dir_path)
            self.assertFalse(True)
        except FSFileNotExistsError as e:
            pass

        fs.mv(dir_path, new_dir_path)
        self.assertTrue(fs.is_exist(new_dir_path))

        fs.mv(new_dir_path, dir_path)
        self.assertTrue(fs.is_exist(dir_path))
        try:
            fs.mv(dir_path, dir_path)
            self.assertFalse(True)
        except FSFileExistsError as e:
            pass

        fs.delete(dir_path)
        self.assertTrue(not fs.is_exist(dir_path))

        fs.mkdirs(dir_path)
        fs.mkdirs(new_dir_path)
        fs.mv(dir_path, new_dir_path, overwrite=True)
        self.assertTrue(not fs.is_exist(dir_path))
        self.assertTrue(fs.is_exist(new_dir_path))

    def _test_touch_file(self, fs):
        file_path = os.path.abspath("./test_file")

        fs.delete(file_path)
        self.assertTrue(not fs.is_exist(file_path))

        fs.touch(file_path)
        self.assertTrue(fs.is_exist(file_path))
        self.assertTrue(not fs.is_dir(file_path) and fs.is_file(file_path))

        new_file_path = os.path.abspath("./new_test_file")
        fs.mv(file_path, new_file_path)
        self.assertTrue(fs.is_exist(new_file_path))

        fs.mv(new_file_path, file_path)
        self.assertTrue(fs.is_exist(file_path))

        fs.delete(file_path)
        self.assertTrue(not fs.is_exist(file_path))

    def _test_upload(self, fs):
        src_file = os.path.abspath("./test_upload.src")
        dst_file = os.path.abspath("./test_uolpad.dst")

        try:
            fs.upload(src_file, dst_file)
            self.assertFalse(True)
        except FSFileNotExistsError as e:
            pass

        local = LocalFS()
        local.touch(src_file)
        fs.delete(dst_file)

        assert fs.need_upload_download()

        fs.upload(src_file, dst_file)
        try:
            fs.upload(src_file, dst_file)
            self.assertFalse(True)
        except FSFileExistsError as e:
            pass

        self.assertTrue(fs.is_exist(dst_file))
        fs.delete(dst_file)
        fs.delete(src_file)

    def _test_try_download(self, fs):
        src_file = os.path.abspath("./test_try_download.src")
        dst_file = os.path.abspath("./test_try_download.dst")

        fs.delete(dst_file)
        fs.delete(src_file)

        try:
            fs._try_download(src_file, dst_file)
            self.assertFalse(True)
        except Exception as e:
            pass

        fs.delete(dst_file)
        fs.delete(src_file)

    def _test_try_upload(self, fs):
        src_file = os.path.abspath("./test_try_upload.src")
        dst_file = os.path.abspath("./test_try_uolpad.dst")

        try:
            fs._try_upload(src_file, dst_file)
            self.assertFalse(True)
        except Exception as e:
            pass

        fs.delete(dst_file)
        fs.delete(src_file)

    def _test_download(self, fs):
        src_file = os.path.abspath("./test_download.src")
        dst_file = os.path.abspath("./test_download.dst")
        fs.delete(dst_file)
        fs.delete(src_file)

        try:
            fs.download(src_file, dst_file)
            self.assertFalse(True)
        except FSFileNotExistsError as e:
            pass

        local = LocalFS()
        local.touch(src_file)
        fs.delete(dst_file)

        assert fs.need_upload_download()

        fs.download(src_file, dst_file)
        try:
            fs.download(src_file, dst_file)
            self.assertFalse(True)
        except FSFileExistsError as e:
            pass

        self.assertTrue(fs.is_exist(dst_file))
        fs.delete(dst_file)
        fs.delete(src_file)

    def _test_mkdirs(self, fs):
        dir_name = "./test_mkdir"
        fs.mkdirs(dir_name)
        fs.mkdirs(dir_name)

    def _test_rm(self, fs):
        dir_name = "./test_rm_no_exist.flag"
        fs.delete(dir_name)
        try:
            fs._rmr(dir_name)
            self.assertFalse(True)
        except Exception as e:
            pass

        try:
            fs._rm(dir_name)
            self.assertFalse(True)
        except Exception as e:
            pass

    def test_exists(self):
        fs = HDFSClient(
            "/usr/local/hadoop-2.7.7/",
            None,
            time_out=15 * 1000,
            sleep_inter=100)
        self.assertFalse(fs.is_exist(os.path.abspath("./xxxx")))
        self.assertFalse(fs.is_dir(os.path.abspath("./xxxx")))
        self.assertTrue(fs.is_dir(os.path.abspath("./xxx/..")))
        dirs, files = fs.ls_dir(os.path.abspath("./test_hdfs.py"))
        self.assertTrue(dirs == [])
        self.assertTrue(len(files) == 1)
        dirs, files = fs.ls_dir(os.path.abspath("./xxx/.."))

    def test_hdfs(self):
        fs = HDFSClient(
            "/usr/local/hadoop-2.7.7/",
            None,
            time_out=15 * 1000,
            sleep_inter=100)
        self._test_rm(fs)
        self._test_touch(fs)
        self._test_dirs(fs)
        self._test_upload(fs)

        self._test_download(fs)
        self._test_mkdirs(fs)
        self._test_list_dir(fs)
        self._test_try_upload(fs)
        self._test_try_download(fs)

    def test_local(self):
        fs = LocalFS()
        self._test_rm(fs)
        self._test_touch(fs)
        self._test_dirs(fs)
        self._test_touch_file(fs)
        self._test_mkdirs(fs)
        self._test_list_dir(fs)
        self._test_try_upload(fs)
        self._test_try_download(fs)

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
        try:
            fs.mv(src, dst, test_exists=False)
            self.assertFalse(1, "can't execute cmd:{} output:{}".format(cmd,
                                                                        output))
        except FSTimeOut as e:
            print("execute mv {} to {} timeout".format(src, dst))

        cmd = "{} -mv {} {}".format(fs._base_cmd, src, dst)
        ret, output = fluid.core.shell_execute_cmd(cmd, 6 * 1000, 2 * 1000)
        self.assertNotEqual(ret, 0)
        print("second mv ret:{} output:{}".format(ret, output))

    def test_is_dir(self):
        fs = HDFSClient(
            "/usr/local/hadoop-2.7.7/",
            None,
            time_out=15 * 1000,
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
            time_out=15 * 1000,
            sleep_inter=100)

    def _test_list_dir(self, fs):
        fs = HDFSClient(
            "/usr/local/hadoop-2.7.7/",
            None,
            time_out=15 * 1000,
            sleep_inter=100)
        fs.ls_dir("test_not_exists")

    def _test_touch(self, fs):
        path = "./touch.flag"
        fs.touch(path, exist_ok=True)
        try:
            fs.touch("./touch.flag", exist_ok=False)
            self.assertFalse(0, "can't reach here")
        except FSFileExistsError as e:
            pass

        try:
            fs._touchz("./touch.flag")
            self.assertFalse(True, "can't reach here")
        except Exception as e:
            pass

        self.assertFalse(fs.is_dir(path))
        fs.delete(path)


if __name__ == '__main__':
    unittest.main()
