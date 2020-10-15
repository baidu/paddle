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

from paddle.distributed.fleet.utils.fs import LocalFS, HDFSClient, FSTimeOut, FSFileExistsError, FSFileNotExistsError

java_home = os.environ["JAVA_HOME"]


class FSTestBase(unittest.TestCase):
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

        self.assertFalse(fs.is_exist(dst_file))
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
