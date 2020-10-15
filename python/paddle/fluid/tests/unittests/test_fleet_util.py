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

from __future__ import print_function
import paddle
import paddle.fluid as fluid
import unittest
import numpy as np
import tarfile
import tempfile
import os
import sys
from paddle.dataset.common import download, DATA_HOME
import paddle.distributed.fleet.base.role_maker as role_maker


class TestFleetUtil(unittest.TestCase):
    proto_data_url = "https://fleet.bj.bcebos.com/fleet_util_data.tgz"
    proto_data_md5 = "59b7f12fd9dc24b64ae8e4629523a92a"
    module_name = "fleet_util_data"
    pruned_dir = os.path.join("fleet_util_data", "pruned_model")
    train_dir = os.path.join("fleet_util_data", "train_program")

    def test_util_base(self):
        import paddle.distributed.fleet as fleet
        util = fleet.UtilBase()
        strategy = fleet.DistributedStrategy()
        util._set_strategy(strategy)
        role_maker = None  # should be fleet.PaddleCloudRoleMaker()
        util._set_role_maker(role_maker)

    def test_util_factory(self):
        import paddle.distributed.fleet as fleet
        factory = fleet.base.util_factory.UtilFactory()
        strategy = fleet.DistributedStrategy()
        role_maker = None  # should be fleet.PaddleCloudRoleMaker()
        optimize_ops = []
        params_grads = []
        context = {}
        context["role_maker"] = role_maker
        context["valid_strategy"] = strategy
        util = factory._create_util(context)
        self.assertEqual(util.role_maker, None)

    def test_get_util(self):
        import paddle.distributed.fleet as fleet
        import paddle.distributed.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        self.assertNotEqual(fleet.util, None)

    def test_set_user_defined_util(self):
        import paddle.distributed.fleet as fleet

        class UserDefinedUtil(fleet.UtilBase):
            def __init__(self):
                super(UserDefinedUtil, self).__init__()

            def get_user_id(self):
                return 10

        import paddle.distributed.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        my_util = UserDefinedUtil()
        fleet.util = my_util
        user_id = fleet.util.get_user_id()
        self.assertEqual(user_id, 10)

    def test_fs(self):
        import paddle.distributed.fleet as fleet
        from paddle.distributed.fleet.utils import LocalFS

        fs = LocalFS()
        dirs, files = fs.ls_dir("test_tmp")
        dirs, files = fs.ls_dir("./")
        self.assertFalse(fs.need_upload_download())
        fleet.util._set_file_system(fs)

    def download_files(self):
        path = download(self.proto_data_url, self.module_name,
                        self.proto_data_md5)
        print('data is downloaded at ' + path)
        tar = tarfile.open(path)
        unzip_folder = tempfile.mkdtemp()
        tar.extractall(unzip_folder)
        return unzip_folder

    def test_get_file_shard(self):
        import paddle.distributed.fleet as fleet
        self.assertRaises(Exception, fleet.util.get_file_shard, "files")

        role = role_maker.UserDefinedRoleMaker(
            is_collective=False,
            init_gloo=False,
            current_id=0,
            role=role_maker.Role.WORKER,
            worker_endpoints=["127.0.0.1:6003", "127.0.0.1:6004"],
            server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"])
        fleet.init(role)

        files = fleet.util.get_file_shard(["1", "2", "3"])
        self.assertTrue(len(files) == 2 and "1" in files and "2" in files)

    def test_program_type_trans(self):
        import paddle.distributed.fleet as fleet
        data_dir = self.download_files()
        program_dir = os.path.join(data_dir, self.pruned_dir)
        text_program = "pruned_main_program.pbtxt"
        binary_program = "pruned_main_program.bin"
        text_to_binary = fleet.util._program_type_trans(program_dir,
                                                        text_program, True)
        binary_to_text = fleet.util._program_type_trans(program_dir,
                                                        binary_program, False)
        self.assertTrue(
            os.path.exists(os.path.join(program_dir, text_to_binary)))
        self.assertTrue(
            os.path.exists(os.path.join(program_dir, binary_to_text)))

    def test_prams_check(self):
        import paddle.distributed.fleet as fleet
        data_dir = self.download_files()

        class config:
            pass

        feed_config = config()
        feed_config.feeded_vars_names = ['concat_1.tmp_0', 'concat_2.tmp_0']
        feed_config.feeded_vars_dims = [682, 1199]
        feed_config.feeded_vars_types = [np.float32, np.float32]
        feed_config.feeded_vars_filelist = [
            os.path.join(data_dir, os.path.join(self.pruned_dir, "concat_1")),
            os.path.join(data_dir, os.path.join(self.pruned_dir, "concat_2"))
        ]

        fetch_config = config()
        fetch_config.fetch_vars_names = ['similarity_norm.tmp_0']

        conf = config()
        conf.batch_size = 1
        conf.feed_config = feed_config
        conf.fetch_config = fetch_config
        conf.dump_model_dir = os.path.join(data_dir, self.pruned_dir)
        conf.dump_program_filename = "pruned_main_program.pbtxt"
        conf.is_text_dump_program = True
        conf.save_params_filename = None

        # test saved var's shape
        conf.dump_program_filename = "pruned_main_program.save_var_shape_not_match"

        self.assertRaises(Exception, fleet.util._params_check)

        # test program.proto without feed_op and fetch_op
        conf.dump_program_filename = "pruned_main_program.no_feed_fetch"
        results = fleet.util._params_check(conf)
        self.assertTrue(len(results) == 1)
        np.testing.assert_array_almost_equal(
            results[0], np.array(
                [[3.0590223e-07]], dtype=np.float32))

        # test feed_var's shape
        conf.dump_program_filename = "pruned_main_program.feed_var_shape_not_match"
        self.assertRaises(Exception, fleet.util._params_check)

        # test correct case with feed_vars_filelist
        conf.dump_program_filename = "pruned_main_program.pbtxt"
        results = fleet.util._params_check(conf)
        self.assertTrue(len(results) == 1)
        np.testing.assert_array_almost_equal(
            results[0], np.array(
                [[3.0590223e-07]], dtype=np.float32))

        # test correct case without feed_vars_filelist
        conf.feed_config.feeded_vars_filelist = None
        # test feed var with lod_level >= 2
        conf.dump_program_filename = "pruned_main_program.feed_lod2"
        self.assertRaises(Exception, fleet.util._params_check)

        conf.dump_program_filename = "pruned_main_program.pbtxt"
        results = fleet.util._params_check(conf)
        self.assertTrue(len(results) == 1)

    def test_proto_check(self):
        import paddle.distributed.fleet as fleet
        data_dir = self.download_files()

        class config:
            pass

        conf = config()
        conf.train_prog_path = os.path.join(
            data_dir, os.path.join(self.train_dir, "join_main_program.pbtxt"))
        conf.is_text_train_program = True

        # test not match
        conf.pruned_prog_path = os.path.join(
            data_dir,
            os.path.join(self.pruned_dir,
                         "pruned_main_program.save_var_shape_not_match"))
        conf.is_text_pruned_program = True
        conf.draw = False
        res = fleet.util._proto_check(conf)
        self.assertFalse(res)

        # test match
        conf.pruned_prog_path = os.path.join(
            data_dir,
            os.path.join(self.pruned_dir, "pruned_main_program.pbtxt"))
        if sys.platform == 'win32' or sys.platform == 'sys.platform':
            conf.draw = False
        else:
            conf.draw = True
            conf.draw_out_name = "pruned_check"
        res = fleet.util._proto_check(conf)
        self.assertTrue(res)

    def test_visualize(self):
        import paddle.distributed.fleet as fleet
        if sys.platform == 'win32' or sys.platform == 'sys.platform':
            pass
        else:
            data_dir = self.download_files()
            program_path = os.path.join(
                data_dir,
                os.path.join(self.train_dir, "join_main_program.pbtxt"))
            is_text = True
            program = fleet.util._load_program(program_path, is_text)
            output_dir = os.path.join(data_dir, self.train_dir)
            output_filename = "draw_prog"
            fleet.util._visualize_graphviz(program, output_dir, output_filename)
            self.assertTrue(
                os.path.exists(
                    os.path.join(output_dir, output_filename + ".dot")))
            self.assertTrue(
                os.path.exists(
                    os.path.join(output_dir, output_filename + ".pdf")))


if __name__ == "__main__":
    unittest.main()
