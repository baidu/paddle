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
import sys
import os
import time
import six
import copy
import json
import unittest
import paddle.fluid as fluid

import paddle.distributed.fleet.ascend_utils as ascend_utils

RANK_TABLE_JSON = {
    "status": "completed",
    "version": "1.0",
    "server_count": "1",
    "server_list": [{
        "server_id": "127.0.0.1",
        "device": [{
            "device_id": "0",
            "device_ip": "192.1.184.23",
            "rank_id": "0"
        }, {
            "device_id": "1",
            "device_ip": "192.2.21.93",
            "rank_id": "1"
        }]
    }]
}


class TestAscendUtil(unittest.TestCase):
    def test_get_cloud_cluster(self):
        cluster, pod = ascend_utils.get_cloud_cluster()
        self.assertTrue(cluster)
        self.assertTrue(pod)

        with open('rank_table_file.json', 'w') as f:
            json.dump(RANK_TABLE_JSON, f)
        rank_table_file = "./rank_table_file.json"
        cluster, pod = ascend_utils.get_cloud_cluster(
            rank_table_file=rank_table_file)
        self.assertTrue(cluster)
        self.assertTrue(pod)


if __name__ == '__main__':
    unittest.main()
