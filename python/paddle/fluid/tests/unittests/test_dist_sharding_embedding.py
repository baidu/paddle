# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
import tempfile
import shutil
from paddle.distributed.fleet.utils.ps_util import sparse_sharding_merge


class ShardingEmbeddingMerge(unittest.TestCase):
    def _write_error_meta2(self, metafile):
        meta = """param=emb_0
shard_id=0
row_names=Moment1,Moment2,Beta1Pow,Beta2Pow,LearningRate
row_dims=3,3,1,1,1
count=4
"""
        with open(metafile, 'w') as wb:
            wb.write(meta)

    def _write_error_meta(self, metafile):
        meta = """param=error_emb
shard_id=0
row_names=Param,Moment1,Moment2,Beta1Pow,Beta2Pow,LearningRate
row_dims=3,3,3,1,1,1
count=4
"""
        with open(metafile, 'w') as wb:
            wb.write(meta)

    def _write_meta(self, metafile):
        meta = """param=emb_0
shard_id=0
row_names=Param,Moment1,Moment2,Beta1Pow,Beta2Pow,LearningRate
row_dims=3,3,3,1,1,1
count=4
"""
        with open(metafile, 'w') as wb:
            wb.write(meta)

    def _write_embedding(self, embfile):
        emb = """332752	40	0	1	0.000024,0.000002,0.000007,0.000007,0.000002,0.000003,0.000006,0.000005,0.000003,0.109419,0.979209,0.000100
572112	80	0	1	0.000035,0.000048,0.000134,0.000004,0.000027,0.000028,0.000024,0.000054,0.000043,0.013303,0.959810,0.000100
994424	800	0	1	0.001493,0.001090,0.002490,0.002742,0.001905,0.001673,0.001102,0.002975,0.001674,0.001256,0.000000,0.676251
859850	40	0	1	0.000006,0.000023,0.000033,0.000006,0.000011,0.000004,0.000012,0.000006,0.000014,0.000009,0.109419,0.979209
"""
        with open(embfile, 'w') as wb:
            wb.write(emb)

    def setUp(self):
        self.dirname = tempfile.mkdtemp()
        self.shards = 2
        self.embedding = "emb_0"

    def test_error(self):
        with self.assertRaises(ValueError):
            sparse_sharding_merge(self.dirname, self.embedding)

        dirname = "{}/{}.shard".format(self.dirname, self.embedding)
        os.mkdir(dirname)

        emb = "{}/{}".format(self.dirname, self.embedding)
        os.mkdir(emb)

        with self.assertRaises(ValueError):
            sparse_sharding_merge(self.dirname, self.embedding)

        shutil.rmtree(emb)

        sparse_sharding_merge(self.dirname, self.embedding)
        os.remove(emb)

        for i in range(self.shards):
            emb = "{}/{}.block{}.txt".format(dirname, self.embedding, i)
            self._write_embedding(emb)

        for i in range(self.shards):
            meta = "{}/{}.block{}.meta".format(dirname, self.embedding, i)
            self._write_error_meta2(meta)

        with self.assertRaises(ValueError):
            sparse_sharding_merge(self.dirname, self.embedding)

        for i in range(self.shards):
            meta = "{}/{}.block{}.meta".format(dirname, self.embedding, i)
            os.remove(meta)
            self._write_error_meta(meta)

        with self.assertRaises(ValueError):
            sparse_sharding_merge(self.dirname, self.embedding)

        shutil.rmtree(dirname)

    def test_right(self):
        dirname = "{}/{}.shard".format(self.dirname, self.embedding)
        os.mkdir(dirname)

        for i in range(self.shards):
            meta = "{}/{}.block{}.meta".format(dirname, self.embedding, i)
            emb = "{}/{}.block{}.txt".format(dirname, self.embedding, i)
            self._write_meta(meta)
            self._write_embedding(emb)
        sparse_sharding_merge(self.dirname, self.embedding)

    def tearDown(self):
        shutil.rmtree(self.dirname)


if __name__ == "__main__":
    unittest.main()
