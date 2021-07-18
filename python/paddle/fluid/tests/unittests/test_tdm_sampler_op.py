# -*-coding:utf-8-*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid.layers as layers
import paddle.fluid as fluid
import random
import six
from sys import version_info


def create_tdm_travel():
    tree_travel = [[1, 3, 7, 14], [1, 3, 7, 15], [1, 3, 8, 16], [1, 3, 8, 17],
                   [1, 4, 9, 18], [1, 4, 9, 19], [1, 4, 10, 20],
                   [1, 4, 10, 21], [2, 5, 11, 22], [2, 5, 11, 23],
                   [2, 5, 12, 24], [2, 5, 12, 25], [2, 6, 13, 0]]
    return tree_travel


def create_tdm_layer():
    tree_layer = [[1, 2], [3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13],
                  [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]]
    return tree_layer


type_dict = {
    "int32": int(core.VarDesc.VarType.INT32),
    "int64": int(core.VarDesc.VarType.INT64)
}


class TestTDMSamplerOp(OpTest):
    def setUp(self):
        self.__class__.op_type = "tdm_sampler"
        self.config()

        self.tree_travel = create_tdm_travel()
        self.tree_layer = create_tdm_layer()

        output_0 = self.x_shape[0]
        output_1 = len(self.neg_samples_num_list) + \
            np.sum(self.neg_samples_num_list)
        self.output_shape = (output_0, output_1)
        self.layer_sample_nums = [1 + i for i in self.neg_samples_num_list]

        layer_node_num_list = [len(i) for i in self.tree_layer]
        tree_layer_offset_lod = [0]
        tree_layer_flat = []
        node_nums = 0
        for layer_idx, layer_node in enumerate(layer_node_num_list):
            tree_layer_flat += self.tree_layer[layer_idx]
            node_nums += layer_node
            tree_layer_offset_lod.append(node_nums)

        travel_np = np.array(self.tree_travel).astype(self.tree_dtype)
        layer_np = np.array(tree_layer_flat).astype(self.tree_dtype)
        layer_np = layer_np.reshape([-1, 1])

        self.x_np = np.random.randint(
            low=0, high=13, size=self.x_shape).astype(self.x_type)

        out = np.random.random(self.output_shape).astype(self.out_dtype)
        label = np.random.random(self.output_shape).astype(self.out_dtype)
        mask = np.random.random(self.output_shape).astype(self.out_dtype)

        self.attrs = {
            'neg_samples_num_list': self.neg_samples_num_list,
            'output_positive': True,
            'layer_offset_lod': tree_layer_offset_lod,
            'seed': 0,
            'dtype': type_dict[self.out_dtype]
        }
        self.inputs = {'X': self.x_np, 'Travel': travel_np, 'Layer': layer_np}
        self.outputs = {'Out': out, 'Labels': label, 'Mask': mask}

    def config(self):
        """set test shape & type"""
        self.neg_samples_num_list = [0, 0, 0, 0]
        self.x_shape = (10, 1)
        self.x_type = 'int32'
        self.tree_dtype = 'int32'
        self.out_dtype = 'int32'

    def test_check_output(self):
        places = self._get_places()
        for place in places:
            outs, fetch_list = self._calc_output(place)
            self.out = [np.array(out) for out in outs]

        x_res = self.out[fetch_list.index('Out')]
        label_res = self.out[fetch_list.index('Labels')]
        mask_res = self.out[fetch_list.index('Mask')]

        # check dtype
        if self.out_dtype == 'int32':
            assert x_res.dtype == np.int32
            assert label_res.dtype == np.int32
            assert mask_res.dtype == np.int32
        elif self.out_dtype == 'int64':
            assert x_res.dtype == np.int64
            assert label_res.dtype == np.int64
            assert mask_res.dtype == np.int64

        x_res = x_res.reshape(self.output_shape)
        label_res = label_res.reshape(self.output_shape)
        mask_res = mask_res.reshape(self.output_shape)

        layer_nums = len(self.neg_samples_num_list)
        for batch_ids, x_batch in enumerate(x_res):
            start_offset = 0
            positive_travel = []
            for layer_idx in range(layer_nums):
                end_offset = start_offset + self.layer_sample_nums[layer_idx]
                sampling_res = x_batch[start_offset:end_offset]
                sampling_res_list = sampling_res.tolist()
                positive_travel.append(sampling_res_list[0])

                label_sampling_res = label_res[batch_ids][start_offset:
                                                          end_offset]
                mask_sampling_res = mask_res[batch_ids][start_offset:end_offset]

                # check unique
                if sampling_res_list[0] != 0:
                    assert len(set(sampling_res_list)) == len(
                        sampling_res_list
                    ), "len(set(sampling_res_list)): {}, len(sampling_res_list): {} , sample_res: {}, label_res:{}, mask_res: {}".format(
                        len(set(sampling_res_list)),
                        len(sampling_res_list), sampling_res,
                        label_sampling_res, mask_sampling_res)
                # check legal
                layer_node = self.tree_layer[layer_idx]
                layer_node.append(0)
                for sample in sampling_res_list:
                    assert (
                        sample in layer_node
                    ), "sample: {}, layer_node: {} , sample_res: {}, label_res: {}, mask_res:{}".format(
                        sample, layer_node, sampling_res, label_sampling_res,
                        mask_sampling_res)

                # check label
                label_flag = 1
                if sampling_res[0] == 0:
                    label_flag = 0
                assert label_sampling_res[0] == label_flag
                # check mask
                padding_index = np.where(sampling_res == 0)
                assert not np.sum(
                    mask_sampling_res[padding_index]
                ), "np.sum(mask_sampling_res[padding_index]): {} ".format(
                    np.sum(mask_sampling_res[padding_index]))
                start_offset = end_offset
            # check travel legal
            assert self.tree_travel[int(self.x_np[
                batch_ids])] == positive_travel


class TestCase1(TestTDMSamplerOp):
    def config(self):
        """test input int64"""
        self.neg_samples_num_list = [0, 0, 0, 0]
        self.x_shape = (10, 1)
        self.x_type = 'int64'
        self.tree_dtype = 'int64'
        self.out_dtype = 'int32'


class TestCase2(TestTDMSamplerOp):
    def config(self):
        """test dtype int64"""
        self.neg_samples_num_list = [0, 0, 0, 0]
        self.x_shape = (10, 1)
        self.x_type = 'int32'
        self.tree_dtype = 'int32'
        self.out_dtype = 'int64'


class TestCase3(TestTDMSamplerOp):
    def config(self):
        """test all dtype int64"""
        self.neg_samples_num_list = [0, 0, 0, 0]
        self.x_shape = (10, 1)
        self.x_type = 'int64'
        self.tree_dtype = 'int64'
        self.out_dtype = 'int64'


class TestCase4(TestTDMSamplerOp):
    def config(self):
        """test one neg"""
        self.neg_samples_num_list = [1, 1, 1, 1]
        self.x_shape = (10, 1)
        self.x_type = 'int64'
        self.tree_dtype = 'int32'
        self.out_dtype = 'int64'


class TestCase5(TestTDMSamplerOp):
    def config(self):
        """test normal neg"""
        self.neg_samples_num_list = [1, 2, 3, 4]
        self.x_shape = (10, 1)
        self.x_type = 'int64'
        self.tree_dtype = 'int32'
        self.out_dtype = 'int64'


class TestCase6(TestTDMSamplerOp):
    def config(self):
        """test huge batchsize"""
        self.neg_samples_num_list = [1, 2, 3, 4]
        self.x_shape = (100, 1)
        self.x_type = 'int64'
        self.tree_dtype = 'int32'
        self.out_dtype = 'int64'


class TestCase7(TestTDMSamplerOp):
    def config(self):
        """test full neg"""
        self.neg_samples_num_list = [1, 3, 6, 11]
        self.x_shape = (10, 1)
        self.x_type = 'int64'
        self.tree_dtype = 'int32'
        self.out_dtype = 'int64'


class TestTDMSamplerShape(unittest.TestCase):
    def test_shape(self):
        x = fluid.layers.data(name='x', shape=[1], dtype='int32', lod_level=1)
        tdm_tree_travel = create_tdm_travel()
        tdm_tree_layer = create_tdm_layer()
        layer_node_num_list = [len(i) for i in tdm_tree_layer]

        tree_layer_flat = []
        for layer_idx, layer_node in enumerate(layer_node_num_list):
            tree_layer_flat += tdm_tree_layer[layer_idx]

        travel_array = np.array(tdm_tree_travel).astype('int32')
        layer_array = np.array(tree_layer_flat).astype('int32')

        neg_samples_num_list = [1, 2, 3, 4]
        leaf_node_num = 13

        sample, label, mask = fluid.contrib.layers.tdm_sampler(
            x,
            neg_samples_num_list,
            layer_node_num_list,
            leaf_node_num,
            tree_travel_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    travel_array)),
            tree_layer_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    layer_array)),
            output_positive=True,
            output_list=True,
            seed=0,
            tree_dtype='int32',
            dtype='int32')

        place = fluid.CPUPlace()
        exe = fluid.Executor(place=place)
        exe.run(fluid.default_startup_program())

        feed = {
            'x': np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9],
                           [10], [11], [12]]).astype('int32')
        }
        exe.run(feed=feed)


class TestTDMSamplerRadiusShape(unittest.TestCase):
    def test_shape(self):
        x = fluid.layers.data(name='x', shape=[1], dtype='int32', lod_level=1)
        tdm_tree_travel = create_tdm_travel()
        tdm_tree_layer = create_tdm_layer()
        layer_node_num_list = [len(i) for i in tdm_tree_layer]

        tree_layer_flat = []
        for layer_idx, layer_node in enumerate(layer_node_num_list):
            tree_layer_flat += tdm_tree_layer[layer_idx]

        travel_array = np.array(tdm_tree_travel).astype('int32')
        layer_array = np.array(tree_layer_flat).astype('int32')

        neg_samples_num_list = [1, 2, 3, 4]
        neg_samples_radius_list = [1, 2, 3, 4]
        leaf_node_num = 13

        sample, label, mask = fluid.contrib.layers.tdm_sampler(
            x,
            neg_samples_num_list,
            layer_node_num_list,
            leaf_node_num,
            neg_samples_radius_list=neg_samples_radius_list,
            tree_travel_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    travel_array)),
            tree_layer_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    layer_array)),
            output_positive=True,
            output_list=True,
            seed=0,
            tree_dtype='int32',
            dtype='int32')

        place = fluid.CPUPlace()
        exe = fluid.Executor(place=place)
        exe.run(fluid.default_startup_program())

        feed = {
            'x': np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9],
                           [10], [11], [12]]).astype('int32')
        }
        exe.run(feed=feed)


if __name__ == "__main__":
    unittest.main()
