#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import random
from op_test import OpTest


def gen_match_and_neg_indices(num_prior,
                              gt_lod=None,
                              neg_lod=None,
                              batch_size=None,
                              num_gt=None):
    if gt_lod is not None and neg_lod is not None:
        if len(gt_lod) != len(neg_lod):
            raise AssertionError("The input arguments are illegal.")

        batch_size = len(gt_lod)

        match_indices = -1 * np.ones((batch_size, num_prior)).astype('int32')
        neg_indices = np.zeros((sum(neg_lod), 1)).astype('int32')

        offset = 0
        for n in range(batch_size):
            gt_num = gt_lod[n]
            ids = random.sample([i for i in range(num_prior)], gt_num)
            match_indices[n, ids] = [i for i in range(gt_num)]

            ret_ids = set([i for i in range(num_prior)]) - set(ids)
            l = neg_lod[n]
            neg_ids = random.sample(ret_ids, l)
            neg_indices[offset:offset + neg_lod[n], :] = np.array(
                neg_ids).astype('int32').reshape(l, 1)
            offset += neg_lod[n]
        return match_indices, neg_indices
    else:
        match_indices = -1 * np.ones((batch_size, num_prior)).astype('int32')
        neg_mask = np.zeros((batch_size, num_prior)).astype('int32')

        for n in range(batch_size):
            ids = random.sample([i for i in range(num_prior)], num_gt)
            match_indices[n, ids] = [i for i in range(num_gt)]

            ret_ids = set([i for i in range(num_prior)]) - set(ids)
            l = random.randint(0, len(ret_ids))
            neg_ids = random.sample(ret_ids, l)
            neg_mask[n, neg_ids] = 1
        return match_indices, neg_mask.reshape((batch_size, num_prior, 1))


def target_assign(encoded_box,
                  gt_label,
                  match_indices,
                  neg_indices=None,
                  gt_lod=None,
                  neg_lod=None,
                  mismatch_value=0):
    batch_size, num_prior = match_indices.shape

    # init target bbox
    trg_box = np.zeros((batch_size, num_prior, 4)).astype('float32')
    # init weight for target bbox
    trg_box_wt = np.zeros((batch_size, num_prior, 1)).astype('float32')
    # init target label
    trg_label = np.ones((batch_size, num_prior, 1)).astype('int32')
    trg_label = trg_label * mismatch_value
    # init weight for target label
    trg_label_wt = np.zeros((batch_size, num_prior, 1)).astype('float32')

    if gt_lod is not None and neg_lod is not None:
        gt_offset = 0
        neg_offset = 0
        for i in range(batch_size):
            cur_indices = match_indices[i]
            col_ids = np.where(cur_indices > -1)
            col_val = cur_indices[col_ids]

            # target bbox
            for v, c in zip(col_val + gt_offset, col_ids[0].tolist()):
                trg_box[i][c][:] = encoded_box[v][c][:]
            # weight for target bbox
            trg_box_wt[i][col_ids] = 1.0

            trg_label[i][col_ids] = gt_label[col_val + gt_offset]
            trg_label_wt[i][col_ids] = 1.0
            # set target label weight to 1.0 for the negative samples
            if neg_indices is not None:
                neg_ids = neg_indices[neg_offset:neg_offset + neg_lod[i]]
                trg_label_wt[i][neg_ids] = 1.0
            # update offset
            gt_offset += gt_lod[i]
            neg_offset += neg_lod[i]
    else:
        for i in range(batch_size):
            cur_indices = match_indices[i]
            col_ids = np.where(cur_indices > -1)
            col_val = cur_indices[col_ids]

            # target bbox
            for v, c in zip(col_val, col_ids[0].tolist()):
                trg_box[i][c][:] = encoded_box[i][v][c][:]
            # weight for target bbox
            trg_box_wt[i][col_ids] = 1.0

            trg_label[i][col_ids] = gt_label[i][col_val]
            trg_label_wt[i][col_ids] = 1.0
        # set target label weight to 1.0 for the negative samples
        if neg_indices is not None:
            trg_label_wt += neg_indices

    return trg_box, trg_box_wt, trg_label, trg_label_wt


class TestTargetAssginFloatType(OpTest):
    def setUp(self):
        self.op_type = "target_assign"
        num_prior = 120
        num_class = 21
        gt_lod = [5, 6, 12]
        neg_lod = [4, 3, 6]
        mismatch_value = 0
        batch_size = len(gt_lod)
        num_gt = sum(gt_lod)

        encoded_box = np.random.random((num_gt, num_prior, 4)).astype('float32')
        gt_label = np.random.randint(
            num_class, size=(num_gt, 1)).astype('int32')

        match_indices, neg_indices = gen_match_and_neg_indices(num_prior,
                                                               gt_lod, neg_lod)

        out, out_wt, _, _ = target_assign(encoded_box, gt_label, match_indices,
                                          neg_indices, gt_lod, neg_lod,
                                          mismatch_value)

        # assign regression targets
        x = encoded_box
        self.inputs = {
            'X': (x, [gt_lod]),
            'MatchIndices': match_indices,
        }
        self.attrs = {'mismatch_value': mismatch_value}
        self.outputs = {
            'Out': out,
            'OutWeight': out_wt,
        }

    def test_check_output(self):
        self.check_output()


class TestTargetAssginIntType(OpTest):
    def setUp(self):
        self.op_type = "target_assign"
        num_prior = 120
        num_class = 21
        gt_lod = [5, 6, 12]
        neg_lod = [4, 3, 6]
        mismatch_value = 0
        batch_size = len(gt_lod)
        num_gt = sum(gt_lod)

        encoded_box = np.random.random((num_gt, num_prior, 4)).astype('float32')
        gt_label = np.random.randint(
            num_class, size=(num_gt, 1)).astype('int32')

        match_indices, neg_indices = gen_match_and_neg_indices(num_prior,
                                                               gt_lod, neg_lod)

        _, _, out, out_wt, = target_assign(encoded_box, gt_label, match_indices,
                                           neg_indices, gt_lod, neg_lod,
                                           mismatch_value)

        # assign cassification argets
        x = np.reshape(gt_label, (num_gt, 1, 1))
        self.inputs = {
            'X': (x, [gt_lod]),
            'MatchIndices': match_indices,
            'NegIndices': (neg_indices, [neg_lod]),
        }
        self.attrs = {'mismatch_value': mismatch_value}
        self.outputs = {
            'Out': out,
            'OutWeight': out_wt,
        }

    def test_check_output(self):
        self.check_output()


class TestTargetAssginFloatTypeWithTensor(OpTest):
    def setUp(self):
        self.op_type = "target_assign"
        num_prior = 120
        num_class = 21
        mismatch_value = 0
        batch_size = 3
        num_gt = 30

        encoded_box = np.random.random((batch_size, num_gt, num_prior,
                                        4)).astype('float32')
        gt_label = np.random.randint(
            num_class, size=(batch_size, num_gt, 1)).astype('int32')

        match_indices, neg_mask = gen_match_and_neg_indices(
            num_prior, batch_size=batch_size, num_gt=num_gt)

        out, out_wt, _, _ = target_assign(
            encoded_box,
            gt_label,
            match_indices,
            neg_mask,
            mismatch_value=mismatch_value)

        # assign regression targets
        x = encoded_box
        self.inputs = {
            'X': x,
            'MatchIndices': match_indices,
        }
        self.attrs = {'mismatch_value': mismatch_value}
        self.outputs = {
            'Out': out,
            'OutWeight': out_wt,
        }

    def test_check_output(self):
        self.check_output()


class TestTargetAssginIntTypeWithTensor(OpTest):
    def setUp(self):
        self.op_type = "target_assign"
        num_prior = 120
        num_class = 21
        mismatch_value = 0
        batch_size = 3
        num_gt = 3

        encoded_box = np.random.random((batch_size, num_gt, num_prior,
                                        4)).astype('float32')
        gt_label = np.random.randint(
            num_class, size=(batch_size, num_gt, 1)).astype('int32')

        match_indices, neg_mask = gen_match_and_neg_indices(
            num_prior, batch_size=batch_size, num_gt=num_gt)

        _, _, out, out_wt, = target_assign(
            encoded_box,
            gt_label,
            match_indices,
            neg_mask,
            mismatch_value=mismatch_value)

        # assign regression targets
        x = gt_label
        self.inputs = {
            'X': x,
            'MatchIndices': match_indices,
            'NegIndices': neg_mask,
        }
        self.attrs = {'mismatch_value': mismatch_value}
        self.outputs = {
            'Out': out,
            'OutWeight': out_wt,
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
