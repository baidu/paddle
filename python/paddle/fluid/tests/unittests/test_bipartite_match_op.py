#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import print_function
import unittest
import numpy as np
from op_test import OpTest


def bipartite_match(distance, match_indices, match_dist):
    """Bipartite Matching algorithm.
    Arg:
        distance (numpy.array) : The distance of two entries with shape [M, N].
        match_indices (numpy.array): the matched indices from column to row
            with shape [1, N], it must be initialized to -1.
        match_dist (numpy.array): The matched distance from column to row
            with shape [1, N], it must be initialized to 0.
    """
    match_pair = []
    row, col = distance.shape
    for i in range(row):
        for j in range(col):
            match_pair.append((i, j, distance[i][j]))

    match_sorted = sorted(match_pair, key=lambda tup: tup[2], reverse=True)

    row_indices = -1 * np.ones((row, ), dtype=np.int)

    idx = 0
    for i, j, dist in match_sorted:
        if idx >= row:
            break
        if match_indices[j] == -1 and row_indices[i] == -1 and dist > 0:
            match_indices[j] = i
            row_indices[i] = j
            match_dist[j] = dist
            idx += 1


def argmax_match(distance, match_indices, match_dist, threshold):
    r, c = distance.shape
    for j in range(c):
        if match_indices[j] != -1:
            continue
        col_dist = distance[:, j]
        indices = np.argwhere(col_dist >= threshold).flatten()
        if len(indices) < 1:
            continue
        match_indices[j] = indices[np.argmax(col_dist[indices])]
        match_dist[j] = col_dist[match_indices[j]]


def batch_bipartite_match(distance,
                          lod=None,
                          match_type=None,
                          dist_threshold=None):
    """Bipartite Matching algorithm for batch input.
    Arg:
        distance (numpy.array) : The distance of two entries with shape [M, N] or
                            shape [B, A, N], where M = B * A.
        lod (list of int): The length of each input in this batch.
    """
    if lod is not None:
        n = len(lod)
        m = distance.shape[1]
        match_indices = -1 * np.ones((n, m), dtype=np.int)
        match_dist = np.zeros((n, m), dtype=np.float32)
        cur_offset = 0
        for i in range(n):
            if lod[i] == 0: continue
            bipartite_match(distance[cur_offset:(cur_offset + lod[i]), :],
                            match_indices[i, :], match_dist[i, :])
            if match_type == 'per_prediction':
                argmax_match(distance[cur_offset:(cur_offset + lod[i]), :],
                             match_indices[i, :], match_dist[i, :],
                             dist_threshold)
            cur_offset += lod[i]
    else:
        n = distance.shape[0]
        m = distance.shape[-1]
        match_indices = -1 * np.ones((n, m), dtype=np.int)
        match_dist = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            bipartite_match(distance[i], match_indices[i, :], match_dist[i, :])
            if match_type == 'per_prediction':
                argmax_match(distance[i], match_indices[i, :], match_dist[i, :],
                             dist_threshold)
    return match_indices, match_dist


class TestBipartiteMatchOpWithLoD(OpTest):
    def setUp(self):
        self.op_type = 'bipartite_match'
        lod = [[5, 6, 12]]
        dist = np.random.random((23, 217)).astype('float32')
        match_indices, match_dist = batch_bipartite_match(dist, lod[0])

        self.inputs = {'DistMat': (dist, lod)}
        self.outputs = {
            'ColToRowMatchIndices': match_indices,
            'ColToRowMatchDist': match_dist,
        }

    def test_check_output(self):
        self.check_output()


class TestBipartiteMatchOpWithoutLoD(OpTest):
    def setUp(self):
        self.op_type = 'bipartite_match'
        lod = [[8]]
        dist = np.random.random((8, 17)).astype('float32')
        match_indices, match_dist = batch_bipartite_match(dist, lod[0])

        self.inputs = {'DistMat': dist}
        self.outputs = {
            'ColToRowMatchIndices': match_indices,
            'ColToRowMatchDist': match_dist,
        }

    def test_check_output(self):
        self.check_output()


class TestBipartiteMatchOpWithoutLoDLargeScaleInput(OpTest):
    def setUp(self):
        self.op_type = 'bipartite_match'
        lod = [[300]]
        dist = np.random.random((300, 17)).astype('float32')
        match_indices, match_dist = batch_bipartite_match(dist, lod[0])

        self.inputs = {'DistMat': dist}
        self.outputs = {
            'ColToRowMatchIndices': match_indices,
            'ColToRowMatchDist': match_dist,
        }

    def test_check_output(self):
        self.check_output()


class TestBipartiteMatchOpWithPerPredictionType(OpTest):
    def setUp(self):
        self.op_type = 'bipartite_match'
        lod = [[5, 6, 12]]
        dist = np.random.random((23, 237)).astype('float32')
        match_indices, match_dist = batch_bipartite_match(dist, lod[0],
                                                          'per_prediction', 0.5)

        self.inputs = {'DistMat': (dist, lod)}
        self.outputs = {
            'ColToRowMatchIndices': match_indices,
            'ColToRowMatchDist': match_dist,
        }
        self.attrs = {
            'match_type': 'per_prediction',
            'dist_threshold': 0.5,
        }

    def test_check_output(self):
        self.check_output()


class TestBipartiteMatchOpWithEmptyLoD(OpTest):
    def setUp(self):
        self.op_type = 'bipartite_match'
        lod = [[5, 6, 0, 12]]
        dist = np.random.random((23, 217)).astype('float32')
        match_indices, match_dist = batch_bipartite_match(dist, lod[0])

        self.inputs = {'DistMat': (dist, lod)}
        self.outputs = {
            'ColToRowMatchIndices': match_indices,
            'ColToRowMatchDist': match_dist,
        }

    def test_check_output(self):
        self.check_output()


class TestBipartiteMatchOpWith3DTensor(OpTest):
    def setUp(self):
        self.op_type = 'bipartite_match'
        dist = np.random.random((3, 20, 123)).astype('float32')
        match_indices, match_dist = batch_bipartite_match(dist)

        self.inputs = {'DistMat': dist}
        self.outputs = {
            'ColToRowMatchIndices': match_indices,
            'ColToRowMatchDist': match_dist,
        }

    def test_check_output(self):
        self.check_output()


class TestBipartiteMatchOpWith3DTensorPerPredictionType(OpTest):
    def setUp(self):
        self.op_type = 'bipartite_match'
        dist = np.random.random((3, 20, 123)).astype('float32')
        match_indices, match_dist = batch_bipartite_match(
            dist, match_type='per_prediction', dist_threshold=0.5)

        self.inputs = {'DistMat': dist}
        self.outputs = {
            'ColToRowMatchIndices': match_indices,
            'ColToRowMatchDist': match_dist,
        }
        self.attrs = {
            'match_type': 'per_prediction',
            'dist_threshold': 0.5,
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
