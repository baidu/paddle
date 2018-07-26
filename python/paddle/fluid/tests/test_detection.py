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

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.framework import Program, program_guard
import unittest


class TestDetection(unittest.TestCase):
    def test_detection_output(self):
        program = Program()
        with program_guard(program):
            pb = layers.data(
                name='prior_box',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            pbv = layers.data(
                name='prior_box_var',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            loc = layers.data(
                name='target_box',
                shape=[2, 10, 4],
                append_batch_size=False,
                dtype='float32')
            scores = layers.data(
                name='scores',
                shape=[2, 10, 20],
                append_batch_size=False,
                dtype='float32')
            out = layers.detection_output(
                scores=scores, loc=loc, prior_box=pb, prior_box_var=pbv)
            self.assertIsNotNone(out)
            self.assertEqual(out.shape[-1], 6)
        print((str(program)))

    def test_detection_api(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[4], dtype='float32')
            y = layers.data(name='y', shape=[4], dtype='float32')
            z = layers.data(name='z', shape=[4], dtype='float32', lod_level=1)
            iou = layers.iou_similarity(x=x, y=y)
            bcoder = layers.box_coder(
                prior_box=x,
                prior_box_var=y,
                target_box=z,
                code_type='encode_center_size')
            self.assertIsNotNone(iou)
            self.assertIsNotNone(bcoder)

            matched_indices, matched_dist = layers.bipartite_match(iou)
            self.assertIsNotNone(matched_indices)
            self.assertIsNotNone(matched_dist)

            gt = layers.data(
                name='gt', shape=[1, 1], dtype='int32', lod_level=1)
            trg, trg_weight = layers.target_assign(
                gt, matched_indices, mismatch_value=0)
            self.assertIsNotNone(trg)
            self.assertIsNotNone(trg_weight)

            gt2 = layers.data(
                name='gt2', shape=[10, 4], dtype='float32', lod_level=1)
            trg, trg_weight = layers.target_assign(
                gt2, matched_indices, mismatch_value=0)
            self.assertIsNotNone(trg)
            self.assertIsNotNone(trg_weight)

        print((str(program)))

    def test_ssd_loss(self):
        program = Program()
        with program_guard(program):
            pb = layers.data(
                name='prior_box',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            pbv = layers.data(
                name='prior_box_var',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            loc = layers.data(name='target_box', shape=[10, 4], dtype='float32')
            scores = layers.data(name='scores', shape=[10, 21], dtype='float32')
            gt_box = layers.data(
                name='gt_box', shape=[4], lod_level=1, dtype='float32')
            gt_label = layers.data(
                name='gt_label', shape=[1], lod_level=1, dtype='int32')
            loss = layers.ssd_loss(loc, scores, gt_box, gt_label, pb, pbv)
            self.assertIsNotNone(loss)
            self.assertEqual(loss.shape[-1], 1)
        print((str(program)))


class TestPriorBox(unittest.TestCase):
    def test_prior_box(self):
        data_shape = [3, 224, 224]
        images = fluid.layers.data(
            name='pixel', shape=data_shape, dtype='float32')
        conv1 = fluid.layers.conv2d(images, 3, 3, 2)
        box, var = layers.prior_box(
            input=conv1,
            image=images,
            min_sizes=[100.0],
            aspect_ratios=[1.],
            flip=True,
            clip=True)
        assert len(box.shape) == 4
        assert box.shape == var.shape
        assert box.shape[3] == 4


class TestAnchorGenerator(unittest.TestCase):
    def test_anchor_generator(self):
        data_shape = [3, 224, 224]
        images = fluid.layers.data(
            name='pixel', shape=data_shape, dtype='float32')
        conv1 = fluid.layers.conv2d(images, 3, 3, 2)
        anchor, var = fluid.layers.anchor_generator(
            input=conv1,
            anchor_sizes=[64, 128, 256, 512],
            aspect_ratios=[0.5, 1.0, 2.0],
            variance=[0.1, 0.1, 0.2, 0.2],
            stride=[16.0, 16.0],
            offset=0.5)
        assert len(anchor.shape) == 4
        assert anchor.shape == var.shape
        assert anchor.shape[3] == 4


class TestMultiBoxHead(unittest.TestCase):
    def test_multi_box_head(self):
        data_shape = [3, 224, 224]
        mbox_locs, mbox_confs, box, var = self.multi_box_head_output(data_shape)

        assert len(box.shape) == 2
        assert box.shape == var.shape
        assert box.shape[1] == 4
        assert mbox_locs.shape[1] == mbox_confs.shape[1]

    def multi_box_head_output(self, data_shape):
        images = fluid.layers.data(
            name='pixel', shape=data_shape, dtype='float32')
        conv1 = fluid.layers.conv2d(images, 3, 3, 2)
        conv2 = fluid.layers.conv2d(conv1, 3, 3, 2)
        conv3 = fluid.layers.conv2d(conv2, 3, 3, 2)
        conv4 = fluid.layers.conv2d(conv3, 3, 3, 2)
        conv5 = fluid.layers.conv2d(conv4, 3, 3, 2)

        mbox_locs, mbox_confs, box, var = layers.multi_box_head(
            inputs=[conv1, conv2, conv3, conv4, conv5, conv5],
            image=images,
            num_classes=21,
            min_ratio=20,
            max_ratio=90,
            aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
            base_size=300,
            offset=0.5,
            flip=True,
            clip=True)

        return mbox_locs, mbox_confs, box, var


class TestDetectionMAP(unittest.TestCase):
    def test_detection_map(self):
        program = Program()
        with program_guard(program):
            detect_res = layers.data(
                name='detect_res',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32')
            label = layers.data(
                name='label',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32')

            map_out = layers.detection_map(detect_res, label, 21)
            self.assertIsNotNone(map_out)
            self.assertEqual(map_out.shape, (1, ))
        print((str(program)))


if __name__ == '__main__':
    unittest.main()
