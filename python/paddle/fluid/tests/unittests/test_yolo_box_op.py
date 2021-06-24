#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division

import unittest
import numpy as np
from op_test import OpTest

import paddle
from paddle.fluid import core


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def YoloBox(x, img_size, attrs):
    n, c, h, w = x.shape
    anchors = attrs['anchors']
    an_num = int(len(anchors) // 2)
    class_num = attrs['class_num']
    conf_thresh = attrs['conf_thresh']
    downsample = attrs['downsample']
    clip_bbox = attrs['clip_bbox']
    scale_x_y = attrs['scale_x_y']
    iou_aware = attrs['iou_aware']
    iou_aware_factor = attrs['iou_aware_factor']
    bias_x_y = -0.5 * (scale_x_y - 1.)
    input_h = downsample * h
    input_w = downsample * w

    if iou_aware:
        ioup = x[:, :an_num, :, :]
        ioup = np.expand_dims(ioup, axis=-1)
        x = x[:, an_num:, :, :]
    x = x.reshape((n, an_num, 5 + class_num, h, w)).transpose((0, 1, 3, 4, 2))

    pred_box = x[:, :, :, :, :4].copy()
    grid_x = np.tile(np.arange(w).reshape((1, w)), (h, 1))
    grid_y = np.tile(np.arange(h).reshape((h, 1)), (1, w))
    pred_box[:, :, :, :, 0] = (
        grid_x + sigmoid(pred_box[:, :, :, :, 0]) * scale_x_y + bias_x_y) / w
    pred_box[:, :, :, :, 1] = (
        grid_y + sigmoid(pred_box[:, :, :, :, 1]) * scale_x_y + bias_x_y) / h

    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    anchors_s = np.array(
        [(an_w / input_w, an_h / input_h) for an_w, an_h in anchors])
    anchor_w = anchors_s[:, 0:1].reshape((1, an_num, 1, 1))
    anchor_h = anchors_s[:, 1:2].reshape((1, an_num, 1, 1))
    pred_box[:, :, :, :, 2] = np.exp(pred_box[:, :, :, :, 2]) * anchor_w
    pred_box[:, :, :, :, 3] = np.exp(pred_box[:, :, :, :, 3]) * anchor_h

    if iou_aware:
        pred_conf = sigmoid(x[:, :, :, :, 4:5])**(
            1 - iou_aware_factor) * sigmoid(ioup)**iou_aware_factor
    else:
        pred_conf = sigmoid(x[:, :, :, :, 4:5])
    pred_conf[pred_conf < conf_thresh] = 0.
    pred_score = sigmoid(x[:, :, :, :, 5:]) * pred_conf
    pred_box = pred_box * (pred_conf > 0.).astype('float32')

    pred_box = pred_box.reshape((n, -1, 4))
    pred_box[:, :, :2], pred_box[:, :, 2:4] = \
        pred_box[:, :, :2] - pred_box[:, :, 2:4] / 2., \
        pred_box[:, :, :2] + pred_box[:, :, 2:4] / 2.0
    pred_box[:, :, 0] = pred_box[:, :, 0] * img_size[:, 1][:, np.newaxis]
    pred_box[:, :, 1] = pred_box[:, :, 1] * img_size[:, 0][:, np.newaxis]
    pred_box[:, :, 2] = pred_box[:, :, 2] * img_size[:, 1][:, np.newaxis]
    pred_box[:, :, 3] = pred_box[:, :, 3] * img_size[:, 0][:, np.newaxis]

    if clip_bbox:
        for i in range(len(pred_box)):
            pred_box[i, :, 0] = np.clip(pred_box[i, :, 0], 0, np.inf)
            pred_box[i, :, 1] = np.clip(pred_box[i, :, 1], 0, np.inf)
            pred_box[i, :, 2] = np.clip(pred_box[i, :, 2], -np.inf,
                                        img_size[i, 1] - 1)
            pred_box[i, :, 3] = np.clip(pred_box[i, :, 3], -np.inf,
                                        img_size[i, 0] - 1)

    return pred_box, pred_score.reshape((n, -1, class_num))


class TestYoloBoxOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'yolo_box'
        x = np.random.random(self.x_shape).astype('float32')
        img_size = np.random.randint(10, 20, self.imgsize_shape).astype('int32')

        self.attrs = {
            "anchors": self.anchors,
            "class_num": self.class_num,
            "conf_thresh": self.conf_thresh,
            "downsample": self.downsample,
            "clip_bbox": self.clip_bbox,
            "scale_x_y": self.scale_x_y,
            "iou_aware": self.iou_aware,
            "iou_aware_factor": self.iou_aware_factor
        }

        self.inputs = {
            'X': x,
            'ImgSize': img_size,
        }
        boxes, scores = YoloBox(x, img_size, self.attrs)
        self.outputs = {
            "Boxes": boxes,
            "Scores": scores,
        }

    def test_check_output(self):
        self.check_output()

    def initTestCase(self):
        self.anchors = [10, 13, 16, 30, 33, 23]
        an_num = int(len(self.anchors) // 2)
        self.batch_size = 32
        self.class_num = 2
        self.conf_thresh = 0.5
        self.downsample = 32
        self.clip_bbox = True
        self.x_shape = (self.batch_size, an_num * (5 + self.class_num), 13, 13)
        self.imgsize_shape = (self.batch_size, 2)
        self.scale_x_y = 1.
        self.iou_aware = False
        self.iou_aware_factor = 0.5


class TestYoloBoxOpNoClipBbox(TestYoloBoxOp):
    def initTestCase(self):
        self.anchors = [10, 13, 16, 30, 33, 23]
        an_num = int(len(self.anchors) // 2)
        self.batch_size = 32
        self.class_num = 2
        self.conf_thresh = 0.5
        self.downsample = 32
        self.clip_bbox = False
        self.x_shape = (self.batch_size, an_num * (5 + self.class_num), 13, 13)
        self.imgsize_shape = (self.batch_size, 2)
        self.scale_x_y = 1.
        self.iou_aware = False
        self.iou_aware_factor = 0.5


class TestYoloBoxOpScaleXY(TestYoloBoxOp):
    def initTestCase(self):
        self.anchors = [10, 13, 16, 30, 33, 23]
        an_num = int(len(self.anchors) // 2)
        self.batch_size = 32
        self.class_num = 2
        self.conf_thresh = 0.5
        self.downsample = 32
        self.clip_bbox = True
        self.x_shape = (self.batch_size, an_num * (5 + self.class_num), 13, 13)
        self.imgsize_shape = (self.batch_size, 2)
        self.scale_x_y = 1.2
        self.iou_aware = False
        self.iou_aware_factor = 0.5


class TestYoloBoxOpIoUAware(TestYoloBoxOp):
    def initTestCase(self):
        self.anchors = [10, 13, 16, 30, 33, 23]
        an_num = int(len(self.anchors) // 2)
        self.batch_size = 32
        self.class_num = 2
        self.conf_thresh = 0.5
        self.downsample = 32
        self.clip_bbox = True
        self.x_shape = (self.batch_size, an_num * (6 + self.class_num), 13, 13)
        self.imgsize_shape = (self.batch_size, 2)
        self.scale_x_y = 1.
        self.iou_aware = True
        self.iou_aware_factor = 0.5


class TestYoloBoxDygraph(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()
        img_size = np.ones((2, 2)).astype('int32')
        img_size = paddle.to_tensor(img_size)

        x1 = np.random.random([2, 14, 8, 8]).astype('float32')
        x1 = paddle.to_tensor(x1)
        boxes, scores = paddle.vision.ops.yolo_box(
            x1,
            img_size=img_size,
            anchors=[10, 13, 16, 30],
            class_num=2,
            conf_thresh=0.01,
            downsample_ratio=8,
            clip_bbox=True,
            scale_x_y=1.)
        assert boxes is not None and scores is not None

        x2 = np.random.random([2, 16, 8, 8]).astype('float32')
        x2 = paddle.to_tensor(x2)
        boxes, scores = paddle.vision.ops.yolo_box(
            x2,
            img_size=img_size,
            anchors=[10, 13, 16, 30],
            class_num=2,
            conf_thresh=0.01,
            downsample_ratio=8,
            clip_bbox=True,
            scale_x_y=1.,
            iou_aware=True,
            iou_aware_factor=0.5)
        paddle.enable_static()


class TestYoloBoxStatic(unittest.TestCase):
    def test_static(self):
        x1 = paddle.static.data('x1', [2, 14, 8, 8], 'float32')
        img_size = paddle.static.data('img_size', [2, 2], 'int32')

        boxes, scores = paddle.vision.ops.yolo_box(
            x1,
            img_size=img_size,
            anchors=[10, 13, 16, 30],
            class_num=2,
            conf_thresh=0.01,
            downsample_ratio=8,
            clip_bbox=True,
            scale_x_y=1.)
        assert boxes is not None and scores is not None

        x2 = paddle.static.data('x2', [2, 16, 8, 8], 'float32')
        boxes, scores = paddle.vision.ops.yolo_box(
            x2,
            img_size=img_size,
            anchors=[10, 13, 16, 30],
            class_num=2,
            conf_thresh=0.01,
            downsample_ratio=8,
            clip_bbox=True,
            scale_x_y=1.,
            iou_aware=True,
            iou_aware_factor=0.5)
        assert boxes is not None and scores is not None


class TestYoloBoxOpHW(TestYoloBoxOp):
    def initTestCase(self):
        self.anchors = [10, 13, 16, 30, 33, 23]
        an_num = int(len(self.anchors) // 2)
        self.batch_size = 32
        self.class_num = 2
        self.conf_thresh = 0.5
        self.downsample = 32
        self.clip_bbox = False
        self.x_shape = (self.batch_size, an_num * (5 + self.class_num), 13, 9)
        self.imgsize_shape = (self.batch_size, 2)
        self.scale_x_y = 1.
        self.iou_aware = False
        self.iou_aware_factor = 0.5


if __name__ == "__main__":
    unittest.main()
