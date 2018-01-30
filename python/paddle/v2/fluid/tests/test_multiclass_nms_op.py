#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import unittest
import numpy as np
import copy
from op_test import OpTest


def iou(box_a, box_b):
    """Apply intersection-over-union overlap between box_a and box_b
    """
    xmin_a = min(box_a[0], box_a[2])
    ymin_a = min(box_a[1], box_a[3])
    xmax_a = max(box_a[0], box_a[2])
    ymax_a = max(box_a[1], box_a[3])

    xmin_b = min(box_b[0], box_b[2])
    ymin_b = min(box_b[1], box_b[3])
    xmax_b = max(box_b[0], box_b[2])
    ymax_b = max(box_b[1], box_b[3])

    area_a = (ymax_a - ymin_a) * (xmax_a - xmin_a)
    area_b = (ymax_b - ymin_b) * (xmax_b - xmin_b)
    if area_a <= 0 and area_b <= 0:
        return 0.0

    xa = max(xmin_a, xmin_b)
    ya = max(ymin_a, ymin_b)
    xb = min(xmax_a, xmax_b)
    yb = min(ymax_a, ymax_b)

    inter_area = max(xb - xa, 0.0) * max(yb - ya, 0.0)

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    iou_ratio = inter_area / (area_a + area_b - inter_area)

    return iou_ratio


def nms(boxes, scores, score_threshold, nms_threshold, top_k=200, eta=1.0):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    all_scores = copy.deepcopy(scores)
    all_scores = all_scores.flatten()
    selected_indices = np.argwhere(all_scores > score_threshold)
    selected_indices = selected_indices.flatten()
    all_scores = all_scores[selected_indices]

    sorted_indices = np.argsort(-all_scores, axis=0)
    sorted_scores = all_scores[sorted_indices]
    if top_k > -1 and top_k < sorted_indices.shape[0]:
        sorted_indices = sorted_indices[:top_k]
        sorted_scores = sorted_scores[:top_k]

    selected_indices = []
    adaptive_threshold = nms_threshold
    for i in range(sorted_scores.shape[0]):
        idx = sorted_indices[i]
        keep = True
        for k in range(len(selected_indices)):
            if keep:
                kept_idx = selected_indices[k]
                overlap = iou(boxes[idx], boxes[kept_idx])
                keep = True if overlap <= adaptive_threshold else False
            else:
                break
        if keep:
            selected_indices.append(idx)
        if keep and eta < 1 and adaptive_threshold > 0.5:
            adaptive_threshold *= eta
    return selected_indices


def multiclass_nms(boxes, scores, background, score_threshold, nms_threshold,
                   nms_top_k, keep_top_k):
    class_num = scores.shape[0]
    priorbox_num = scores.shape[1]

    selected_indices = []
    num_det = 0
    for c in range(class_num):
        if c == background: continue
        indices = nms(boxes, scores[c], score_threshold, nms_threshold,
                      nms_top_k)
        for idx in indices:
            selected_indices.append((c, idx))
        num_det += len(indices)

    if keep_top_k > -1 and num_det > keep_top_k:
        score_index = []
        for c, idx in selected_indices:
            score_index.append((scores[c][idx], c, idx))

        sorted_score_index = sorted(
            score_index, key=lambda tup: tup[0], reverse=True)
        sorted_score_index = sorted_score_index[:keep_top_k]
        selected_indices = []
        for s, c, idx in sorted_score_index:
            selected_indices.append((c, idx))

    return selected_indices


def batched_multiclass_nms(boxes, scores, background, score_threshold,
                           nms_threshold, nms_top_k, keep_top_k):
    batch_size = scores.shape[0]

    det_outs = []
    lod = [0]
    for n in range(batch_size):
        nmsed_outs = multiclass_nms(boxes, scores[n], background,
                                    score_threshold, nms_threshold, nms_top_k,
                                    keep_top_k)
        lod.append(lod[-1] + len(nmsed_outs))
        if len(nmsed_outs) == 0: continue
        for c, idx in nmsed_outs:
            xmin, ymin, xmax, ymax = boxes[idx][:]
            det_outs.append([c, scores[n][c][idx], xmin, ymin, xmax, ymax])
    return det_outs, lod


class TestMulticlassNMSOp(OpTest):
    def setUp(self):
        N = 7
        M = 1240
        C = 21
        BOX_SIZE = 4
        background = 0
        nms_threshold = 0.3
        nms_top_k = 400
        keep_top_k = 200
        score_threshold = 0.01

        scores = np.random.random((N * M, C)).astype('float32')

        def softmax(x):
            shiftx = x - np.max(x).clip(-64.)
            exps = np.exp(shiftx)
            return exps / np.sum(exps)

        scores = np.apply_along_axis(softmax, 1, scores)
        scores = np.reshape(scores, (N, M, C))
        scores = np.transpose(scores, (0, 2, 1))

        boxes = np.random.random((M, BOX_SIZE)).astype('float32')
        boxes[:, 0:2] = boxes[:, 0:2] * 0.5
        boxes[:, 2:4] = boxes[:, 0:2] * 0.5 + 0.5

        nmsed_outs, lod = batched_multiclass_nms(boxes, scores, background,
                                                 score_threshold, nms_threshold,
                                                 nms_top_k, keep_top_k)
        nmsed_outs = np.array(nmsed_outs).astype('float32')

        self.op_type = 'multiclass_nms'
        self.inputs = {'Bboxes': boxes, 'Scores': scores}
        self.outputs = {'Out': (nmsed_outs, [lod])}
        self.attrs = {
            'background_label': 0,
            'nms_threshold': nms_threshold,
            'nms_top_k': nms_top_k,
            'keep_top_k': keep_top_k,
            'score_threshold': score_threshold,
            'nms_eta': 1.0,
        }

    def test_check_output(self):
        self.check_output()


class TestIOU(unittest.TestCase):
    def test_iou(self):
        box1 = np.array([4.0, 3.0, 7.0, 5.0]).astype('float32')
        box2 = np.array([3.0, 4.0, 6.0, 8.0]).astype('float32')

        expt_output = np.array([2.0 / 16.0]).astype('float32')
        calc_output = np.array([iou(box1, box2)]).astype('float32')
        self.assertTrue(np.allclose(calc_output, expt_output))


if __name__ == '__main__':
    unittest.main()
