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
"""
Fluid Metrics

The metrics are accomplished via Python natively. 
"""
import numpy as np
import copy
import warnings

__all__ = [
    'MetricBase',
    'CompositeMetric',
    'Precision',
    'Recall',
    'Accuracy',
    'ChunkEvaluator',
    'EditDistance',
    'DetectionMAP',
    'Auc',
]


def _is_numpy_(var):
    return isinstance(var, (np.ndarray, np.generic))


def _is_number_(var):
    return isinstance(var, int) or isinstance(var, float) or (isinstance(
        var, np.ndarray) and var.shape == (1, ))


def _is_number_or_matrix_(var):
    return _is_number_(var) or isinstance(var, np.ndarray)


class MetricBase(object):
    """
    Base Class for all Metrics.
    MetricBase define a group of interfaces for the
    model evaluation methods. Metrics accumulate metric states between
    consecutive minibatches, at every minibatch, use update
    interface to add current minibatch value to global states.
    Use eval to compute accumative metric value from last reset()
    or from scratch on.
    If you need to custom a new metric, please inherit from MetricBase and
    custom implementation.

    Args:
        name(str): The name of metric instance. such as, "accuracy".
                  It needed if you want to distinct different metrics in a model.

    """

    def __init__(self, name):
        self._name = str(name) if name != None else self.__class__.__name__

    def __str__(self):
        return self._name

    def reset(self):
        """
        reset clear the states of metrics. By default, the states
        are the members who do not has _ prefix, reset set them to inital states.
        If you violate the implicit name rule, please also custom the reset
        interface.
        """
        states = {
            attr: value
            for attr, value in list(self.__dict__.items())
            if not attr.startswith("_")
        }
        for attr, value in list(states.items()):
            if isinstance(value, int):
                setattr(self, attr, 0)
            elif isinstance(value, float):
                setattr(self, attr, .0)
            elif isinstance(value, (np.ndarray, np.generic)):
                setattr(self, attr, np.zeros_like(value))
            else:
                setattr(self, attr, None)

    def get_config(self):
        """
        Get the metric and current states.
        The states are the members who do not has "_" prefix.

        Args:
            None

        Returns:
            dict: a dict of metric and states
        """
        states = {
            attr: value
            for attr, value in list(self.__dict__.items())
            if not attr.startswith("_")
        }
        config = {}
        config.update({"name": self._name, "states": copy.deepcopy(states)})
        return config

    def update(self, preds, labels):
        """
        Updates the metric states at every minibatch.
        One user can compute the minibatch metric via pure Python, or
        via a c++ operator.

        Args:
            preds(numpy.array): the predictions of current minibatch
            labels(numpy.array): the labels of current minibatch, if the label is one-hot
                               or soft-label, should custom the corresponding update rule.
        """
        raise NotImplementedError(
            "Should not use it directly, please extend it.")

    def eval(self):
        """
        Evalute the current metrics based the accumulated states.

        Returns:
            float|list(float)|numpy.array: the metrics via Python.
        """
        raise NotImplementedError(
            "Should not use it directly, please extend it.")


class CompositeMetric(MetricBase):
    """
    Composite multiple metrics in one instance.
    for example, merge F1, accuracy, recall into one Metric.
    
    Examples:
        .. code-block:: python
    
          labels = fluid.layers.data(name="data", shape=[1], dtype="int32")
          data = fluid.layers.data(name="data", shape=[32, 32], dtype="int32")
          pred = fluid.layers.fc(input=data, size=1000, act="tanh")
          comp = fluid.metrics.CompositeMetric()
          acc = fluid.metrics.Precision()
          recall = fluid.metrics.Recall()
          comp.add_metric(acc)
          comp.add_metric(recall)
          for pass in range(PASSES):
            comp.reset()
            for data in train_reader():
                loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
            comp.update(preds=preds, labels=labels)
            numpy_acc, numpy_recall = comp.eval()
    """

    def __init__(self, name=None):
        super(CompositeMetric, self).__init__(name)
        self._metrics = []

    def add_metric(self, metric):
        """
        add one metric instance to CompositeMetric.

        Args:
            metric: a instance of MetricBase.
        """
        if not isinstance(metric, MetricBase):
            raise ValueError("SubMetric should be inherit from MetricBase.")
        self._metrics.append(metric)

    def update(self, preds, labels):
        """
        Update every metrics in sequence.

        Args:
            preds(numpy.array): the predictions of current minibatch
            labels(numpy.array): the labels of current minibatch, if the label is one-hot
                               or soft-label, should custom the corresponding update rule.
        """
        for m in self._metrics:
            ans.append(m.update(preds, labels))

    def eval(self):
        """
        Evaluate every metrics in sequence.

        Returns:
            list(float|numpy.array): a list of metrics value in Python.
        """
        ans = []
        for m in self._metrics:
            ans.append(m.eval())
        return ans


class Precision(MetricBase):
    """
    Precision (also called positive predictive value) is the fraction of
    relevant instances among the retrieved instances.
    https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers

    Note Precision is different with Accuracy in binary classifiers.
    accuracy = true positive / total instances
    precision = true positive / all positive instance

    Examples:
        .. code-block:: python

        metric = fluid.metrics.Precision()
        for pass in range(PASSES):
            metric.reset()
            for data in train_reader():
                loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
            metric.update(preds=preds, labels=labels)
            numpy_precision = metric.eval()
    """

    def __init__(self, name=None):
        super(Precision, self).__init__(name)
        self.tp = 0  # true positive
        self.fp = 0  # false positive

    def update(self, preds, labels):
        if not _is_numpy_(preds):
            raise ValueError("The 'preds' must be a numpy ndarray.")
        if not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray.")
        sample_num = labels[0]
        for i in range(sample_num):
            pred = preds[i].astype("int32")
            label = labels[i]
            if label == 1:
                if pred == label:
                    self.tp += 1
                else:
                    self.fp += 1

    def eval(self):
        ap = self.tp + self.fp
        return float(self.tp) / ap if ap != 0 else .0


class Recall(MetricBase):
    """
    Recall (also known as sensitivity) is the fraction of
    relevant instances that have been retrieved over the
    total amount of relevant instances

    https://en.wikipedia.org/wiki/Precision_and_recall

    Examples:
        .. code-block:: python

        metric = fluid.metrics.Recall()
        for pass in range(PASSES):
            metric.reset()
            for data in train_reader():
                loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
            metric.update(preds=preds, labels=labels)
            numpy_recall = metric.eval()
    """

    def __init__(self, name=None):
        super(Recall, self).__init__(name)
        self.tp = 0  # true positive
        self.fn = 0  # false negtive

    def update(self, preds, labels):
        if not _is_numpy_(preds):
            raise ValueError("The 'preds' must be a numpy ndarray.")
        if not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray.")
        sample_num = labels[0]
        for i in range(sample_num):
            pred = preds[i].astype("int32")
            label = labels[i]
            if label == 1:
                if pred == label:
                    self.tp += 1
            else:
                if pred != label:
                    self.fn += 1

    def eval(self):
        recall = self.tp + self.fn
        return float(self.tp) / recall if recall != 0 else .0


class Accuracy(MetricBase):
    """
    Accumulate the accuracy from minibatches and compute the average accuracy
    for every pass.
    https://en.wikipedia.org/wiki/Accuracy_and_precision

    Args:
       name: the metrics name

    Examples:
        .. code-block:: python

            labels = fluid.layers.data(name="data", shape=[1], dtype="int32")
            data = fluid.layers.data(name="data", shape=[32, 32], dtype="int32")
            pred = fluid.layers.fc(input=data, size=1000, act="tanh")
            minibatch_accuracy = fluid.layers.accuracy(pred, label)
            accuracy_evaluator = fluid.metrics.Accuracy()
            for pass in range(PASSES):
                accuracy_evaluator.reset()
                for data in train_reader():
                    batch_size = data[0]
                    loss = exe.run(fetch_list=[cost, minibatch_accuracy])
                accuracy_evaluator.update(value=minibatch_accuracy, weight=batch_size)
                numpy_acc = accuracy_evaluator.eval()
    """

    def __init__(self, name=None):
        super(Accuracy, self).__init__(name)
        self.value = .0
        self.weight = .0

    def update(self, value, weight):
        """
        Update minibatch states.

        Args:
            value(float|numpy.array): accuracy of one minibatch.
            weight(int|float): batch size.
        """
        if not _is_number_or_matrix_(value):
            raise ValueError(
                "The 'value' must be a number(int, float) or a numpy ndarray.")
        if not _is_number_(weight):
            raise ValueError("The 'weight' must be a number(int, float).")
        self.value += value * weight
        self.weight += weight

    def eval(self):
        if self.weight == 0:
            raise ValueError("There is no data in Accuracy Metrics. \
                Please check layers.accuracy output has added to Accuracy.")
        return self.value / self.weight


class ChunkEvaluator(MetricBase):
    """
    Accumulate counter numbers output by chunk_eval from mini-batches and
    compute the precision recall and F1-score using the accumulated counter
    numbers.
    For some basics of chunking, please refer to
    'Chunking with Support Vector Machines <https://aclanthology.info/pdf/N/N01/N01-1025.pdf>'.
    ChunkEvalEvaluator computes the precision, recall, and F1-score of chunk detection,
    and supports IOB, IOE, IOBES and IO (also known as plain) tagging schemes.

    Examples:
        .. code-block:: python

            labels = fluid.layers.data(name="data", shape=[1], dtype="int32")
            data = fluid.layers.data(name="data", shape=[32, 32], dtype="int32")
            pred = fluid.layers.fc(input=data, size=1000, act="tanh")
            precision, recall, f1_score, num_infer_chunks, num_label_chunks, num_correct_chunks = layers.chunk_eval(
                input=pred,
                label=label)
            metric = fluid.metrics.ChunkEvaluator()
            for data in train_reader():
                loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
                metric.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
                numpy_precision, numpy_recall, numpy_f1 = metric.eval()
    """

    def __init__(self, name=None):
        super(ChunkEvaluator, self).__init__(name)
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0

    def update(self, num_infer_chunks, num_label_chunks, num_correct_chunks):
        """
        Update the states based on the layers.chunk_eval() ouputs.
        Args:
            num_infer_chunks(int|numpy.array): The number of chunks in Inference on the given minibatch.
            num_label_chunks(int|numpy.array): The number of chunks in Label on the given mini-batch.
            num_correct_chunks(int|float|numpy.array): The number of chunks both in Inference and Label on the
                                                  given mini-batch.
        """
        if not _is_number_or_matrix_(num_infer_chunks):
            raise ValueError(
                "The 'num_infer_chunks' must be a number(int) or a numpy ndarray."
            )
        if not _is_number_or_matrix_(num_label_chunks):
            raise ValueError(
                "The 'num_label_chunks' must be a number(int, float) or a numpy ndarray."
            )
        if not _is_number_or_matrix_(num_correct_chunks):
            raise ValueError(
                "The 'num_correct_chunks' must be a number(int, float) or a numpy ndarray."
            )
        self.num_infer_chunks += num_infer_chunks
        self.num_label_chunks += num_label_chunks
        self.num_correct_chunks += num_correct_chunks

    def eval(self):
        precision = float(
            self.num_correct_chunks
        ) / self.num_infer_chunks if self.num_infer_chunks else 0
        recall = float(self.num_correct_chunks
                       ) / self.num_label_chunks if self.num_label_chunks else 0
        f1_score = float(2 * precision * recall) / (
            precision + recall) if self.num_correct_chunks else 0
        return precision, recall, f1_score


class EditDistance(MetricBase):
    """
    Edit distance is a way of quantifying how dissimilar two strings
    (e.g., words) are to one another by counting the minimum number
    of operations required to transform one string into the other.
    Refer to https://en.wikipedia.org/wiki/Edit_distance

    Accumulate edit distance sum and sequence number from mini-batches and
    compute the average edit_distance and instance error of all batches.

    Args:
        name: the metrics name

    Examples:
        .. code-block:: python

            distances, seq_num = fluid.layers.edit_distance(input, label)
            distance_evaluator = fluid.metrics.EditDistance()
            for epoch in PASS_NUM:
                distance_evaluator.reset()
                for data in batches:
                    loss = exe.run(fetch_list=[cost] + list(edit_distance_metrics))
                distance_evaluator.update(distances, seq_num)
                distance, instance_error = distance_evaluator.eval()

        In the above example:
        'distance' is the average of the edit distance in a pass.
        'instance_error' is the instance error rate in a pass.

    """

    def __init__(self, name):
        super(EditDistance, self).__init__(name)
        self.total_distance = .0
        self.seq_num = 0
        self.instance_error = 0

    def update(self, distances, seq_num):
        if not _is_numpy_(distances):
            raise ValueError("The 'distances' must be a numpy ndarray.")
        if not _is_number_(seq_num):
            raise ValueError("The 'seq_num' must be a number(int, float).")
        seq_right_count = np.sum(distances == 0)
        total_distance = np.sum(distances)
        self.seq_num += seq_num
        self.instance_error += seq_num - seq_right_count
        self.total_distance += total_distance

    def eval(self):
        if self.seq_num == 0:
            raise ValueError(
                "There is no data in EditDistance Metric. Please check layers.edit_distance output has been added to EditDistance."
            )
        avg_distance = self.total_distance / self.seq_num
        avg_instance_error = self.instance_error / self.seq_num
        return avg_distance, avg_instance_error


class DetectionMAP(MetricBase):
    """
    Calculate the detection mean average precision (mAP).
    mAP is the metric to measure the accuracy of object detectors
    like Faster R-CNN, SSD, etc.
    It is the average of the maximum precisions at different recall values.
    Please get more information from the following articles:
      https://sanchom.wordpress.com/tag/average-precision/

      https://arxiv.org/abs/1512.02325

    The general steps are as follows:

        1. calculate the true positive and false positive according to the input
            of detection and labels.
        2. calculate mAP value, support two versions: '11 point' and 'integral'.

    Examples:
        .. code-block:: python

            pred = fluid.layers.fc(input=data, size=1000, act="tanh")
            batch_map = layers.detection_map(
                input,
                label,
                class_num,
                background_label,
                overlap_threshold=overlap_threshold,
                evaluate_difficult=evaluate_difficult,
                ap_version=ap_version)
            metric = fluid.metrics.DetectionMAP()
            for data in train_reader():
                loss, preds, labels = exe.run(fetch_list=[cost, batch_map])
                batch_size = data[0]
                metric.update(value=batch_map, weight=batch_size)
                numpy_map = metric.eval()
    """

    def __init__(self, name=None):
        super(DetectionMAP, self).__init__(name)
        # the current map value
        self.value = .0
        self.weight = .0

    def update(self, value, weight):
        if not _is_number_or_matrix_(value):
            raise ValueError(
                "The 'value' must be a number(int, float) or a numpy ndarray.")
        if not _is_number_(weight):
            raise ValueError("The 'weight' must be a number(int, float).")
        self.value += value
        self.weight += weight

    def eval(self):
        if self.weight == 0:
            raise ValueError(
                "There is no data in DetectionMAP Metrics. "
                "Please check layers.detection_map output has added to DetectionMAP."
            )
        return self.value / self.weight


class Auc(MetricBase):
    """
    Auc metric adapts to the binary classification.
    Refer to https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
    Need to note that auc metric compute the value via Python natively.
    If you concern the speed, please use the fluid.layers.auc instead.

    The `auc` function creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the AUC. To discretize the AUC curve, a linearly spaced set of
    thresholds is used to compute pairs of recall and precision values. The area
    under the ROC-curve is therefore computed using the height of the recall
    values by the false positive rate, while the area under the PR-curve is the
    computed using the height of the precision values by the recall.

    Args:
        name: metric name
        curve: Specifies the name of the curve to be computed, 'ROC' [default] or
          'PR' for the Precision-Recall-curve.
        num_thresholds: The number of thresholds to use when discretizing the roc
            curve.

    "NOTE: only implement the ROC curve type via Python now."

    Examples:
        .. code-block:: python

            pred = fluid.layers.fc(input=data, size=1000, act="tanh")
            metric = fluid.metrics.Auc()
            for data in train_reader():
                loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
                metric.update(preds, labels)
                numpy_auc = metric.eval()
    """

    def __init__(self, name, curve='ROC', num_thresholds=200):
        super(Auc, self).__init__(name=name)
        self._curve = curve
        self._num_thresholds = num_thresholds
        self._epsilon = 1e-6
        self.tp_list = np.zeros((num_thresholds, ))
        self.fn_list = np.zeros((num_thresholds, ))
        self.tn_list = np.zeros((num_thresholds, ))
        self.fp_list = np.zeros((num_thresholds, ))

    def update(self, preds, labels):
        if not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray.")
        if not _is_numpy_(preds):
            raise ValueError("The 'predictions' must be a numpy ndarray.")

        kepsilon = 1e-7  # to account for floating point imprecisions
        thresholds = [(i + 1) * 1.0 / (self._num_thresholds - 1)
                      for i in range(self._num_thresholds - 2)]
        thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

        # calculate TP, FN, TN, FP count
        for idx_thresh, thresh in enumerate(thresholds):
            tp, fn, tn, fp = 0, 0, 0, 0
            for i, lbl in enumerate(labels):
                if lbl:
                    if preds[i, 1] >= thresh:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if preds[i, 1] >= thresh:
                        fp += 1
                    else:
                        tn += 1
            self.tp_list[idx_thresh] += tp
            self.fn_list[idx_thresh] += fn
            self.tn_list[idx_thresh] += tn
            self.fp_list[idx_thresh] += fp

    def eval(self):
        epsilon = self._epsilon
        num_thresholds = self._num_thresholds
        tpr = (self.tp_list.astype("float32") + epsilon) / (
            self.tp_list + self.fn_list + epsilon)
        fpr = self.fp_list.astype("float32") / (
            self.fp_list + self.tn_list + epsilon)
        rec = (self.tp_list.astype("float32") + epsilon) / (
            self.tp_list + self.fp_list + epsilon)

        x = fpr[:num_thresholds - 1] - fpr[1:]
        y = (tpr[:num_thresholds - 1] + tpr[1:]) / 2.0
        auc_value = np.sum(x * y)
        return auc_value
