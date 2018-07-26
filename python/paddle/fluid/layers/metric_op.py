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
All layers just related to metric.
"""

import warnings
from ..layer_helper import LayerHelper
from ..initializer import Normal, Constant
from ..framework import Variable
from ..param_attr import ParamAttr
from . import nn

__all__ = ['accuracy', 'auc']


def accuracy(input, label, k=1, correct=None, total=None):
    """
    accuracy layer.
    Refer to the https://en.wikipedia.org/wiki/Precision_and_recall

    This function computes the accuracy using the input and label.
    If the correct label occurs in top k predictions, then correct will increment by one.
    Note: the dtype of accuracy is determined by input. the input and label dtype can be different.

    Args:
        input(Variable): The input of accuracy layer, which is the predictions of network.
          Carry LoD information is supported.
        label(Variable): The label of dataset.
        k(int): The top k predictions for each class will be checked.
        correct(Variable): The correct predictions count.
        total(Variable): The total entries count.

    Returns:
        Variable: The correct rate.

    Examples:
        .. code-block:: python

           data = fluid.layers.data(name="data", shape=[-1, 32, 32], dtype="float32")
           label = fluid.layers.data(name="data", shape=[-1,1], dtype="int32")
           predict = fluid.layers.fc(input=data, size=10)
           acc = fluid.layers.accuracy(input=predict, label=label, k=5)

    """
    helper = LayerHelper("accuracy", **locals())
    topk_out, topk_indices = nn.topk(input, k=k)
    acc_out = helper.create_tmp_variable(dtype="float32")
    if correct is None:
        correct = helper.create_tmp_variable(dtype="int64")
    if total is None:
        total = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="accuracy",
        inputs={
            "Out": [topk_out],
            "Indices": [topk_indices],
            "Label": [label]
        },
        outputs={
            "Accuracy": [acc_out],
            "Correct": [correct],
            "Total": [total],
        })
    return acc_out


def auc(input, label, curve='ROC', num_thresholds=200, topk=1):
    """
    **Area Under the Curve (AUC) Layer**

    This implementation computes the AUC according to forward output and label.
    It is used very widely in binary classification evaluation. 

    Note: If input label contains values other than 0 and 1, it will be cast 
    to `bool`. Find the relevant definitions `here <https://en.wikipedia.org\
    /wiki/Receiver_operating_characteristic#Area_under_the_curve>`_.

    There are two types of possible curves:

        1. ROC: Receiver operating characteristic;
        2. PR: Precision Recall

    Args:
        input(Variable): A floating-point 2D Variable, values are in the range 
                         [0, 1]. Each row is sorted in descending order. This 
                         input should be the output of topk. Typically, this 
                         Variable indicates the probability of each label.
        label(Variable): A 2D int Variable indicating the label of the training 
                         data. The height is batch size and width is always 1.
        curve(str): Curve type, can be 'ROC' or 'PR'. Default 'ROC'.
        num_thresholds(int): The number of thresholds to use when discretizing 
                             the roc curve. Default 200.
        topk(int): only topk number of prediction output will be used for auc.

    Returns:
        Variable: A scalar representing the current AUC.

    Examples:
        .. code-block:: python
        
            # network is a binary classification model and label the ground truth
            prediction = network(image, is_infer=True)
            auc_out=fluid.layers.auc(input=prediction, label=label)
    """
    helper = LayerHelper("auc", **locals())
    auc_out = helper.create_tmp_variable(dtype="float64")
    # make tp, tn, fp, fn persistable, so that can accumulate all batches.
    tp = helper.create_global_variable(persistable=True, dtype='int64')
    tn = helper.create_global_variable(persistable=True, dtype='int64')
    fp = helper.create_global_variable(persistable=True, dtype='int64')
    fn = helper.create_global_variable(persistable=True, dtype='int64')
    for var in [tp, tn, fp, fn]:
        helper.set_variable_initializer(
            var, Constant(
                value=0.0, force_cpu=True))

    helper.append_op(
        type="auc",
        inputs={
            "Predict": [input],
            "Label": [label],
            "TP": [tp],
            "TN": [tn],
            "FP": [fp],
            "FN": [fn]
        },
        attrs={"curve": curve,
               "num_thresholds": num_thresholds},
        outputs={
            "AUC": [auc_out],
            "TPOut": [tp],
            "TNOut": [tn],
            "FPOut": [fp],
            "FNOut": [fn]
        })
    return auc_out, [tp, tn, fp, fn]
