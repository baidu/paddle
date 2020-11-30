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

from paddle.fluid.data_feeder import check_variable_and_dtype, check_type
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import Variable

__all__ = ['check_finite_and_unscale', 'update_loss_scaling']


def check_finite_and_unscale(x, scale, name=None):
    """
    Check if input X contains all finite data, if yes, scale it by input Scale.

    $$Out = X / scale$$

    If any tensor in X contains Inf or Nan, the Out will generate a indicator.
    FoundInfinite will be 1 (True), and Out will not be scaled. In this case, the data of 
    Out should not be used, and its data may not be deterministic. 
    Otherwise, FoundInfinite will be 0 (False).
    Args:
        x(list|tuple): The input tensors of check_finite_and_unscale operator.
        scale: The scale of check_finite_and_unscale operator.
    """
    check_type(x, 'x', (tuple, list), 'check_finite_and_unscale')
    for e in x:
        check_variable_and_dtype(e, "x", ['float32', 'float64'],
                                 'check_finite_and_unscale')

    helper = LayerHelper("check_finite_and_unscale", **locals())
    found_inf = helper.create_variable_for_type_inference(dtype='bool')

    inputs = {'X': x, 'Scale': scale}
    outputs = {'Out': x, 'FoundInfinite': found_inf}
    helper.append_op(
        type='check_finite_and_unscale', inputs=inputs, outputs=outputs)

    return x, found_inf


def update_loss_scaling(found_inf,
                        prev_loss_scaling,
                        num_good_steps,
                        num_bad_steps,
                        incr_every_n_steps,
                        decr_every_n_nan_or_inf,
                        incr_ratio,
                        decr_ratio,
                        name=None):
    """
    Update loss scaling according to overall gradients. If all gradients is 
    finite after incr_every_n_steps, loss scaling will increase by incr_ratio. 
    Otherwise, loss scaling will decrease by decr_ratio after
    decr_every_n_nan_or_inf steps and each step some gradients are infinite.

    Args:
        found_inf (Variable): A boolean variable indicates whether
                                     there is any infinite gradient.
        prev_loss_scaling (Variable): Previous loss scaling.
        num_good_steps (Variable): A variable accumulates good steps in which 
                                   all gradients are finite.
        num_bad_steps (Variable): A variable accumulates bad steps in which 
                                  some gradients are infinite.
        incr_every_n_steps (int): A variable represents increasing loss 
                                       scaling every n consecutive steps with 
                                       finite gradients.
        decr_every_n_nan_or_inf (int): A variable represents decreasing 
                                            loss scaling every n accumulated 
                                            steps with nan or inf gradients.
        incr_ratio(float): The multiplier to use when increasing the loss 
                           scaling.
        decr_ratio(float): The less-than-one-multiplier to use when decreasing 
                           loss scaling.
    """

    check_variable_and_dtype(prev_loss_scaling, "prev_loss_scaling",
                             ['float32', 'float64'], "update_loss_scaling")

    helper = LayerHelper("update_loss_scaling", **locals())

    inputs = {
        'FoundInfinite': found_inf,
        'PrevLossScaling': prev_loss_scaling,
        'InGoodSteps': num_good_steps,
        'InBadSteps': num_bad_steps
    }

    outputs = {
        'LossScaling': prev_loss_scaling,
        'OutGoodSteps': num_good_steps,
        'OutBadSteps': num_bad_steps
    }

    attrs = {
        'incr_every_n_steps': incr_every_n_steps,
        'decr_every_n_nan_or_inf': decr_every_n_nan_or_inf,
        'incr_ratio': incr_ratio,
        'decr_ratio': decr_ratio,
    }

    helper.append_op(
        type='update_loss_scaling', inputs=inputs, outputs=outputs, attrs=attrs)

    return prev_loss_scaling
