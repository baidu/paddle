# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import Variable, in_dygraph_mode, _dygraph_tracer, default_main_program
from paddle.fluid import core
from paddle.fluid.data_feeder import check_variable_and_dtype


def dropout_bias_fuse(x,
                      bias,
                      dropout_prob,
                      is_test=None,
                      seed=None,
                      name=None,
                      dropout_implementation="downgrade_in_infer"):
    if dropout_prob == 0:
        return x + bias

    def get_attrs(prog, dropout_prob, is_test, seed):
        if (seed is None or seed == 0) and prog.random_seed != 0:
            seed = prog.random_seed
        attrs = {
            'dropout_prob': dropout_prob,
            'is_test': is_test,
            'fix_seed': seed is not None,
            'seed': seed if seed is not None else 0,
            'dropout_implementation': dropout_implementation,
        }
        return attrs

    if in_dygraph_mode():
        if (seed is None or seed == 0) and \
                default_main_program().random_seed != 0:
            seed = default_main_program().random_seed
        if is_test is None:
            is_test = not _dygraph_tracer()._train_mode
        out, mask = core.ops.fused_dropout_bias(
            x, bias, 'dropout_prob', dropout_prob, 'is_test', is_test,
            'fix_seed', seed is not None, 'seed', seed if seed is not None else
            0, 'dropout_implementation', dropout_implementation)
        return out

    helper = LayerHelper('fused_dropout_bias', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    mask = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.UINT8, stop_gradient=True)
    attrs = get_attrs(helper.main_program, dropout_prob, is_test, seed)
    helper.append_op(
        type='fused_dropout_bias',
        inputs={'X': [x],
                'Bias': [bias]},
        outputs={'Out': [out],
                 'Mask': [mask]},
        attrs=attrs)
    return out
