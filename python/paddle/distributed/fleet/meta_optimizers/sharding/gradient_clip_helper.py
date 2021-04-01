# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole


class GradientClipHelper(object):
    def __init__(self, mp_ring_id):
        self.mp_ring_id = mp_ring_id

    def _is_gradient_clip_op(self, op):
        return op.desc.has_attr("op_namescope") \
            and op.desc.attr("op_namescope").startswith("/gradient_clip")

    def prune_gradient_clip(self, block, shard):
        """
        prune gradient_clip related ops for params that not belong to cur shard
        prune: square, reduce_sum, elementwise_mul
        keep: sum, sqrt, elementwise_max, elementwise_div
        """
        deperated_vars = set()
        deperate_op_idx = set()
        reversed_x_paramname = []
        for idx, op in enumerate(block.ops):
            if not self._is_gradient_clip_op(op):
                continue
            if op.type == "sum":
                continue
            deperate_op = False
            for input_name in op.desc.input_arg_names():
                if input_name in deperated_vars:
                    deperate_op = True
                param_name = input_name.strip("@GRAD")
                if shard.is_param(param_name) and \
                  not shard.has_param(param_name):
                    deperate_op = True
                elif shard.is_param(param_name) :
                    reversed_x_paramname.append(param_name)

            if deperate_op:
                deperate_op_idx.add(idx)
                for output_name in op.desc.output_arg_names():
                    deperated_vars.add(output_name)

        if not deperated_vars:
            # got no gradient_clip op
            return

        for idx, op in reversed(list(enumerate(block.ops))):
            if not self._is_gradient_clip_op(op):
                continue
            if idx in deperate_op_idx:
                block._remove_op(idx, sync=False)
                continue
            reversed_inputs = []
            if op.type == "sum":
                for input_name in op.desc.input_arg_names():
                    if input_name not in deperated_vars:
                        reversed_inputs.append(input_name)
                        
                op.desc.set_input("X", reversed_inputs)
                assert (len(op.desc.output_arg_names()) == 1)
                sum_res = op.desc.output_arg_names()[0]

                # this allreduce should not overlap with calc and should be scheduled in calc stream
                # block._insert_op_without_sync(
                #     idx + 1,
                #     type='c_sync_comm_stream',
                #     inputs={'X': sum_res},
                #     outputs={'Out': sum_res},
                #     attrs={'ring_id': 0,
                #            OP_ROLE_KEY: OpRole.Optimize})
                block._insert_op_without_sync(
                    idx + 1,
                    type='c_allreduce_sum',
                    inputs={'X': sum_res},
                    outputs={'Out': sum_res},
                    attrs={
                        'ring_id': self.mp_ring_id,
                        'op_namescope': "/gradient_clip_model_parallelism",
                        'use_calc_stream': True,
                        OP_ROLE_KEY: OpRole.Optimize,
                    })
                # block._insert_op_without_sync(
                #     idx + 1,
                #     type='c_sync_calc_stream',
                #     inputs={'X': sum_res},
                #     outputs={'Out': sum_res},
                #     attrs={OP_ROLE_KEY: OpRole.Optimize})

        # the grad sum here should take the all and only param in the current shard
        to_check_param = set(reversed_x_paramname)
        should_check_param = set(shard.global_params).intersection(set([param for param, worker_idx in shard.global_param2device.items() if worker_idx == shard.worker_idx]))
        assert to_check_param == should_check_param, "amp check_finite_and_unscale checking miss [{}] and got unexpected [{}]".format(should_check_param - to_check_param, to_check_param - should_check_param)
       
        for var_name in deperated_vars:
            block._remove_var(var_name, sync=False)
        block._sync_with_cpp()
        return
