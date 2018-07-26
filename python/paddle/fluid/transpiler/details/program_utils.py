# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


def delete_ops(block, ops):
    try:
        start = list(block.ops).index(ops[0])
        end = list(block.ops).index(ops[-1])
        [block._remove_op(start) for _ in range(end - start + 1)]
    except Exception as e:
        raise e
    block.program._sync_with_cpp()


def find_op_by_input_arg(block, arg_name):
    for index, op in enumerate(block.ops):
        if arg_name in op.input_arg_names:
            return index
    return -1


def find_op_by_output_arg(block, arg_name):
    for index, op in enumerate(block.ops):
        if arg_name in op.output_arg_names:
            return index
    return -1
