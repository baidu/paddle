# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
Print all ops desc in dict:
    {op1_name:
        {INPUTS:
            {input_name1:
                {DISPENSABLE: bool,
                 INTERMEDIATE: bool,
                 DUPLICABLE: bool
                },
            input_name2:{}
            },
         OUTPUTS:{},
         ATTRS:
            {attr_name1:
                {TYPE: int,
                 GENERATED: bool,
                 DEFAULT_VALUE: int/str/etc,
                }
            }
        }
     op2_name:{}
    }

Usage:
    python print_op_desc.py > op_desc.spec
"""

import paddle.fluid.framework as framework
from paddle.fluid import core
import json
from paddle import compat as cpt

INPUTS = "Inputs"
OUTPUTS = "Outputs"
ATTRS = "Attrs"

DUPLICABLE = "duplicable"
INTERMEDIATE = "intermediate"
DISPENSABLE = "dispensable"

TYPE = "type"
GENERATED = "generated"
DEFAULT_VALUE = "default_value"


def get_attr_default_value(op_name):
    return core.get_op_attrs_default_value(cpt.to_bytes(op_name))


def get_vars_info(op_vars_proto):
    vars_info = {}
    for var_proto in op_vars_proto:
        name = str(var_proto.name)
        vars_info[name] = {}
        vars_info[name][DUPLICABLE] = var_proto.duplicable
        vars_info[name][DISPENSABLE] = var_proto.dispensable
        vars_info[name][INTERMEDIATE] = var_proto.intermediate
    return vars_info


def get_attrs_info(op_proto, op_attrs_proto):
    attrs_info = {}
    attrs_default_values = get_attr_default_value(op_proto.type)
    for attr_proto in op_attrs_proto:
        attr_name = str(attr_proto.name)
        attrs_info[attr_name] = {}
        attrs_info[attr_name][TYPE] = attr_proto.type
        attrs_info[attr_name][GENERATED] = attr_proto.generated
        attrs_info[attr_name][DEFAULT_VALUE] = attrs_default_values[
            attr_name] if attr_name in attrs_default_values else None
    return attrs_info


def get_op_desc(op_proto):
    op_info = {}
    op_info[INPUTS] = get_vars_info(op_proto.inputs)
    op_info[OUTPUTS] = get_vars_info(op_proto.outputs)
    op_info[ATTRS] = get_attrs_info(op_proto, op_proto.attrs)
    return op_info


def get_all_ops_desc():
    all_op_protos_dict = {}
    all_op_protos = framework.get_all_op_protos()
    for op_proto in all_op_protos:
        op_type = str(op_proto.type)
        all_op_protos_dict[op_type] = get_op_desc(op_proto)

    return all_op_protos_dict


all_op_protos_dict = get_all_ops_desc()
result = json.dumps(
    all_op_protos_dict,
    sort_keys=True,
    indent=2,
    separators=(', ', ': '), )
print(result)
