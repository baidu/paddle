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

import paddle
import numpy as np


def convert_params_for_cell(np_cell, paddle_cell):
    state = np_cell.parameters
    for k, v in paddle_cell.named_parameters():
        v.set_value(state[k])


def convert_params_for_cell_static(np_cell, paddle_cell, place):
    state = np_cell.parameters
    for k, v in paddle_cell.named_parameters():
        scope = paddle.static.global_scope()
        tensor = scope.find_var(v.name).get_tensor()
        tensor.set(state[k], place)


def convert_params_for_net(np_net, paddle_net):
    for np_layer, paddle_layer in zip(np_net, paddle_net):
        if hasattr(np_layer, "cell"):
            convert_params_for_cell(np_layer.cell, paddle_layer.cell)
        else:
            convert_params_for_cell(np_layer.cell_fw, paddle_layer.cell_fw)
            convert_params_for_cell(np_layer.cell_bw, paddle_layer.cell_bw)


def convert_params_for_net_static(np_net, paddle_net, place):
    for np_layer, paddle_layer in zip(np_net, paddle_net):
        if hasattr(np_layer, "cell"):
            convert_params_for_cell_static(np_layer.cell, paddle_layer.cell,
                                           place)
        else:
            convert_params_for_cell_static(np_layer.cell_fw,
                                           paddle_layer.cell_fw, place)
            convert_params_for_cell_static(np_layer.cell_bw,
                                           paddle_layer.cell_bw, place)


def get_params_for_cell(np_cell, num_layers, idx):
    state = np_cell.parameters
    weight_list = [
        ('{}.weight_{}'.format(num_layers, idx), state['weight_ih']),
        ('{}.weight_{}'.format(num_layers, idx + 1), state['weight_hh'])
    ]
    bias_list = [('{}.bias_{}'.format(num_layers, idx), state['bias_ih']),
                 ('{}.bias_{}'.format(num_layers, idx + 1), state['bias_hh'])]
    return weight_list, bias_list


def get_params_for_net(np_net):
    weight_list = []
    bias_list = []
    for layer_idx, np_layer in enumerate(np_net):
        if hasattr(np_layer, "cell"):
            weight, bias = get_params_for_cell(np_layer.cell, layer_idx, 0)
            for w, b in zip(weight, bias):
                weight_list.append(w)
                bias_list.append(b)
        else:
            for count, cell in enumerate([np_layer.cell_fw, np_layer.cell_bw]):
                weight, bias = get_params_for_cell(cell, layer_idx, count * 2)
                for w, b in zip(weight, bias):
                    weight_list.append(w)
                    bias_list.append(b)

    weight_list.extend(bias_list)
    return weight_list
