# -*- coding:UTF-8 -*-

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""generate hccl config file script"""
import os
import sys
import json
import socket
from argparse import ArgumentParser
from typing import Dict, Any


def parse_args():
    """
    parse args .

    Args:

    Returns:
        args.

    Examples:
        >>> parse_args()
    """
    parser = ArgumentParser(description="mindspore distributed training launch "
                            "helper utilty that will generate hccl"
                            " config file")
    parser.add_argument(
        "--device_num",
        type=str,
        default="[0,8)",
        help="The number of the Ascend accelerators used. please note that the Ascend accelerators"
        "used must be continuous, such [0,4) means to use four chips "
        "0，1，2，3; [0,1) means to use chip 0; The first four chips are"
        "a group, and the last four chips are a group. In addition to"
        "the [0,8) chips are allowed, other cross-group such as [3,6)"
        "are prohibited.")
    parser.add_argument(
        "--visible_devices",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="will use the visible devices sequentially")
    parser.add_argument("--server_ip", type=str, default="", help="server ip")
    args = parser.parse_args()
    return args


def get_host_ip():
    """
    get host ip
    """
    ip = None

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
    except EOFError:
        pass

    return ip


def main():
    print("start", __file__)
    args = parse_args()

    # visible_devices
    visible_devices = args.visible_devices.split(',')
    print('visible_devices:{}'.format(visible_devices))

    # server_id
    ip = get_host_ip()
    if args.server_ip:
        server_id = args.server_ip
    elif ip:
        server_id = ip
    else:
        raise ValueError("please input server ip!")
    print('server_id:{}'.format(server_id))

    # device_num
    first_num = int(args.device_num[1])
    last_num = int(args.device_num[3])
    if first_num < 0 or last_num > 8:
        raise ValueError("device num {} must be in range [0,8] !".format(
            args.device_num))
    if first_num > last_num:
        raise ValueError(
            "First num {} of device num {} must less than last num {} !".format(
                first_num, args.device_num, last_num))
    if first_num < 4:
        if last_num > 4:
            if first_num == 0 and last_num == 8:
                pass
            else:
                raise ValueError(
                    "device num {} must be in the same group of [0,4] or [4,8] !".
                    format(args.device_num))

    device_num_list = list(range(first_num, last_num))
    print("device_num_list:", device_num_list)

    assert len(visible_devices) >= len(device_num_list)

    # construct hccn_table
    device_ips = {}
    with open('/etc/hccn.conf', 'r') as fin:
        for hccn_item in fin.readlines():
            if hccn_item.strip().startswith('address_'):
                device_id, device_ip = hccn_item.split('=')
                device_id = device_id.split('_')[1]
                device_ips[device_id] = device_ip.strip()

    hccn_table = {'version': '1.0', 'server_count': '1', 'server_list': []}
    device_list = []
    rank_id = 0
    for instance_id in device_num_list:
        device_id = visible_devices[instance_id]
        device_ip = device_ips[device_id]
        device = {
            'device_id': device_id,
            'device_ip': device_ip,
            'rank_id': str(rank_id)
        }
        print('rank_id:{}, device_id:{}, device_ip:{}'.format(
            rank_id, device_id, device_ip))
        rank_id += 1
        device_list.append(device)
    hccn_table['server_list'].append({
        'server_id': server_id,
        'device': device_list,
        'host_nic_ip': 'reserve'
    })
    hccn_table['status'] = 'completed'

    # save hccn_table to file
    table_path = os.getcwd()
    table_fn = os.path.join(table_path, 'hccl_{}p_{}_{}.json'.format(
        len(device_num_list), "".join(map(str, device_num_list)), server_id))
    with open(table_fn, 'w') as table_fp:
        json.dump(hccn_table, table_fp, indent=4)
    sys.stdout.flush()
    print("Completed: hccl file was save in :", table_fn)


if __name__ == "__main__":
    main()
