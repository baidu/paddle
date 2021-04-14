#!/bin/bash

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

set -e

curr_host_ip=`hostname -i`
python hccl_tools.py --device_num "[0,4)" --server_ip ${curr_host_ip}

export RANK_TABLE_FILE="${PWD}/hccl_4p_0123_${curr_host_ip}.json"

# use ascend
echo "begin test use ascend npu"

distributed_args="--run_mode=collective --log_dir=testlog"
python -m paddle.distributed.fleet.launch ${distributed_args} \
  ascend_group.py fleetascendgroup
