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

import sys
import time
import os
import json

# event
for line in sys.stdin:
    if line.find("x_bytes_phy") > 0:
        format_str = "tx" if line.find("tx_bytes_ph") > 0 else "rx"
        di = {}
        di['name'] = format_str
        mbs = line[line.find('='):].split()[1]
        di['args'] = {"0": mbs}
        di['pid'] = 10 if format_str == "tx" else 11
        time_ms = int(time.time() * 1000)
        di['ts'] = time_ms
        di['cat'] = format_str
        di['tid'] = "0"
        di['ph'] = "C"
        print(json.dumps(di))
