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

import sys
import ssl
import requests


def download_file():
    """Get disabled unit tests"""
    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://sys-p0.bj.bcebos.com/prec/{}".format('disable_ut')
    f = requests.get(url)
    data = f.text
    status_code = f.status_code
    if len(data.strip()) == 0 or status_code != 200:
        sys.exit(1)
    else:
        lt = data.strip().split('\n')
        lt = '^' + '$|^'.join(lt) + '$'
        print(lt)
        sys.exit(0)


if __name__ == '__main__':
    try:
        download_file()
    except Exception as e:
        print(e)
        sys.exit(1)
