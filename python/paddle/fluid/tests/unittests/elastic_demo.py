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

import os, sys
import time

sys.stderr.write("{}-DISTRIBUTED_TRAINER_ENDPOINTS={}\n".format(os.environ[
    'PADDLE_TRAINER_ID'], os.environ['DISTRIBUTED_TRAINER_ENDPOINTS']))
sys.stderr.write("{}-PADDLE_TRAINERS={}\n".format(os.environ[
    'PADDLE_TRAINER_ID'], os.environ['PADDLE_TRAINERS']))

time.sleep(600)
