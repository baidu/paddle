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

# 1. download the mobilenetv1 model to test config and predictor
if [ ! -d mobilenetv1 ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/mobilenetv1.tgz
    tar xzf mobilenetv1.tgz 
fi

# 2. set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$PWD/paddle_inference_c/paddle/lib

# 3. go test
go test -v ./...
