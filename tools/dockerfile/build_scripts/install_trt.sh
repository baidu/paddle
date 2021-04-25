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

VERSION=$(nvcc --version | grep release | grep -oEi "release ([0-9]+)\.([0-9])"| sed "s/release //")

if [[ "$VERSION" == "10.1" ]];then
  wget -q https://paddle-ci.gz.bcebos.com/TRT/TensorRT6-cuda10.1-cudnn7.tar.gz --no-check-certificate
  tar -zxf TensorRT6-cuda10.1-cudnn7.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT6-cuda10.1-cudnn7/include/* /usr/include/ && cp -rf /usr/local/TensorRT6-cuda10.1-cudnn7/lib/* /usr/lib/
  rm TensorRT6-cuda10.1-cudnn7.tar.gz
elif [[ "$VERSION" == "11.2" ]];then
  wget -q https://paddle-ci.gz.bcebos.com/TRT/TensorRT7-cuda11.1-cudnn8.1.tar.gz --no-check-certificate
  tar -zxf TensorRT7-cuda11.1-cudnn8.1.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT-7.2.3.4/include/* /usr/include/ && cp -rf /usr/local/TensorRT-7.2.3.4/lib/* /usr/lib/
  rm TensorRT7-cuda11.1-cudnn8.1.tar.gz
elif [[ "$VERSION" == "11.0" ]];then
  wget -q https://paddle-ci.cdn.bcebos.com/TRT/TensorRT-7.1.3.4.Ubuntu-16.04.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz --no-check-certificate
  tar -zxf TensorRT-7.1.3.4.Ubuntu-16.04.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT-7.1.3.4/include/* /usr/include/ && cp -rf /usr/local/TensorRT-7.1.3.4/lib/* /usr/lib/
  rm TensorRT-7.1.3.4.Ubuntu-16.04.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz
elif [[ "$VERSION" == "10.2" ]];then
  wget https://paddle-ci.gz.bcebos.com/TRT/TensorRT7-cuda10.2-cudnn8.tar.gz --no-check-certificate
  tar -zxf TensorRT7-cuda10.2-cudnn8.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT-7.1.3.4/include/* /usr/include/ && cp -rf /usr/local/TensorRT-7.1.3.4/lib/* /usr/lib/
  rm TensorRT7-cuda10.2-cudnn8.tar.gz
elif [[ "$VERSION" == "10.0" ]];then
  wget -q https://paddle-ci.gz.bcebos.com/TRT/TensorRT6-cuda10.0-cudnn7.tar.gz --no-check-certificate
  tar -zxf TensorRT6-cuda10.0-cudnn7.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT6-cuda10.0-cudnn7/include/* /usr/include/ && cp -rf /usr/local/TensorRT6-cuda10.0-cudnn7/lib/* /usr/lib/
  rm TensorRT6-cuda10.0-cudnn7.tar.gz
elif [[ "$VERSION" == "9.0" ]];then
  wget -q https://paddle-ci.gz.bcebos.com/TRT/TensorRT6-cuda9.0-cudnn7.tar.gz --no-check-certificate
  tar -zxf TensorRT6-cuda9.0-cudnn7.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT6-cuda9.0-cudnn7/include/* /usr/include/ && cp -rf /usr/local/TensorRT6-cuda9.0-cudnn7/lib/* /usr/lib/
  rm TensorRT6-cuda9.0-cudnn7.tar.gz
fi
