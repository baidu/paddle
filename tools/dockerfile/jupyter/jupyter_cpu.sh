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

  
function ref_whl(){
  if [[ ${WITH_GPU} == "ON" ]]; then
      ref_gpu=gpu-cuda${ref_CUDA_MAJOR}-cudnn${CUDNN_MAJOR}
      install_gpu="_gpu"
  else
      ref_gpu="cpu-avx"
      install_gpu=""
  fi
  
  if [[ ${WITH_MKL} == "ON" ]]; then
      ref_mkl=mkl
  else
      ref_mkl=openblas
  fi

  if [[ ${WITH_GPU} != "ON" ]]; then
    ref_gcc=""
  elif [[ ${gcc_version} == "8.2.0" ]];then
    ref_gcc=-gcc8.2
  fi

  if [[ ${ref_CUDA_MAJOR} == "11.0" ]];then
      ref_version=.post110
  elif [[ ${ref_CUDA_MAJOR} == "11.2" ]];then
      ref_version=.post112
  elif [[ ${ref_CUDA_MAJOR} == "10" ]];then
      ref_version=.post100
  elif [[ ${ref_CUDA_MAJOR} == "10.1" ]];then
      ref_version=.post101
  elif [[ ${ref_CUDA_MAJOR} == "10.2" && ${PADDLE_VERSION} == "develop" ]];then
      ref_version=.post102
  elif [[ ${ref_CUDA_MAJOR} == "10.2" && ${PADDLE_VERSION} != "develop" ]];then
      ref_version=""
  elif [[ ${ref_CUDA_MAJOR} == "9" ]];then
      ref_version=.post90
  fi

  ref_dev=2.1.0.dev0
  
  ref_web="https:\\/\\/paddle-wheel.bj.bcebos.com\\/${PADDLE_BRANCH}-${ref_gpu}-${ref_mkl}${ref_gcc}"
  
  if [[ ${PADDLE_VERSION} == "develop" && ${WITH_GPU} == "ON" ]]; then
    ref_paddle38_whl=paddlepaddle${install_gpu}-${ref_dev}${ref_version}-cp38-cp38-linux_x86_64.whl
  elif [[ ${PADDLE_VERSION} == "develop" && ${WITH_GPU} != "ON" ]]; then
    ref_paddle38_whl=paddlepaddle${install_gpu}-${ref_dev}-cp38-cp38-linux_x86_64.whl
  elif [[ ${PADDLE_VERSION} != "develop" && ${WITH_GPU} == "ON" ]]; then
    ref_paddle38_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}${ref_version}-cp38-cp38-linux_x86_64.whl
  else
    ref_paddle38_whl=paddlepaddle${install_gpu}-${PADDLE_VERSION}-cp38-cp38-linux_x86_64.whl
  fi

  ref_whl_package=${ref_web}\\/${ref_paddle38_whl}
}


function make_dockerfile(){
  sed "s/<paddle_whl>/${ref_whl_package}/g" Dockerfile.jupyter_cpu >Dockerfile.tmp
}


function main(){
  ref_whl
  make_dockerfile
}

main $@
