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
  ref_gpu=gpu-cuda${ref_CUDA_MAJOR}-cudnn${CUDNN_MAJOR}

  if [[ ${ref_CUDA_MAJOR} == "11.0" ]];then
      ref_version=.post110
      ref_dockerfile=cuda110
  elif [[ ${ref_CUDA_MAJOR} == "10.1" ]];then
      ref_version=.post101
      ref_dockerfile=cuda101
  fi

  ref_web="https://paddle-wheel.bj.bcebos.com/with-trt/${PADDLE_BRANCH}-${ref_gpu}-mkl-gcc8.2"
  ref_paddle37_whl=paddlepaddle_gpu-${PADDLE_VERSION}${ref_version}-cp37-cp37m-linux_x86_64.whl
}


function install_whl(){
  dockerfile_line=`wc -l Dockerfile.tmp|awk '{print $1}'`
  sed -i "${dockerfile_line}i RUN wget -q ${ref_web}/${ref_paddle37_whl} && pip3.7 install ${ref_paddle37_whl} && rm -f ${ref_paddle37_whl}" Dockerfile.tmp
}


function install_gcc(){
  if [ "${gcc_version}" == "8.2.0" ];then
    sed -i 's#<install_gcc>#WORKDIR /usr/bin \
      COPY tools/dockerfile/build_scripts /build_scripts \
      RUN bash /build_scripts/install_gcc.sh gcc82 \&\& rm -rf /build_scripts \
      RUN cp gcc gcc.bak \&\& cp g++ g++.bak \&\& rm gcc \&\& rm g++ \
      RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/local/bin/gcc \
      RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/local/bin/g++ \
      RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/gcc \
      RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/bin/g++ \
      ENV PATH=/usr/local/gcc-8.2/bin:$PATH #g' Dockerfile.tmp
  else
    sed -i 's#<install_gcc>#RUN apt-get update \
      WORKDIR /usr/bin \
      RUN apt install -y gcc g++ #g' Dockerfile.tmp
  fi
}

function make_dockerfile(){
  cp Dockerfile.${ref_dockerfile} Dockerfile.tmp
}


function main(){
  ref_whl
  make_dockerfile
  # install_gcc
  install_whl
}

main $@
