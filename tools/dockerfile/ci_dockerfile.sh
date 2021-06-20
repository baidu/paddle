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

function make_ubuntu_dockerfile(){
  dockerfile_name="Dockerfile.cuda10_cudnn7_gcc82_ubuntu16"
  sed "s/<baseimg>/10.1-cudnn7-devel-ubuntu16.04/g" ./Dockerfile.ubuntu >${dockerfile_name}
  sed -i "s#liblzma-dev#liblzma-dev openmpi-bin openmpi-doc libopenmpi-dev#g" ${dockerfile_name} 
  dockerfile_line=$(wc -l ${dockerfile_name}|awk '{print $1}')
  sed -i "${dockerfile_line}i RUN wget --no-check-certificate -q https://paddle-edl.bj.bcebos.com/hadoop-2.7.7.tar.gz \&\& \
     tar -xzf  hadoop-2.7.7.tar.gz && mv hadoop-2.7.7 /usr/local/" ${dockerfile_name} 
  sed -i "${dockerfile_line}i RUN apt remove git -y \&\& apt install -y libcurl4-openssl-dev gettext \&\& wget -q https://paddle-ci.gz.bcebos.com/git-2.17.1.tar.gz \&\& \
    tar -xvf git-2.17.1.tar.gz \&\& \
    cd git-2.17.1 \&\& \
    ./configure --with-openssl --with-curl --prefix=/usr/local \&\& \
    make -j8 \&\& make install " ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN pip install wheel \&\& pip3 install PyGithub wheel \&\& pip3.7 install PyGithub " ${dockerfile_name}
  sed -i "s#<install_gcc>#WORKDIR /usr/bin \\
    COPY tools/dockerfile/build_scripts /build_scripts \\
    RUN bash /build_scripts/install_gcc.sh gcc82 \&\& rm -rf /build_scripts \\
    RUN cp gcc  gcc.bak \&\& cp g++  g++.bak \&\& rm gcc \&\& rm g++ \\
    RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/local/bin/gcc \\
    RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/local/bin/g++ \\
    RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/gcc \\
    RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/bin/g++ \\
    ENV PATH=/usr/local/gcc-8.2/bin:\$PATH #g" ${dockerfile_name}
  sed -i "s#bash /build_scripts/install_nccl2.sh#wget -q --no-proxy https://nccl2-deb.cdn.bcebos.com/nccl-repo-ubuntu1604-2.7.8-ga-cuda10.1_1-1_amd64.deb \\
    RUN dpkg -i nccl-repo-ubuntu1604-2.7.8-ga-cuda10.1_1-1_amd64.deb \\
    RUN apt update \&\& apt remove -y libnccl* --allow-change-held-packages \&\&  apt-get install -y libnccl2=2.7.8-1+cuda10.1 libnccl-dev=2.7.8-1+cuda10.1 pigz numactl --allow-change-held-packages #g" ${dockerfile_name}
}


function make_centos_dockerfile(){
  dockerfile_name="Dockerfile.cuda9_cudnn7_gcc48_py35_centos6"
  sed "s/<baseimg>/11.0-cudnn8-devel-centos7/g" Dockerfile.centos >${dockerfile_name}
  sed -i "s#COPY build_scripts /build_scripts#COPY tools/dockerfile/build_scripts  ./build_scripts#g" ${dockerfile_name} 
  dockerfile_line=$(wc -l ${dockerfile_name}|awk '{print $1}')
  sed -i "${dockerfile_line}i RUN yum install -y pigz graphviz numactl" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN rm -f /usr/bin/cc && ln -s /usr/local/gcc-5.4/bin/gcc /usr/bin/cc" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN rm -f /usr/bin/g++ && ln -s /usr/local/gcc-5.4/bin/g++ /usr/bin/g++" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN rm -f /usr/bin/c++ && ln -s /usr/local/gcc-5.4/bin/c++ /usr/bin/c++" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN rm -f /usr/bin/gcc && ln -s /usr/local/gcc-5.4/bin/gcc /usr/bin/gcc" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN ln -s /usr/lib64/libz.so /usr/local/lib/libz.so \\
    RUN ln -s /usr/local/lib/libnccl.so /usr/local/cuda/lib64/" ${dockerfile_name}
  sed -i $"${dockerfile_line}i RUN wget --no-check-certificate -q  https://paddle-edl.bj.bcebos.com/hadoop-2.7.7.tar.gz \\
    RUN tar -xzf hadoop-2.7.7.tar.gz && mv hadoop-2.7.7 /usr/local/" ${dockerfile_name}
  sed -i "s#RUN bash build_scripts/build.sh#RUN bash build_scripts/install_gcc.sh gcc54 \nRUN mv /usr/bin/cc /usr/bin/cc.bak \&\& ln -s /usr/local/gcc-5.4/bin/gcc /usr/bin/cc \nENV PATH=/usr/local/gcc-5.4/bin:\$PATH \nRUN bash build_scripts/build.sh#g" ${dockerfile_name}
}


function main() {
  make_ubuntu_dockerfile
  make_centos_dockerfile
}

main "$@"
