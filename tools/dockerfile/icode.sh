#!/bin/bash


function install_gcc(){
  sed -i 's#<install_gcc>#RUN apt-get update \
    WORKDIR /usr/bin \
    RUN apt install -y gcc-4.8 g++-4.8 \&\& cp gcc gcc.bak \&\& cp g++ g++.bak \&\& rm gcc \&\& rm g++ \&\& ln -s gcc-4.8 gcc \&\& ln -s g++-4.8 g++ #g' $1
}


function install_gcc8(){
  sed -i 's#<install_gcc>#WORKDIR /usr/bin \
    COPY tools/dockerfile/build_scripts /build_scripts \
    RUN bash /build_scripts/install_gcc.sh gcc82 \&\& rm -rf /build_scripts \
    RUN cp gcc gcc.bak \&\& cp g++ g++.bak \&\& rm gcc \&\& rm g++ \
    RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/local/bin/gcc \
    RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/local/bin/g++ \
    RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/gcc \
    RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/bin/g++ \
    ENV PATH=/usr/local/gcc-8.2/bin:$PATH #g' $1
}


function centos_gcc8(){
  sed -i "s#COPY build_scripts /build_scripts#COPY build_scripts /build_scripts \nRUN bash build_scripts/install_gcc.sh gcc82 \nENV PATH=/usr/local/gcc-8.2/bin:\$PATH#g" $1
}


function centos() {
  # centos6
  sed 's#<baseimg>#8.0-cudnn7-devel-centos6#g'  Dockerfile.centos >test/centos_6_cpu_runtime.dockerfile 
  sed 's#<baseimg>#9.0-cudnn7-devel-centos6#g'  Dockerfile.centos >test/centos_6_gpu_cuda9.0_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#9.1-cudnn7-devel-centos6#g'  Dockerfile.centos >test/centos_6_gpu_cuda9.1_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#9.2-cudnn7-devel-centos6#g'  Dockerfile.centos >test/centos_6_gpu_cuda9.2_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#10.0-cudnn7-devel-centos6#g' Dockerfile.centos >test/centos_6_gpu_cuda10.0_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#10.1-cudnn7-devel-centos6#g' Dockerfile.centos >test/centos_6_gpu_cuda10.1_cudnn7_single_gpu_runtime.dockerfile
  centos_gcc8 "test/centos_6_gpu_cuda10.1_cudnn7_single_gpu_runtime.dockerfile"
  
  # centos7
  sed 's#<baseimg>#8.0-cudnn7-devel-centos7#g'  Dockerfile.centos >test/centos_7_cpu_runtime.dockerfile 
  sed 's#<baseimg>#9.0-cudnn7-devel-centos7#g'  Dockerfile.centos >test/centos_7_gpu_cuda9.0_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#9.1-cudnn7-devel-centos7#g'  Dockerfile.centos >test/centos_7_gpu_cuda9.1_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#9.2-cudnn7-devel-centos7#g'  Dockerfile.centos >test/centos_7_gpu_cuda9.2_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#10.0-cudnn7-devel-centos7#g' Dockerfile.centos >test/centos_7_gpu_cuda10.0_cudnn7_single_gpu_runtime.dockerfile
  sed 's#<baseimg>#10.1-cudnn7-devel-centos7#g' Dockerfile.centos >test/centos_7_gpu_cuda10.1_cudnn7_single_gpu_runtime.dockerfile
  centos_gcc8 "test/centos_7_gpu_cuda10.1_cudnn7_single_gpu_runtime.dockerfile"
}


function ubuntu() {
  # ubuntu 14
  sed 's#<baseimg>#8.0-cudnn7-devel-ubuntu14.04#g'  Dockerfile.ubuntu >test/ubuntu_1404_cpu.dockerfile
  install_gcc "test/ubuntu_1404_cpu.dockerfile"
  sed 's#<baseimg>#9.0-cudnn7-devel-ubuntu14.04#g'  Dockerfile.ubuntu >test/ubuntu_1404_gpu_cuda9.0_cudnn7_runtime.dockerfile
  install_gcc "test/ubuntu_1404_gpu_cuda9.0_cudnn7_runtime.dockerfile"
  sed 's#<baseimg>#9.1-cudnn7-devel-ubuntu14.04#g'  Dockerfile.ubuntu >test/ubuntu_1404_gpu_cuda9.1_cudnn7_runtime.dockerfile
  install_gcc "test/ubuntu_1404_gpu_cuda9.1_cudnn7_runtime.dockerfile"
  sed 's#<baseimg>#9.2-cudnn7-devel-ubuntu14.04#g'  Dockerfile.ubuntu >test/ubuntu_1404_gpu_cuda9.2_cudnn7_runtime.dockerfile
  install_gcc "test/ubuntu_1404_gpu_cuda9.2_cudnn7_runtime.dockerfile"
  sed 's#<baseimg>#10.0-cudnn7-devel-ubuntu14.04#g' Dockerfile.ubuntu >test/ubuntu_1404_gpu_cuda10.0_cudnn7_runtime.dockerfile
  install_gcc "test/ubuntu_1404_gpu_cuda10.0_cudnn7_runtime.dockerfile"
  sed 's#<baseimg>#10.1-cudnn7-devel-ubuntu14.04#g' Dockerfile.ubuntu >test/ubuntu_1404_gpu_cuda10.1_cudnn7_runtime.dockerfile
  install_gcc8 "test/ubuntu_1404_gpu_cuda10.1_cudnn7_runtime.dockerfile"
 
  # ubuntu 16
  sed 's#<baseimg>#8.0-cudnn7-devel-ubuntu16.04#g'  Dockerfile.ubuntu >test/ubuntu_1604_cpu.dockerfile
  install_gcc "test/ubuntu_1604_cpu.dockerfile"
  sed 's#<baseimg>#9.0-cudnn7-devel-ubuntu16.04#g'  Dockerfile.ubuntu >test/ubuntu_1604_gpu_cuda9.0_cudnn7_runtime.dockerfile
  install_gcc "test/ubuntu_1604_gpu_cuda9.0_cudnn7_runtime.dockerfile"
  sed 's#<baseimg>#9.1-cudnn7-devel-ubuntu16.04#g'  Dockerfile.ubuntu >test/ubuntu_1604_gpu_cuda9.1_cudnn7_runtime.dockerfile
  install_gcc "test/ubuntu_1604_gpu_cuda9.1_cudnn7_runtime.dockerfile"
  sed 's#<baseimg>#9.2-cudnn7-devel-ubuntu16.04#g'  Dockerfile.ubuntu >test/ubuntu_1604_gpu_cuda9.2_cudnn7_runtime.dockerfile
  install_gcc "test/ubuntu_1604_gpu_cuda9.2_cudnn7_runtime.dockerfile"
  sed 's#<baseimg>#10.0-cudnn7-devel-ubuntu16.04#g' Dockerfile.ubuntu >test/ubuntu_1604_gpu_cuda10.0_cudnn7_runtime.dockerfile
  install_gcc "test/ubuntu_1604_gpu_cuda10.0_cudnn7_runtime.dockerfile"
  sed 's#<baseimg>#10.1-cudnn7-devel-ubuntu16.04#g' Dockerfile.ubuntu >test/ubuntu_1604_gpu_cuda10.1_cudnn7_runtime.dockerfile
  install_gcc8 "test/ubuntu_1604_gpu_cuda10.1_cudnn7_runtime.dockerfile"

  # ubuntu 18
  sed 's#<baseimg>#8.0-cudnn7-devel-ubuntu18.04#g'  Dockerfile.ubuntu >test/ubuntu_1804_cpu.dockerfile
  install_gcc "test/ubuntu_1804_cpu.dockerfile"
  sed 's#<baseimg>#9.0-cudnn7-devel-ubuntu18.04#g'  Dockerfile.ubuntu >test/ubuntu_1804_gpu_cuda9.0_cudnn7_runtime.dockerfile
  install_gcc "test/ubuntu_1804_gpu_cuda9.0_cudnn7_runtime.dockerfile"
  sed 's#<baseimg>#9.1-cudnn7-devel-ubuntu18.04#g'  Dockerfile.ubuntu >test/ubuntu_1804_gpu_cuda9.1_cudnn7_runtime.dockerfile
  install_gcc "test/ubuntu_1804_gpu_cuda9.1_cudnn7_runtime.dockerfile"
  sed 's#<baseimg>#9.2-cudnn7-devel-ubuntu18.04#g'  Dockerfile.ubuntu >test/ubuntu_1804_gpu_cuda9.2_cudnn7_runtime.dockerfile
  install_gcc "test/ubuntu_1804_gpu_cuda9.2_cudnn7_runtime.dockerfile"
  sed 's#<baseimg>#10.0-cudnn7-devel-ubuntu18.04#g' Dockerfile.ubuntu >test/ubuntu_1804_gpu_cuda10.0_cudnn7_runtime.dockerfile
  install_gcc "test/ubuntu_1804_gpu_cuda10.0_cudnn7_runtime.dockerfile"
  sed 's#<baseimg>#10.1-cudnn7-devel-ubuntu18.04#g' Dockerfile.ubuntu >test/ubuntu_1804_gpu_cuda10.1_cudnn7_runtime.dockerfile
  install_gcc8 "test/ubuntu_1804_gpu_cuda10.1_cudnn7_runtime.dockerfile"
}


function main() {
  if [ ! -d "test" ];then
    mkdir test
  fi

  centos
  ubuntu
}


main
