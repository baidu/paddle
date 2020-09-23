#!/bin/bash
function make_ubuntu_dockerfile(){
  dockerfile_name="Dockerfile.cuda10_cudnn7_gcc82_ubuntu16"
  sed 's/<baseimg>/10.1-cudnn7-devel-ubuntu16.04/g' ./Dockerfile.ubuntu >${dockerfile_name}
  sed -i 's#liblzma-dev#liblzma-dev openmpi-bin openmpi-doc libopenmpi-dev#g' ${dockerfile_name} 
  dockerfile_line=`wc -l ${dockerfile_name}|awk '{print $1}'`
  sed -i "${dockerfile_line}i RUN wget --no-check-certificate  -q https://paddle-edl.bj.bcebos.com/hadoop-2.7.7.tar.gz && \
     tar -xzf  hadoop-2.7.7.tar.gz && mv hadoop-2.7.7 /usr/local/" ${dockerfile_name} 
  sed -i 's#<install_gcc>#WORKDIR /usr/bin \
      COPY tools/dockerfile/build_scripts /build_scripts \
      RUN bash /build_scripts/install_gcc.sh gcc82 \&\& rm -rf /build_scripts \
      RUN cp gcc gcc.bak \&\& cp g++ g++.bak \&\& rm gcc \&\& rm g++ \
      RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/local/bin/gcc \
      RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/local/bin/g++ \
      RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/gcc \
      RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/bin/g++ \
      ENV PATH=/usr/local/gcc-8.2/bin:$PATH #g' ${dockerfile_name}

}


function make_centos_dockerfile(){
  dockerfile_name="Dockerfile.cuda10_cudnn7_gcc8_py35_centos6"
  sed 's/<baseimg>/9.0-cudnn7-devel-centos6/g' Dockerfile.centos >${dockerfile_name}
  sed -i 's#COPY build_scripts /build_scripts#COPY tools/dockerfile/build_scripts ./build_scripts#g' ${dockerfile_name} 
  dockerfile_line=`wc -l ${dockerfile_name}|awk '{print $1}'`
  sed -i "${dockerfile_line}i RUN ln -s /usr/lib64/libz.so /usr/local/lib/libz.so && \
     ln -s /usr/local/lib/libnccl.so /usr/local/cuda/lib64/ && \
     rm -rf /usr/include/NvInfer*" ${dockerfile_name}
  sed -i "${dockerfile_line}i RUN wget --no-check-certificate  -q https://paddle-edl.bj.bcebos.com/hadoop-2.7.7.tar.gz && \
     tar -xzf  hadoop-2.7.7.tar.gz && mv hadoop-2.7.7 /usr/local/" ${dockerfile_name}
}


function main() {
  make_ubuntu_dockerfile
  make_centos_dockerfile
}

main $@
