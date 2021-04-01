#!/bin/bash -ex
mkdir -p build
cd build
export https_proxy=http://172.19.57.45:3128
export http_proxy=http://172.19.57.45:3128
cmake ..  -DWITH_DISTRIBUTE=ON -DWITH_GRPC=OFF -DWITH_BRPC=OFF -DWITH_GPU=ON -DWITH_FAST_BUNDLE_TEST=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -DWITH_PROFILER=OFF -DPY_VERSION=2.7 -DWITH_FLUID_ONLY=ON  -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=ON -DWITH_MKLDNN=OFF
make -j$(nproc)
unset https_proxy
unset http_proxy
