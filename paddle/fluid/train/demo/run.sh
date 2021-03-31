#!/bin/bash

set -x

PADDLE_ROOT=/home/yiak/WorkSpace/Github/Paddle
TURN_ON_MKL=ON # use MKL or Openblas

# generate models program description
python demo_trainer.py

# build demo trainer
paddle_install_dir=${PADDLE_ROOT}/build

mkdir -p build
cd build
rm -rf *
cmake .. -DPADDLE_LIB=$paddle_install_dir \
         -DWITH_MKLDNN=$TURN_ON_MKL \
         -DWITH_MKL=$TURN_ON_MKL
make

cd ..

# run demo trainer
build/demo_trainer
