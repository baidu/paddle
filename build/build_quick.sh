#!/bin/bash
set -e

source /home/gongwb/env.sh

export PATH=/home/gongwb/.local/bin:/usr/local/gcc-8.2/bin:$PATH
export LD_LIBRARY_PATH=/home/gongwb/.local/lib:${LD_LIBRARY_PATH}

./build_ascend.sh

