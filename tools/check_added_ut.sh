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

set +e
set -x
if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi

export CI_SKIP_CPP_TEST=OFF
PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
CURDIR=`pwd`
cd $PADDLE_ROOT
cp $PADDLE_ROOT/paddle/scripts/paddle_build.sh $PADDLE_ROOT/paddle/scripts/paddle_build_pre.sh
CURBRANCH=`git rev-parse --abbrev-ref HEAD`
echo $CURBRANCH
git checkout -b prec_added_ut upstream/${BRANCH}
mkdir prec_build
cd prec_build
bash $PADDLE_ROOT/paddle/scripts/paddle_build_pre.sh cmake_gen_in_current_dir >prebuild.log 2>&1
ctest -N | awk -F ':' '{print $2}' | sed '/^$/d' | sed '$d' | sed 's/ //g' > /$PADDLE_ROOT/br-ut
cd $PADDLE_ROOT/build
ctest -N | awk -F ':' '{print $2}' | sed '/^$/d' | sed '$d' | sed 's/ //g' > /$PADDLE_ROOT/pr-ut
cd $PADDLE_ROOT
grep -F -x -v -f br-ut pr-ut > $PADDLE_ROOT/added_ut
sort pr-ut |uniq -d > $PADDLE_ROOT/duplicate_ut
echo "New-UT:"
cat $PADDLE_ROOT/added_ut
rm -rf prec_build
rm $PADDLE_ROOT/br-ut $PADDLE_ROOT/pr-ut $PADDLE_ROOT/paddle/scripts/paddle_build_pre.sh
git checkout $CURBRANCH
echo $CURBRANCH
git branch -D prec_added_ut
cd $CURDIR
echo $CURDIR
export CI_SKIP_CPP_TEST=
