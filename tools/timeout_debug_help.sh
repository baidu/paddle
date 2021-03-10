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
failed_uts=$1
need_debug_ut_re='test_dist_fleet'
cat_log_judge=$(echo "${failed_uts}" | grep -oEi "$need_debug_ut_re" )
if [[ "$cat_log_judge" != "" ]];then 
    cat tmp/tr0_err.log tmp/tr1_err.log tmp/ps0_err.log tmp/ps1_err.log
    cat tmp/heter0_err.log tmp/heter1_err.log
fi
set -e
