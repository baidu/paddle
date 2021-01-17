#!/bin/env python
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
import sys
import os 

def added_ut_repeat_check(br_ut,pr_ut):
    """Check whether the new unit test has the same name as the existing unit test"""
    repeat_ut=[]
    if os.path.exists(br_ut) and os.path.exists(pr_ut):
        with open(br_ut,'r') as f:
            br_ut = f.read().strip().split()
        with open(pr_ut,'r') as f:
            pr_ut = f.read().strip().split()
        for item in br_ut:
            if item in pr_ut:
                pr_ut.remove(item)
        for item in pr_ut:
            if item in br_ut:
                repeat_ut.append(item)
        print('\n'.join(repeat_ut))
    else:
        exit(102)


if __name__=='__main__':
    if len(sys.argv)==3:
        try:
            added_ut_repeat_check(sys.argv[1],sys.argv[2])
        except Exception as e:
            print(e)
            sys.exit(103)
    else:
        print(
            "Usage: python check_ut_duplicate_name file1 file2"
    )
