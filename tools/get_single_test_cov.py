# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import json
import time
import sys
import re


def getFNDAFile(rootPath, test):
    filename = '%s/build/ut_map/%s/coverage.info.tmp' % (rootPath, test)
    fn_filename = '%s/build/ut_map/%s/fnda.tmp' % (rootPath, test)
    os.system('touch %s' % fn_filename)
    f = open(filename)
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '')
        if line.startswith(('SF:')):
            os.system('echo %s >> %s' % (line, fn_filename))
        elif line.startswith(('FNDA:')):
            hit = int(line.split('FNDA:')[1].split(',')[0])
            if hit != 0:
                os.system('echo %s >> %s' % (line, fn_filename))
    f.close()


def analysisFNDAFile(rootPath, test):
    related_ut_map_file = '%s/build/ut_map/%s/related_%s.txt' % (rootPath, test,
                                                                 test)
    notrelated_ut_map_file = '%s/build/ut_map/%s/notrelated_%s.txt' % (
        rootPath, test, test)
    os.system('touch %s' % related_ut_map_file)
    os.system('touch %s' % notrelated_ut_map_file)
    fn_filename = '%s/build/ut_map/%s/fnda.tmp' % (rootPath, test)
    f = open(fn_filename)
    data = f.read().split('SF:')
    related_file_list = []
    for message in data:
        message_list = message.split('\n')
        clazz_filename = message_list[0]
        if '/build/' in clazz_filename:
            clazz_filename = clazz_filename.replace('/build', '')
        if '.pb.h' in clazz_filename:
            clazz_filename = clazz_filename.replace('.pb.h', '.proto')
        if '.pb.cc' in clazz_filename:
            clazz_filename = clazz_filename.replace('.pb.cc', '.proto')
        if 'FNDA:' in message:
            OP_REGIST = True
            for i in range(1, len(message_list) - 1):
                fn = message_list[i]
                matchObj = re.match(
                    r'(.*)Maker(.*)|(.*)Touch(.*)Regist(.*)|(.*)Touch(.*)JitKernel(.*)|(.*)converterC2Ev(.*)',
                    fn, re.I)
                if matchObj == None:
                    OP_REGIST = False
                    break
            if OP_REGIST == False:
                related_file_list.append(clazz_filename)
                os.system('echo %s >> %s' %
                          (clazz_filename, related_ut_map_file))
            else:
                os.system('echo %s >> %s' %
                          (clazz_filename, notrelated_ut_map_file))
        else:
            if clazz_filename != '':
                if clazz_filename not in related_file_list:  # xx.pb.cc in RELATED xx.pb.h not in RELATED 
                    os.system('echo %s >> %s' %
                              (clazz_filename, notrelated_ut_map_file))
    f.close()


def getCovinfo(rootPath, test):
    ut_map_path = '%s/build/ut_map/%s' % (rootPath, test)
    os.system(
        'cd %s && lcov --capture -d . -o coverage.info --rc lcov_branch_coverage=0 > /dev/null 2>&1'
        % ut_map_path)
    os.system(
        "cd %s && lcov --extract coverage.info '/paddle/paddle/fluid/framework/*' '/paddle/paddle/fluid/imperative/*' '/paddle/paddle/fluid/inference/*' '/paddle/paddle/fluid/memory/*' '/paddle/paddle/fluid/operators/*' '/paddle/paddle/fluid/string/*' '/paddle/paddle/fluid/distributed/*' '/paddle/paddle/fluid/extension/*' '/paddle/paddle/fluid/platform/*' '/paddle/paddle/fluid/pybind/*' '/paddle/build/*' -o coverage.info.tmp --rc lcov_branch_coverage=0 > /dev/null 2>&1"
        % ut_map_path)
    os.system('rm -rf %s/paddle' % ut_map_path)
    os.system('rm -rf %s/coverage.info' % ut_map_path)
    getFNDAFile(rootPath, test)
    analysisFNDAFile(rootPath, test)


if __name__ == "__main__":
    rootPath = sys.argv[1]
    case = sys.argv[2]
    getCovinfo(rootPath, case)
