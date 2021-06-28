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
import sys
import re
import json


def get_all_paddle_file(rootPath):
    """get all file in Paddle repo: paddle/fluild, python"""
    traverse_files = ['%s' % rootPath]
    all_file_paddle = '%s/build/all_file_paddle' % rootPath
    all_file_paddle_list = []
    with open(all_file_paddle, 'w') as f:
        for filename in traverse_files:
            g = os.walk(filename)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    all_file_paddle_list.append(os.path.join(path, file_name))
    return all_file_paddle_list


def get_all_uts(rootPath):
    all_uts_paddle = '%s/build/all_uts_paddle' % rootPath
    os.system(
        'cd %s/build && ctest -N -V | grep -Ei "Test[ \t]+#" | grep -oEi "\w+$" > %s'
        % (rootPath, all_uts_paddle))


def remove_useless_file(rootPath):
    """remove useless file in ut_file_map.json"""
    all_file_paddle_list = get_all_paddle_file(rootPath)
    ut_file_map_new = {}
    ut_file_map = "%s/build/ut_file_map.json" % rootPath
    with open(ut_file_map, 'r') as load_f:
        load_dict = json.load(load_f)
    for key in load_dict:
        if key in all_file_paddle_list:
            ut_file_map_new[key] = load_dict[key]

    with open("%s/build/ut_file_map.json" % rootPath, "w") as f:
        json.dump(ut_file_map_new, f, indent=4)
        print("remove_useless_file ut_file_map success!!")


def handle_ut_file_map(rootPath):
    utNotSuccess_list = []
    ut_map_path = "%s/build/ut_map" % rootPath
    files = os.listdir(ut_map_path)
    ut_file_map = {}
    count = 0
    not_success_file = open("%s/build/prec_delta" % rootPath, 'w')
    for ut in files:
        count = count + 1
        print("ut %s: %s" % (count, ut))
        coverage_info = '%s/%s/coverage.info.tmp' % (ut_map_path, ut)
        if os.path.exists(coverage_info):
            filename = '%s/%s/related_%s.txt' % (ut_map_path, ut, ut)
            f = open(filename)
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '').strip()
                if line == '':
                    continue
                elif line.startswith('/paddle/build'):
                    source_file = line.replace('/build', '')
                    #source_file = re.sub('.pb.*', '.proto', source_file)
                elif 'precise test map fileeee:' in line:
                    source_file = line.split('precise test map fileeee:')[
                        1].strip()
                else:
                    source_file = line
                if source_file not in ut_file_map:
                    ut_file_map[source_file] = []
                if ut not in ut_file_map[source_file]:
                    ut_file_map[source_file].append(ut)
        else:
            not_success_file.write('%s\n' % ut)
            utNotSuccess_list.append(ut)
    not_success_file.close()
    
    print("utNotSuccess:")
    print(utNotSuccess_list)

    for ut in files:
        if ut not in utNotSuccess_list:
            filename = '%s/%s/notrelated_%s.txt' % (ut_map_path, ut, ut)
            f = open(filename)
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '').strip()
                if line == '':
                    continue
                elif line.startswith('/paddle/build'):
                    source_file = line.replace('/build', '')
                else:
                    source_file = line
                if source_file not in ut_file_map:
                    ut_file_map[source_file] = []

    with open("%s/build/ut_file_map.json" % rootPath, "w") as f:
        json.dump(ut_file_map, f, indent=4)

def notsuccessfuc(rootPath):
    utNotSuccess = ''
    ut_map_path = "%s/build/ut_map" % rootPath
    files = os.listdir(ut_map_path)
    count = 0
    # ut failed!!
    for ut in files:
        coverage_info = '%s/%s/coverage.info.tmp' % (ut_map_path, ut)
        if os.path.exists(coverage_info):
            pass
        else:
            count = count + 1
            utNotSuccess = utNotSuccess + '^%s$|' % ut

    # ut not exec
    get_all_uts(rootPath)
    with open("/paddle/build/all_uts_paddle", "r") as f:
        data = f.readlines()
    for ut in data:
        ut = ut.replace('\n', '').strip()
        if ut not in files:
            print(ut)
            count = count + 1
            utNotSuccess = utNotSuccess + '^%s$|' % ut

    if utNotSuccess != '':
        print("utNotSuccess count: %s" % count)
        f = open('%s/build/utNotSuccess' % rootPath, 'w')
        f.write(utNotSuccess[:-1])
        f.close()


def ut_file_map_supplement(rootPath):
    ut_file_map_new = "%s/build/ut_file_map.json" % rootPath
    os.system('mkdir /pre_test')
    os.system(
        'cd /pre_test && wget --no-proxy https://paddle-docker-tar.bj.bcebos.com/pre_test/ut_file_map.json --no-check-certificate'
    )
    ut_file_map_old = "/pre_test/ut_file_map.json"
    with open(ut_file_map_new, 'r') as load_f:
        load_dict_new = json.load(load_f)
    with open(ut_file_map_old, 'r') as f:
        load_dict_old = json.load(f)

    all_uts_paddle = '%s/build/all_uts_paddle' % rootPath
    with open(all_uts_paddle, 'r') as f:
        all_uts_paddle_list = []
        for ut in f.readlines():
            all_uts_paddle_list.append(ut.strip())
        f.close()

    for filename in load_dict_old:
        if filename not in load_dict_new:
            load_dict_new[filename] = load_dict_old[filename]

    with open("/pre_test/ut_file_map.json", "w") as f:
        json.dump(load_dict_new, f, indent=4)
        print("load_dict_new success!!")

    os.system(
        'cd /pre_test && wget --no-proxy https://paddle-docker-tar.bj.bcebos.com/pre_test/prec_delta --no-check-certificate'
    )
    prec_delta_old = '/pre_test/prec_delta'
    prec_delta_new = "%s/build/prec_delta" % rootPath
    with open(prec_delta_old, 'r') as f:
        prec_delta_old_list = []
        for ut in f.readlines():
            prec_delta_old_list.append(ut.strip())
        f.close()
    with open(prec_delta_new, 'r') as f:
        prec_delta_new_list = []
        for ut in f.readlines():
            prec_delta_new_list.append(ut.strip())
        f.close()
    for ut in prec_delta_old_list:
        filename = '%s/build/ut_map/%s/coverage.info.tmp' % (rootPath, ut)
        if ut in all_uts_paddle_list:
            if not os.path.exists(filename) and ut not in prec_delta_new_list:
                prec_delta_new_list.append(ut)
    prec_delta_file = open("/pre_test/prec_delta", 'w')
    for ut in prec_delta_new_list:
        prec_delta_file.write(ut + '\n')
    print("prec_delta_file success!!")
    prec_delta_file.close()


if __name__ == "__main__":
    func = sys.argv[1]
    if func == 'get_not_success_ut':
        rootPath = sys.argv[2]
        notsuccessfuc(rootPath)
    elif func == 'get_ut_map':
        rootPath = sys.argv[2]
        handle_ut_file_map(rootPath)
        remove_useless_file(rootPath)
        ut_file_map_supplement(rootPath)
