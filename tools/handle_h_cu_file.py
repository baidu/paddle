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

import queue
import threading
import os
import json
import time
import sys

taskQueue = queue.Queue()


def worker(fun):
    while True:
        temp = taskQueue.get()
        fun(temp)
        taskQueue.task_done()


def threadPool(threadPoolNum):
    threadPool = []
    for i in range(threadPoolNum):
        thread = threading.Thread(target=worker, args={doFun, })
        thread.daemon = True
        threadPool.append(thread)
    return threadPool


def get_h_file_md5(rootPath):
    h_cu_files = '%s/tools/h_cu_files.log' % rootPath
    f = open(h_cu_files)
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        os.system('md5sum %s >> %s/tools/h_cu_md5.log' % (line, rootPath))


def insert_pile_to_h_file(rootPath):
    h_cu_files = '%s/tools/h_cu_files.log' % rootPath
    f = open(h_cu_files)
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        func = line.replace('/', '_').replace('.', '_')
        os.system('echo "\n#ifndef _PRECISE%s_\n" >> %s' % (func.upper(), line))
        os.system('echo "#define _PRECISE%s_" >> %s' % (func.upper(), line))
        os.system('echo "\n#include <cstdio>\n" >> %s' % line)
        os.system(
            'echo "__attribute__((constructor)) static void calledFirst%s()\n{" >> %s'
            % (func, line))
        os.system(
            'echo \'    printf("precise test map fileeee: %%s\\\\n", __FILE__);\n}\' >> %s'
            % line)
        os.system('echo "\n#endif" >> %s' % line)


def remove_pile_from_h_file(rootPath):
    h_cu_files = '%s/tools/h_cu_files.log' % rootPath
    f = open(h_cu_files)
    lines = f.readlines()
    count = 12
    for line in lines:
        line = line.strip()
        while count > 0:
            os.system("sed -i '$d' %s" % line)
            count = count - 1
        count = 12


def get_h_cu_file(file_path):
    rootPath = file_path[0]
    dir_path = file_path[1]
    filename = file_path[2]
    ut = filename.replace('^', '').replace('$', '').replace('.log', '')
    os.system(
        "cat %s/%s | grep 'precise test map fileeee:'| uniq >> %s/build/ut_map/%s/related_%s.txt"
        % (dir_path, filename, rootPath, ut, ut))


def doFun(file_path):
    get_h_cu_file(file_path)


def main(rootPath, dir_path):
    """
    get useful message
    """
    startTime = int(time.time())
    test_h_cu_dict = {}
    pool = threadPool(23)
    for i in range(pool.__len__()):
        pool[i].start()
    files = os.listdir(dir_path)
    for filename in files:
        file_path = [rootPath, dir_path, filename]
        taskQueue.put(file_path)
    taskQueue.join()
    endTime = int(time.time())
    print('analy h/cu file cost Time: %s' % (endTime - startTime))


if __name__ == "__main__":
    func = sys.argv[1]
    if func == 'get_h_file_md5':
        rootPath = sys.argv[2]
        get_h_file_md5(rootPath)
    elif func == 'insert_pile_to_h_file':
        rootPath = sys.argv[2]
        insert_pile_to_h_file(rootPath)
    elif func == 'analy_h_cu_file':
        dir_path = sys.argv[2]
        rootPath = sys.argv[3]
        main(rootPath, dir_path)
    elif func == 'remove_pile_from_h_file':
        rootPath = sys.argv[2]
        remove_pile_from_h_file(rootPath)
