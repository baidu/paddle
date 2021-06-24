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

from xml.etree import ElementTree
import re
import time
import queue
import threading
import os
import json
import sys

taskQueue = queue.Queue()
lock = threading.RLock()


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


def getPyCovResult(params):
    rootPath = params[0]
    ut = params[1]
    print("ut: %s" % ut)
    startTime = int(time.time())
    path = '%s/build/pytest/%s' % (rootPath, ut)
    os.system('cd %s && coverage combine `ls python-coverage.data.*`' % path)
    os.system('cd %s && pwd && coverage xml -i -o python-coverage.xml' % path)
    xml_path = '%s/python-coverage.xml' % path
    os.system("python2.7 %s/tools/analysisPyXml.py %s %s" %
              (rootPath, rootPath, ut))
    endTime = int(time.time())
    print('pyCov Time: %s' % (endTime - startTime))


def doFun(params):
    getPyCovResult(params)


def main(rootPath):
    """
    1. get gcov file
    2. get gcov file not coverageratio = 0
    """
    path = '%s/build/pytest' % rootPath
    dirs = os.listdir(path)
    pool = threadPool(23)
    for i in range(pool.__len__()):
        pool[i].start()
    for ut in dirs:
        params = [rootPath, ut]
        taskQueue.put(params)
    taskQueue.join()


if __name__ == "__main__":
    rootPath = sys.argv[1]
    main(rootPath)
