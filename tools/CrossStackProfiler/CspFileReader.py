# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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
import time
import json
import glob
import logging
import pandas as pd
from multiprocessing import Process, Lock
""" Some terms to clarify the code
    in most case, one or more paremeters may be set as input args for a class or a function
    in form of single variable or k-v dict

    1.  trainerId
    2.  gpuId
    3.  rankId
    4.  gpuPerTrainer
    5.  groupSize
    6.  groupId
    7.  groupNum
    8.  displaySize
    9.  dataPath
    10. resultPath
    11. fileOrganizeForm -- "byRank" OR "byTrainer" or "other"

"""

PIPELINEINFO_TRACE_NUM = 1

dcgmMetricParameterMap = {
    "02_gpuUtility": [("GPUTL", "GPUTL"), ("GRACT", "GRACT")],
    "03_smUtility": [("SMACT", "SMACT"), ("SMOCC", "SMOCC")],
    "04_memUtility": [("FB_USED_RATIO", "FB_USED_RATIO"), ("DRAMA", "DRAMA")],
    "05_txUtility": [("NVLTX", "NVLTX"), ("NVLRX", "NVLRX"), ("PCITX", "PCITX"),
                     ("PCIRX", "PCIRX")],
    "06_calUtility":
    [("FP32A", "FP32A"), ("FP16A", "FP16A"), ("TENSO", "TENSO")]
}
DCGMINFO_TRACE_NUM = len(dcgmMetricParameterMap.keys())
NETINFO_TRACE_NUM = 2

DCGM_PATH = "dcgm"
NET_PATH = "net"
TIME_PATH = "time"
PROFILE_PATH = "profile"

FILEORGANIZEFORM_BYRANK = "byRank"
FILEORGANIZEFORM_BYTRAINER = "byTrainer"
FILEORGANIZEFORM_BYOTHER = "other"
FILEORGANIZEFORM = [
    FILEORGANIZEFORM_BYRANK, FILEORGANIZEFORM_BYTRAINER,
    FILEORGANIZEFORM_BYOTHER
]


class FileReader(object):
    def __init__(self, logger, args):
        self._logger = logger
        self._args = args

        self._fileList = []
        self._fileNum = 0

        self._dataPath = ""
        self._groupSize = 0
        self._displaySize = 0
        self._organizeForm = FILEORGANIZEFORM_BYOTHER
        self._gpuPerTrainer = 0

        self._checkArgs()
        self._getFileList()

        self._lock = Lock()

    def printArgs(self):
        self._logger.info("dataPath:")
        self._logger.info(self._dataPath)
        self._logger.info("groupSize:")
        self._logger.info(self._groupSize)
        self._logger.info("displaySize:")
        self._logger.info(self._displaySize)
        self._logger.info("organizeForm:")
        self._logger.info(self._organizeForm)
        self._logger.info("gpuPerTrainer:")
        self._logger.info(self._gpuPerTrainer)
        self._logger.info("minTimeStamp:")
        self._logger.info(self._minTimeStamp)

    def _checkArgsKey(self, key, type):
        if not self._args.has_key(key):
            raise KeyError("args should has key [%s]!" % key)

        if not isinstance(self._args[key], type):
            raise TypeError(
                "Invalid type of key [%s] in args dict, it should be a %s!" %
                (key, type))

        exec("self._%s = self._args[\"%s\"]" % (key, key))

    def _align_ts(self, ts):
        return ts - self._minTimeStamp

    def _checkArgs(self):
        if not isinstance(self._args, dict):
            raise TypeError("Invalid type of args, it should be a dict!")

        self._checkArgsKey("organizeForm", str)
        if self._organizeForm not in FILEORGANIZEFORM or \
            self._organizeForm == FILEORGANIZEFORM_BYOTHER:
            raise NotImplementedError(
                "we have not known how to process this form of file [%s]!" %
                self._organizeForm)

        self._checkArgsKey("gpuPerTrainer", int)

        self._checkArgsKey("dataPath", str)
        if not os.path.exists(self._dataPath):
            raise IOError("input data path [%s] not existed!" %
                          (self._dataPath))

        self._checkArgsKey("groupSize", int)
        self._checkArgsKey("displaySize", int)
        self._checkArgsKey("minTimeStamp", int)

    def getFileListByGroup(self, groupId):
        lIndext = 0
        rIndext = 0

        if self._organizeForm == FILEORGANIZEFORM_BYTRAINER:
            lIndext = groupId * self._groupSize
            rIndext = (groupId + 1) * self._groupSize
        elif self._organizeForm == FILEORGANIZEFORM_BYRANK:
            lIndext = groupId * self._groupSize * self._gpuPerTrainer
            rIndext = (groupId + 1) * self._groupSize * self._gpuPerTrainer

        try:
            return self._fileList[lIndext:rIndext]
        except IndexError:
            raise IndexError("invalid index of file list")

    def getFileList(self):
        return self._getFileList

    def _cmp(self, x, y):
        return self._getId(x, self._organizeForm) - self._getId(
            y, self._organizeForm)

    def _getFileList(self):
        self._fileList = glob.glob(os.path.join(self._dataPath, "*.*"))

        # check unique
        idList = []
        newFileList = []
        for file in self._fileList:
            id = self._getId(file, self._organizeForm)
            if id not in idList:
                idList.append(id)
                newFileList.append(file)
            else:
                raise NotImplementedError(
                    "[%s] is repeated by id, we don not how to process it!" %
                    file)

        if not self._fileList:
            if (self._getId(self._fileList[-1]) - self._getId(self._fileList[0])
                ) != len(self._fileList) - 1:
                raise Exception("The file id should be countious!")
        # sort
        def _sortBySuffix(elem):
            return int(elem.split(".")[-1])

        self._fileList.sort(key=_sortBySuffix)

        if not self._fileList:
            self._logger.warning("we can not find any file in dir [%s]!" %
                                 self._dataPath)
        else:
            self._logger.info("file list in dir [%s] is : %s !" %
                              (self._dataPath, ',  '.join(self._fileList)))

        return self._fileList

    def _getId(self, fileName, organizeForm, sed="."):
        if self._organizeForm != organizeForm:
            raise TypeError("Can not get rank id when organizer form is not %s!"
                            % organizeForm)

        if not os.path.isfile(fileName):
            raise IOError("[%s] is not a valid file!" % (fileName))

        try:
            prefix_str = fileName.split(sed)[-1]
            try:
                return int(prefix_str)
            except ValueError, Argument:
                print(Argument)
                raise TypeError("invalid fileName [%s]" % fileName)

        except IndexError, Argument:
            print(Argument)
            raise TypeError(
                "invalid fileName [%s], the prefix should be a number!" %
                fileName)

    def getRankId(self, fileName, sed="."):
        return self._getId(fileName, FILEORGANIZEFORM_BYRANK, sed)

    def getRankNum(self):
        if self._organizeForm == FILEORGANIZEFORM_BYRANK:
            return len(self._fileList)

        elif self._organizeForm == FILEORGANIZEFORM_BYTRAINER:
            return len(self._fileList) * self._gpuPerTrainer

    def getTrainerNum(self):
        if self._organizeForm == FILEORGANIZEFORM_BYRANK:
            return len(self._fileList) / self._gpuPerTrainer

        elif self._organizeForm == FILEORGANIZEFORM_BYTRAINER:
            return len(self._fileList)

    def getTrainerId(self, fileName, sed="."):
        return self._getId(fileName, FILEORGANIZEFORM_BYTRAINER, sed)

    def _splitTaskListForMultiProcess(self, ls, n):
        if not isinstance(ls, list) or not isinstance(n, int):
            return []
        ls_len = len(ls)
        if n <= 0 or 0 == ls_len:
            return []
        if n >= ls_len:
            return [[i] for i in ls]
        else:
            j = int((ls_len + n - 1) / n)
            k = ls_len % n
            ls_return = []
            end = 0
            for i in range(0, (n) * j, j):
                if i < len(ls) and (i + j) < len(ls):
                    ls_return.append(ls[i:i + j])
                    end = i + j
            ls_return.append(ls[end:])
            return ls_return

    def getOpInfoFileName(self, groupId, gpuId, tmpPath="./tmp"):
        return self.getFileName("opinfo", groupId, gpuId, tmpPath)

    def getPipeLineInfoFileName(self, groupId, gpuId, tmpPath="./tmp"):
        return self.getFileName("pipilineinfo", groupId, gpuId, tmpPath)

    def getDCGMInfoFileName(self, groupId, gpuId, tmpPath="./tmp"):
        return self.getFileName("dcgm", groupId, gpuId, tmpPath)

    def getFileName(self, name, groupId, gpuId, tmpPath="./tmp"):
        return os.path.join(tmpPath, "%s_%d_%d.json" % (name, groupId, gpuId))

    def getOpInfoDict(self, groupId, gpuId, tmpPath="./tmp"):
        return self.getDict("opinfo", groupId, gpuId, tmpPath)

    def getDcgmInfoDict(self, groupId, gpuId, tmpPath="./tmp"):
        return self.getDict("dcgm", groupId, gpuId, tmpPath)

    def getDict(self, name, groupId, gpuId, tmpPath="./tmp"):
        fileName = self.getFileName(name, groupId, gpuId, tmpPath)
        if not os.path.isfile(fileName):
            raise IOError("[%s] is not existed!" % fileName)

        data = {}
        with open(fileName, "r") as rf:
            try:
                data = json.load(rf)
            except Exception:
                self._logger.error("read [%s] error. not a json file!" %
                                   (fileName))
                raise TypeError("read [%s] error. not a json file!" %
                                (fileName))
        return data

    def dumpOpInfoDict(self,
                       data,
                       groupId,
                       gpuId,
                       pretty=False,
                       tmpPath="./tmp"):
        return self.dumpDict(
            data, "opinfo", groupId, gpuId, pretty=False, tmpPath="./tmp")

    def dumpDCGMDict(self, data, groupId, gpuId, pretty=False, tmpPath="./tmp"):
        return self.dumpDict(
            data, "dcgm", groupId, gpuId, pretty=False, tmpPath="./tmp")

    def dumpDict(self,
                 data,
                 name,
                 groupId,
                 gpuId,
                 pretty=False,
                 tmpPath="./tmp"):
        self._lock.acquire()
        if not os.path.exists(tmpPath):
            os.makedirs(tmpPath)
        self._lock.release()
        if pretty:
            jsObj = json.dumps(data, indent=4, separators=(',', ': '))
        else:
            jsObj = json.dumps(data, separators=(',', ':'))

        fileName = self.getFileName(name, groupId, gpuId, tmpPath)
        if os.path.isfile(fileName):
            os.remove(fileName)

        fileObject = open(fileName, 'w')
        fileObject.write(jsObj)
        fileObject.close()
        self._logger.info("dump [%s] sucessfully!" % fileName)


def getLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    rq = time.strftime('%Y%m%d%H%M.%s', time.localtime(time.time()))
    log_path = os.path.dirname(os.getcwd()) + '/Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_name = log_path + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(process)d - %(levelname)s: %(message)s"
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger


def test_FileReader(args):
    try:
        testReader = FileReader(None, args)
    except Exception, Argument:
        print(Argument)
    else:
        testReader.printArgs()


if __name__ == "__main__":
    args = 0
    test_FileReader(args)

    args = {
        "dataPath": ".",
        "groupSize": 1,
        "displaySize": 1,
        "gpuPerTrainer": 8,
        "organizeForm": FILEORGANIZEFORM_BYOTHER,
    }
    test_FileReader(args)

    args = {
        "dataPath": ".",
        "groupSize": 1,
        "displaySize": 1,
        "gpuPerTrainer": 8,
        "organizeForm": FILEORGANIZEFORM_BYTRAINER,
    }
    test_FileReader(args)

    args = {
        "dataPath": "./res",
        "groupSize": 1,
        "displaySize": 1,
        "gpuPerTrainer": 8,
        "organizeForm": FILEORGANIZEFORM_BYTRAINER,
    }
    test_FileReader(args)

    args = {
        "dataPath": ".",
        "groupSize": "",
        "displaySize": 1,
        "gpuPerTrainer": 8,
        "organizeForm": FILEORGANIZEFORM_BYTRAINER,
    }
    test_FileReader(args)
