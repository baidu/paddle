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
import glob
import logging
import argparse
import multiprocessing

import pandas as pd
from multiprocessing import Process

from NetFileReader import netFileReader
from DCGMFileReader import dcgmFileReader
from ProfileFileReader import profileFileReader

from CspFileReader import getLogger
from CspFileReader import TIME_PATH, DCGM_PATH, NET_PATH, PROFILE_PATH
from CspFileReader import NETINFO_TRACE_NUM, DCGMINFO_TRACE_NUM, PIPELINEINFO_TRACE_NUM
from CspFileReader import FILEORGANIZEFORM_BYRANK, FILEORGANIZEFORM_BYTRAINER, FILEORGANIZEFORM_BYOTHER, FILEORGANIZEFORM


def get_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--profile_path',
        type=str,
        default='.',
        help='Working path that store the monitor data.')

    parser.add_argument(
        '--timeline_path',
        type=str,
        default='.',
        help='Output timeline file name.')

    parser.add_argument(
        '--gpuPerTrainer', type=int, default=8, help='Gpus per trainer.')

    parser.add_argument(
        '--trainerNum', type=int, default=4, help='Num of trainer.')

    parser.add_argument(
        '--groupSize', type=int, default=8, help='Num of trainer in a group.')

    parser.add_argument(
        '--displaySize',
        type=int,
        default=2,
        help='Num of line need to display in a group.')

    return parser.parse_args()


class CspReporter(object):
    def __init__(self, args):
        self._args = args
        print(self._args)

        self._workPath = self._args.profile_path
        self._saveFilePath = self._args.timeline_path
        self._gpuPerTrainer = self._args.gpuPerTrainer
        self._groupSize = self._args.groupSize
        self._displaySize = self._args.displaySize
        self._trainerNum = self._args.trainerNum

        self._checkArgs()

        self._init_logger()
        self._init_timeInfo()
        self._init_reader()

    def _checkArgs(self):
        if self._trainerNum % self._groupSize != 0:
            raise Exception(
                "Input args error: trainerNum[%d] %% groupSize[%d] != 0" %
                (self._trainerNum, self._groupSize))

    def _init_logger(self):
        self._logger = getLogger()

    def _init_reader(self):
        self._dcgmPath = os.path.join(self._workPath, DCGM_PATH)
        self._netPath = os.path.join(self._workPath, NET_PATH)
        self._profilePath = os.path.join(self._workPath, PROFILE_PATH)

        self._netFileReaderArgs = {
            "dataPath": self._netPath,
            "groupSize": self._groupSize,
            "displaySize": self._displaySize,
            "gpuPerTrainer": self._gpuPerTrainer,
            "minTimeStamp": self._minTimeStamp,
            "organizeForm": FILEORGANIZEFORM_BYTRAINER,
        }

        self._dcgmFileReaderArgs = {
            "dataPath": self._dcgmPath,
            "groupSize": self._groupSize,
            "displaySize": self._displaySize,
            "gpuPerTrainer": self._gpuPerTrainer,
            "minTimeStamp": self._minTimeStamp,
            "organizeForm": FILEORGANIZEFORM_BYTRAINER,
        }

        self._profileFileReaderArgs = {
            "dataPath": self._profilePath,
            "groupSize": self._groupSize,
            "displaySize": self._displaySize,
            "gpuPerTrainer": self._gpuPerTrainer,
            "minTimeStamp": self._minTimeStamp,
            "organizeForm": FILEORGANIZEFORM_BYRANK,
        }

        self._dcgmFileReader = dcgmFileReader(self._logger,
                                              self._dcgmFileReaderArgs)
        self._profileFileReader = profileFileReader(self._logger,
                                                    self._profileFileReaderArgs)

    def _init_timeInfo(self):
        self._timePath = os.path.join(self._workPath, TIME_PATH)
        self._timeInfo = {}
        self._minTimeStamp = 0
        self._set_timeInfo()

    def _set_timeInfo(self, timeFileNamePrefix="time.txt", sed="."):
        timeFileNameList = glob.glob(
            os.path.join(self._timePath, timeFileNamePrefix, sed, "*"))
        for timeFileName in timeFileNameList:
            trainerId = int(timeFileName.split(sed)[-1])
            gpuId = int(timeFileName.split(sed)[-2])
            info = {}
            with open(timeFileName, "r") as rf:
                for line in rf:
                    if line.startswith("start time:"):
                        info["start_time"] = int(
                            float(line.split(":")[-1]) * 1e9)

                        self._minTimeStamp = min(self._minTimeStamp,
                                                 info["start_time"])

                    if line.startswith("end time:"):
                        info["end_time"] = int(float(line.split(":")[-1]) * 1e9)
            if not info:
                self._timeInfo[gpuId * trainerId] = info

    def _generateTraceFileByGroupAndGpuId(self, pipileInfo, netInfo, groupId,
                                          gpuId):
        dcgmInfoDict = self._dcgmFileReader.getDcgmInfoDict(groupId, gpuId)
        opInfoDict = self._profileFileReader.getOpInfoDict(groupId, gpuId)

        traceObj = {}
        traceObj["traceEvents"] = pipileInfo[str(gpuId)] + opInfoDict[
            "traceEvents"] + dcgmInfoDict["traceEvents"] + netInfo[
                "traceEvents"]

        self._profileFileReader.dumpDict(traceObj, "traceFile", groupId, gpuId,
                                         False, self._saveFilePath)

    def _generateTraceFileByGroup(self, groupId, processNum):
        # first we need to generate pipeline info
        pipileInfo = self._profileFileReader.getPipeLineInfo(groupId,
                                                             processNum)
        # second we need to generate dcgm info
        dcgmInfo = self._dcgmFileReader.getDCGMTraceInfo(groupId, processNum)

        # third we need to generate net info
        netInfo = {}
        netInfo["traceEvents"] = []
        # netInfo = self._netFileReader.parseFileByGroup(groupId, processNum)

        # forth we need to generate op info
        opInfo = self._profileFileReader.getOPTraceInfo(groupId)

        # finially we need dump this information into disk
        processPool = []
        pidList = []

        for gpuId in range(self._gpuPerTrainer):
            subproc = Process(
                target=self._generateTraceFileByGroupAndGpuId,
                args=(
                    pipileInfo,
                    netInfo,
                    groupId,
                    gpuId, ))
            processPool.append(subproc)
            subproc.start()
            pidList.append(subproc.pid)
            self._logger.info(
                "[traceFile]: process [%d] has been started, total task num is %d ..."
                % (subproc.pid, 1))

        for t in processPool:
            t.join()
            pidList.remove(t.pid)
            self._logger.info(
                "[traceFile]: process [%d] has exited! remained %d process!" %
                (t.pid, len(pidList)))

    def generateTraceFile(self, processNum=8):
        processPool = []
        pidList = []
        for groupId in range(self._trainerNum / self._groupSize):
            subproc = Process(
                target=self._generateTraceFileByGroup,
                args=(
                    groupId,
                    processNum, ))
            processPool.append(subproc)
            subproc.start()
            pidList.append(subproc.pid)
            self._logger.info(
                "[GroupTraceFile]: process [%d] has been started, total task num is %d ..."
                % (subproc.pid, 1))
        for t in processPool:
            t.join()
            pidList.remove(t.pid)
            self._logger.info(
                "[GroupTraceFile]: process [%d] has exited! remained %d process!"
                % (t.pid, len(pidList)))


if __name__ == '__main__':
    args = get_argparse()
    tl = CspReporter(args)
    tl.generateTraceFile()
