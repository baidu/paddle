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
import re
import json
import glob
import logging
import tempfile
import argparse
import pandas as pd
import multiprocessing
from multiprocessing import Process

from CspChromeTraceFormatter import ChromeTraceFormatter

from CspFileReader import FileReader
from CspFileReader import getLogger
from CspFileReader import dcgmMetricParameterMap
from CspFileReader import TIME_PATH, DCGM_PATH, NET_PATH, PROFILE_PATH
from CspFileReader import NETINFO_TRACE_NUM, DCGMINFO_TRACE_NUM, PIPELINEINFO_TRACE_NUM
from CspFileReader import FILEORGANIZEFORM_BYRANK, FILEORGANIZEFORM_BYTRAINER, FILEORGANIZEFORM_BYOTHER, FILEORGANIZEFORM


class dcgmFileReader(FileReader):
    def parseFileByGroup(self, groupId, processNum=8):
        fileFist = self.getFileListByGroup(groupId)
        displaySize = min(self._displaySize, len(fileFist))
        fileFist = fileFist[:displaySize]

        if processNum == 0:
            return self._parseTask(fileFist)

        else:
            self._logger.info("using [%d] process to do this work!" %
                              processNum)
            processPool = []
            pidList = []

            manager = multiprocessing.Manager()
            q = manager.Queue()

            taskList = self._splitTaskListForMultiProcess(fileFist, processNum)
            for task in taskList:
                subproc = Process(
                    target=self._parseTask, args=(
                        task,
                        q, ))
                processPool.append(subproc)
                subproc.start()
                pidList.append(subproc.pid)
                self._logger.info(
                    "[DCGM reader]: process [%d] has been started, total task num is %d ..."
                    % (subproc.pid, len(processPool)))

            for t in processPool:
                t.join()
                pidList.remove(t.pid)
                self._logger.info(
                    "[DCGM reader]: process [%d] has exited! remained %d process!"
                    % (t.pid, len(pidList)))

            isFistProcess = True
            for t in processPool:
                if isFistProcess:
                    isFistProcess = False
                    dcgm_data = q.get()
                else:
                    dcgm_data = pd.concat(
                        [dcgm_data, q.get()], axis=0, join='outer')

            return dcgm_data

    def _parseTask(self, taskList, q=None):
        is_first = True
        for fileName in taskList:
            self._logger.info("I am processing %s!" % fileName)
            tmp_data = self._parseSingleFile(fileName)
            if tmp_data is None:
                continue

            if is_first:
                is_first = False
                dcgm_data = tmp_data
            else:
                dcgm_data = pd.concat(
                    [dcgm_data, tmp_data], axis=0, join='outer')
        dcgm_data = dcgm_data.dropna()
        if not q is None:
            q.put(dcgm_data)
        self._logger.info("I finish processing %s!" % fileName)
        return dcgm_data

    def _parseSingleFile(self, fileName):
        trainerId = self.getTrainerId(fileName)

        if not os.path.exists(fileName):
            logging.warning(fileName + ' not found')
            return

        regex_list = [
            (re.compile(r' +'), ','),
            (re.compile(r'^,'), ''),
        ]

        csv_tempfile = tempfile.TemporaryFile()
        with open(fileName, 'r') as fp:
            has_header = False

            for line in fp:
                # skip `nvidia-dcgm-dmon.sh` init and fini info lines
                if 'nv-hostengine' in line or 'dmon' in line or 'Host Engine Listener Started' in line:
                    continue

                if not line.strip().startswith("GPU") and not line.strip(
                ).startswith("# Entity"):
                    continue

                # skip non-needed headers (only the header in 1th line was needed)
                if line.strip().startswith("# Entity"):
                    line = line.strip()[2:]

                if 'Entity' == line[0:len('Entity')]:
                    if has_header:
                        continue
                    else:
                        has_header = True

                if line.strip().startswith("GPU"):
                    line = line.strip()[3:]

                for r in regex_list:
                    line = r[0].sub(r[1], line)

                csv_tempfile.write(bytes(line + "\n"))

        csv_tempfile.seek(0)

        dcgm = pd.read_csv(csv_tempfile, header=0, delimiter=',')
        # dcgm.info()
        dcgm['FB_USED_RATIO'] = dcgm['FBUSD'] / dcgm['FBTTL']
        dcgm['GPUTL'] = dcgm['GPUTL'] / 100.0
        dcgm['ts'] = dcgm['TIMESTAMP'] * 1e9
        dcgm['trainerId'] = trainerId

        return dcgm

    def _getDCGMTraceInfoByGpuId(self,
                                 groupId,
                                 gpuId,
                                 dcgm_data,
                                 pid_map,
                                 q=None):
        self._logger.info(
            "Begin to generate dcgm info, groupId = %d, gpuID = %d ..." %
            (groupId, gpuId))

        gpuDcgmData = dcgm_data[dcgm_data['Entity'].isin([gpuId])]

        traceEventList = []
        for metric, parameteList in dcgmMetricParameterMap.items():
            metaInfo = {}
            metaInfo['name'] = 'process_name'
            metaInfo['ph'] = 'M'
            metaInfo['pid'] = pid_map[metric]
            metaInfo['args'] = {'name': metric}
            traceEventList.append(metaInfo)

        for index, row in gpuDcgmData.iterrows():
            for metric, parameteList in dcgmMetricParameterMap.items():
                trainerId = int(row['trainerId']) % self._groupSize
                if trainerId >= self._displaySize:
                    continue

                di = {}
                # name = "%s_%d" % (metric, trainerId)
                name = "%s" % (metric)
                di['name'] = name
                di['pid'] = pid_map[metric]
                di['ts'] = self._align_ts(int(row['ts']))
                # di['ts'] = int(row['ts'])
                di['cat'] = metric
                di['tid'] = "%d_%d" % (groupId, trainerId)
                di['ph'] = "C"
                di['id'] = trainerId

                args = {}
                for p in parameteList:
                    args[p[0]] = row[p[1]]
                di['args'] = args

                traceEventList.append(di)
        trace = {}
        trace['traceEvents'] = traceEventList
        self.dumpDCGMDict(trace, groupId, gpuId, True)

        return trace

    def getDCGMTraceInfo(self, groupId, processNum=8):
        dcgm_data = self.parseFileByGroup(groupId, processNum)

        pid_map = {}
        init_pid = PIPELINEINFO_TRACE_NUM

        for metric in dcgmMetricParameterMap.keys():
            pid_map[metric] = init_pid
            init_pid = init_pid + 1

        manager = multiprocessing.Manager()
        q = manager.Queue()
        processPool = []
        pidList = []

        for gpuId in range(self._gpuPerTrainer):
            subproc = Process(
                target=self._getDCGMTraceInfoByGpuId,
                args=(
                    groupId,
                    gpuId,
                    dcgm_data,
                    pid_map,
                    q, ))
            processPool.append(subproc)
            subproc.start()
            pidList.append(subproc.pid)
            self._logger.info(
                "[DCGM info]: process [%d] has been started, total task num is %d ..."
                % (subproc.pid, 1))

        for t in processPool:
            t.join()
            pidList.remove(t.pid)
            self._logger.info(
                "[DCGM info]: process [%d] has exited! remained %d process!" %
                (t.pid, len(pidList)))

        dcgmInfo = {}

        return dcgmInfo


def test_dcgmFileReader():
    args = {
        "dataPath": "data/newdata/dcgm",
        "groupSize": 4,
        "displaySize": 8,
        "gpuPerTrainer": 8,
        "minTimeStamp": 0,
        "organizeForm": FILEORGANIZEFORM_BYTRAINER,
    }

    testReader = dcgmFileReader(getLogger(), args)
    testReader.printArgs()
    data = testReader.getDCGMTraceInfo(0, 8)


if __name__ == "__main__":
    test_dcgmFileReader()
