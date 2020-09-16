#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""hdfs_utils.py will move to fluid/incubate/fleet/utils/hdfs.py"""

import os
import sys
import subprocess
import multiprocessing
from datetime import datetime
from functools import cmp_to_key

import re
import copy
import errno

import logging
from paddle.fluid.log_helper import get_logger

__all__ = ["HDFSClient", "multi_download", "multi_upload"]

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class HDFSClient(object):
    """
    A tool of HDFS 

    Args:
        hadoop_home (string): hadoop_home 
        configs (dict): hadoop config, it is a dict, please contain \
            key "fs.default.name" and "hadoop.job.ugi"
        Can be a float value
    Examples:
        hadoop_home = "/home/client/hadoop-client/hadoop/"

        configs = {
            "fs.default.name": "hdfs://xxx.hadoop.com:54310",
            "hadoop.job.ugi": "hello,hello123"
        }

        client = HDFSClient(hadoop_home, configs)

        client.ls("/user/com/train-25")
        files = client.lsr("/user/com/train-25/models")
    """

    def __init__(self, hadoop_home, configs):
        self.pre_commands = []
        hadoop_bin = '%s/bin/hadoop' % hadoop_home
        self.pre_commands.append(hadoop_bin)
        dfs = 'fs'
        self.pre_commands.append(dfs)

        for k, v in configs.items():
            config_command = '-D%s=%s' % (k, v)
            self.pre_commands.append(config_command)

    def __run_hdfs_cmd(self, commands, retry_times=5):
        whole_commands = copy.deepcopy(self.pre_commands)
        whole_commands.extend(commands)

        print('Running system command: {0}'.format(' '.join(whole_commands)))

        ret_code = 0
        ret_out = None
        ret_err = None
        whole_commands = " ".join(whole_commands)
        for x in range(retry_times + 1):
            proc = subprocess.Popen(
                whole_commands,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True)
            (output, errors) = proc.communicate()
            ret_code, ret_out, ret_err = proc.returncode, output, errors
            if ret_code:
                _logger.warn(
                    'Times: %d, Error running command: %s. Return code: %d, Error: %s'
                    % (x, ' '.join(whole_commands), proc.returncode, errors))
            else:
                break
        return ret_code, ret_out, ret_err

    def upload(self, hdfs_path, local_path, overwrite=False, retry_times=5):
        """
        upload the local file to hdfs

        Args:
            hdfs_path(str): the hdfs file path
            local_path(str): the local file path
            overwrite(bool|None): will overwrite the file on HDFS or not
            retry_times(int|5): retry times

        Returns:
                True or False
        """
        assert hdfs_path is not None
        assert local_path is not None and os.path.exists(local_path)

        if os.path.isdir(local_path):
            _logger.warn(
                "The Local path: {} is dir and I will support it later, return".
                format(local_path))
            return False

        base = os.path.basename(local_path)
        if not self.is_exist(hdfs_path):
            self.makedirs(hdfs_path)
        else:
            if self.is_exist(os.path.join(hdfs_path, base)):
                if overwrite:
                    _logger.error(
                        "The HDFS path: {} is exist and overwrite is True, delete it".
                        format(hdfs_path))
                    self.delete(hdfs_path)
                else:
                    _logger.error(
                        "The HDFS path: {} is exist and overwrite is False, return".
                        format(hdfs_path))
                    return False

        put_commands = ["-put", local_path, hdfs_path]
        returncode, output, errors = self.__run_hdfs_cmd(put_commands,
                                                         retry_times)
        if returncode:
            _logger.error("Put local path: {} to HDFS path: {} failed".format(
                local_path, hdfs_path))
            return False
        else:
            _logger.info("Put local path: {} to HDFS path: {} successfully".
                         format(local_path, hdfs_path))
            return True

    def download(self, hdfs_path, local_path, overwrite=False, unzip=False):
        """
        download file from HDFS

        Args:
            hdfs_path(str): the hdfs file path
            local_path(str): the local file path
            overwrite(bool|None): will overwrite the file on HDFS or not
            unzip(bool|False): if the download file is compressed by zip, unzip it or not.

        Returns:
            True or False
        """
        _logger.info('Downloading %r to %r.', hdfs_path, local_path)
        _logger.info('Download of %s to %r complete.', hdfs_path, local_path)

        if not self.is_exist(hdfs_path):
            print("HDFS path: {} do not exist".format(hdfs_path))
            return False
        if self.is_dir(hdfs_path):
            _logger.error(
                "The HDFS path: {} is dir and I will support it later, return".
                format(hdfs_path))

        if os.path.exists(local_path):
            base = os.path.basename(hdfs_path)
            local_file = os.path.join(local_path, base)
            if os.path.exists(local_file):
                if overwrite:
                    os.remove(local_file)
                else:
                    _logger.error(
                        "The Local path: {} is exist and overwrite is False, return".
                        format(local_file))
                    return False

        self.make_local_dirs(local_path)

        download_commands = ["-get", hdfs_path, local_path]
        returncode, output, errors = self.__run_hdfs_cmd(download_commands)
        if returncode:
            _logger.error("Get local path: {} from HDFS path: {} failed".format(
                local_path, hdfs_path))
            return False
        else:
            _logger.info("Get local path: {} from HDFS path: {} successfully".
                         format(local_path, hdfs_path))
            return True

    def is_exist(self, hdfs_path=None):
        """
        whether the remote HDFS path exists

        Args:
            hdfs_path(str): the hdfs file path

        Returns:
            True or False
        """
        exist_cmd = ['-test', '-e', hdfs_path]
        returncode, output, errors = self.__run_hdfs_cmd(
            exist_cmd, retry_times=1)

        if returncode:
            _logger.error("HDFS is_exist HDFS path: {} failed".format(
                hdfs_path))
            return False
        else:
            _logger.info("HDFS is_exist HDFS path: {} successfully".format(
                hdfs_path))
            return True

    def is_dir(self, hdfs_path=None):
        """
        whether the remote HDFS path is directory

        Args:
            hdfs_path(str): the hdfs file path

        Returns:
            True or False
        """

        if not self.is_exist(hdfs_path):
            return False

        dir_cmd = ['-test', '-d', hdfs_path]
        returncode, output, errors = self.__run_hdfs_cmd(dir_cmd, retry_times=1)

        if returncode:
            _logger.error("HDFS path: {} failed is not a directory".format(
                hdfs_path))
            return False
        else:
            _logger.info("HDFS path: {} successfully is a directory".format(
                hdfs_path))
            return True

    def delete(self, hdfs_path):
        """
        Remove a file or directory from HDFS.

        whether the remote HDFS path exists

        Args:
        hdfs_path: HDFS path.

        Returns:
            True or False
            This function returns `True` if the deletion was successful and `False` if
            no file or directory previously existed at `hdfs_path`.
        """
        _logger.info('Deleting %r.', hdfs_path)

        if not self.is_exist(hdfs_path):
            _logger.warn("HDFS path: {} do not exist".format(hdfs_path))
            return True

        if self.is_dir(hdfs_path):
            del_cmd = ['-rmr', hdfs_path]
        else:
            del_cmd = ['-rm', hdfs_path]

        returncode, output, errors = self.__run_hdfs_cmd(del_cmd, retry_times=0)

        if returncode:
            _logger.error("HDFS path: {} delete files failure".format(
                hdfs_path))
            return False
        else:
            _logger.info("HDFS path: {} delete files successfully".format(
                hdfs_path))
            return True

    def rename(self, hdfs_src_path, hdfs_dst_path, overwrite=False):
        """
        Move a file or folder on HDFS.

        Args:
        hdfs_path(str): HDFS path.
        overwrite(bool|False): If the path already exists and overwrite is False, will return False.

        Returns:
            True or False
        """
        assert hdfs_src_path is not None
        assert hdfs_dst_path is not None

        if not self.is_exist(hdfs_src_path):
            _logger.info("HDFS path do not exist: {}".format(hdfs_src_path))
        if self.is_exist(hdfs_dst_path) and not overwrite:
            _logger.error("HDFS path is exist: {} and overwrite=False".format(
                hdfs_dst_path))

        rename_command = ['-mv', hdfs_src_path, hdfs_dst_path]
        returncode, output, errors = self.__run_hdfs_cmd(
            rename_command, retry_times=1)

        if returncode:
            _logger.error("HDFS rename path: {} to {} failed".format(
                hdfs_src_path, hdfs_dst_path))
            return False
        else:
            _logger.info("HDFS rename path: {} to {} successfully".format(
                hdfs_src_path, hdfs_dst_path))
            return True

    @staticmethod
    def make_local_dirs(local_path):
        """
        create a directory local, is same to mkdir
        Args:
            local_path: local path that wants to create a directory.
        """
        try:
            os.makedirs(local_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def makedirs(self, hdfs_path):
        """
        Create a remote directory, recursively if necessary.

        Args:
        hdfs_path(str): Remote path. Intermediate directories will be created appropriately.

        Returns:
            True or False
        """
        _logger.info('Creating directories to %r.', hdfs_path)
        assert hdfs_path is not None

        if self.is_exist(hdfs_path):
            _logger.error("HDFS path is exist: {}".format(hdfs_path))
            return

        mkdirs_commands = ['-mkdir', hdfs_path]
        returncode, output, errors = self.__run_hdfs_cmd(
            mkdirs_commands, retry_times=1)

        if returncode:
            _logger.error("HDFS mkdir path: {} failed".format(hdfs_path))
            return False
        else:
            _logger.error("HDFS mkdir path: {} successfully".format(hdfs_path))
            return True

    def ls(self, hdfs_path):
        """
        ls directory contents about HDFS hdfs_path

        Args:
        hdfs_path(str): Remote HDFS path will be ls.

        Returns:
            List: a contents list about hdfs_path.
        """
        assert hdfs_path is not None

        if not self.is_exist(hdfs_path):
            return []

        ls_commands = ['-ls', hdfs_path]
        returncode, output, errors = self.__run_hdfs_cmd(
            ls_commands, retry_times=1)

        if returncode:
            _logger.error("HDFS list path: {} failed".format(hdfs_path))
            return []
        else:
            _logger.info("HDFS list path: {} successfully".format(hdfs_path))

            ret_lines = []
            regex = re.compile('\s+')
            out_lines = output.strip().split("\n")
            for line in out_lines:
                re_line = regex.split(line)
                if len(re_line) == 8:
                    ret_lines.append(re_line[7])
            return ret_lines

    def lsr(self, hdfs_path, only_file=True, sort=True):
        """
        list directory contents about HDFS hdfs_path recursively

        Args:
        hdfs_path(str): Remote HDFS path.
        only_file(bool|True): will discard folders.
        sort(bool|True): will be sorted by create time.

        Returns:
            List: a contents list about hdfs_path.
        """

        def sort_by_time(v1, v2):
            v1_time = datetime.strptime(v1[1], '%Y-%m-%d %H:%M')
            v2_time = datetime.strptime(v2[1], '%Y-%m-%d %H:%M')
            return v1_time > v2_time

        assert hdfs_path is not None

        if not self.is_exist(hdfs_path):
            return []

        ls_commands = ['-lsr', hdfs_path]
        returncode, output, errors = self.__run_hdfs_cmd(
            ls_commands, retry_times=1)

        if returncode:
            _logger.error("HDFS list all files: {} failed".format(hdfs_path))
            return []
        else:
            _logger.info("HDFS list all files: {} successfully".format(
                hdfs_path))
            lines = []
            regex = re.compile('\s+')
            out_lines = output.strip().decode().split("\n")
            for line in out_lines:
                re_line = regex.split(line)
                if len(re_line) == 8:
                    if only_file and re_line[0][0] == "d":
                        continue
                    else:
                        lines.append(
                            (re_line[7], re_line[5] + " " + re_line[6]))
            if sort:
                sorted(lines, key=cmp_to_key(sort_by_time))
            ret_lines = [ret[0] for ret in lines]
            return ret_lines


def multi_download(client,
                   hdfs_path,
                   local_path,
                   trainer_id,
                   trainers,
                   multi_processes=5):
    """
    Download files from HDFS using multi process.

    Args:
        client(HDFSClient): instance of HDFSClient
        hdfs_path(str): path on hdfs
        local_path(str): path on local
        trainer_id(int): current trainer id
        trainers(int): all trainers number
        multi_processes(int|5): the download data process at the same time, default=5

    Returns:
        List:
        Download files in local folder.
    """

    def __subprocess_download(datas):
        for data in datas:
            re_path = os.path.relpath(os.path.dirname(data), hdfs_path)
            if re_path == os.curdir:
                sub_local_re_path = local_path
            else:
                sub_local_re_path = os.path.join(local_path, re_path)
            client.download(data, sub_local_re_path)

    assert isinstance(client, HDFSClient)

    client.make_local_dirs(local_path)
    _logger.info("Make local dir {} successfully".format(local_path))

    all_need_download = client.lsr(hdfs_path, sort=True)
    need_download = all_need_download[trainer_id::trainers]
    _logger.info("Get {} files From all {} files need to be download from {}".
                 format(len(need_download), len(all_need_download), hdfs_path))

    _logger.info("Start {} multi process to download datas".format(
        multi_processes))
    procs = []
    for i in range(multi_processes):
        process_datas = need_download[i::multi_processes]
        p = multiprocessing.Process(
            target=__subprocess_download, args=(process_datas, ))
        procs.append(p)
        p.start()

    # complete the processes
    for proc in procs:
        proc.join()

    _logger.info("Finish {} multi process to download datas".format(
        multi_processes))

    local_downloads = []
    for data in need_download:
        data_name = os.path.basename(data)
        re_path = os.path.relpath(os.path.dirname(data), hdfs_path)
        if re_path == os.curdir:
            local_re_path = os.path.join(local_path, data_name)
        else:
            local_re_path = os.path.join(local_path, re_path, data_name)
        local_downloads.append(local_re_path)

    return local_downloads


def getfilelist(path):
    rlist = []
    for dir, folder, file in os.walk(path):
        for i in file:
            t = os.path.join(dir, i)
            rlist.append(t)
    for r in rlist:
        print(r)


def multi_upload(client,
                 hdfs_path,
                 local_path,
                 multi_processes=5,
                 overwrite=False,
                 sync=True):
    """
    Upload files to HDFS using multi process.

    Args:
        client(HDFSClient): instance of HDFSClient
        hdfs_path(str): path on hdfs
        local_path(str): path on local
        multi_processes(int|5): the upload data process at the same time, default=5
        overwrite(bool|False): will overwrite file on HDFS or not
        sync(bool|True): upload files sync or not.

    Returns:
        None
    """

    def __subprocess_upload(datas):
        for data in datas:
            re_path = os.path.relpath(os.path.dirname(data), local_path)
            hdfs_re_path = os.path.join(hdfs_path, re_path)
            client.upload(hdfs_re_path, data, overwrite, retry_times=5)

    def get_local_files(path):
        rlist = []

        if not os.path.isdir(path):
            return rlist

        for dirname, folder, files in os.walk(path):
            for i in files:
                t = os.path.join(dirname, i)
                rlist.append(t)
        return rlist

    assert isinstance(client, HDFSClient)

    all_files = get_local_files(local_path)
    if not all_files:
        _logger.info("there are nothing need to upload, exit")
        return
    _logger.info("Start {} multi process to upload datas".format(
        multi_processes))
    procs = []
    for i in range(multi_processes):
        process_datas = all_files[i::multi_processes]
        p = multiprocessing.Process(
            target=__subprocess_upload, args=(process_datas, ))
        procs.append(p)
        p.start()

    # complete the processes
    for proc in procs:
        proc.join()

    _logger.info("Finish {} multi process to upload datas".format(
        multi_processes))


if __name__ == "__main__":
    hadoop_home = "/home/client/hadoop-client/hadoop/"

    configs = {
        "fs.default.name": "hdfs://xxx.hadoop.com:54310",
        "hadoop.job.ugi": "hello,hello123"
    }

    client = HDFSClient(hadoop_home, configs)

    client.ls("/user/com/train-25")
    files = client.lsr("/user/com/train-25/models")

    downloads = multi_download(
        client,
        "/user/com/train-25/model",
        "/home/xx/data1",
        1,
        5,
        100,
        multi_processes=5)

    multi_upload(client, "/user/com/train-25/model", "/home/xx/data1")
