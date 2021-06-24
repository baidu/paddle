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

import os
import sys
import subprocess
import multiprocessing
from datetime import datetime

import re
import copy
import errno
import time
import logging
import six
import abc
import paddle.fluid as fluid
from paddle.fluid import core
import functools

import shutil

__all__ = []


class ExecuteError(Exception):
    pass


class FSFileExistsError(Exception):
    pass


class FSFileNotExistsError(Exception):
    pass


class FSTimeOut(Exception):
    pass


class FSShellCmdAborted(ExecuteError):
    pass


class FS(object):
    @abc.abstractmethod
    def ls_dir(self, fs_path):
        raise NotImplementedError

    @abc.abstractmethod
    def is_file(self, fs_path):
        raise NotImplementedError

    @abc.abstractmethod
    def is_dir(self, fs_path):
        raise NotImplementedError

    @abc.abstractmethod
    def is_exist(self, fs_path):
        raise NotImplementedError

    @abc.abstractmethod
    def upload(self, local_path, fs_path):
        raise NotImplementedError

    @abc.abstractmethod
    def download(self, fs_path, local_path):
        raise NotImplementedError

    @abc.abstractmethod
    def mkdirs(self, fs_path):
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, fs_path):
        raise NotImplementedError

    @abc.abstractmethod
    def need_upload_download(self):
        raise NotImplementedError

    @abc.abstractmethod
    def rename(self, fs_src_path, fs_dst_path):
        raise NotImplementedError

    @abc.abstractmethod
    def mv(self, fs_src_path, fs_dst_path, overwrite=False, test_exists=False):
        raise NotImplementedError

    @abc.abstractmethod
    def upload_dir(self, local_dir, dest_dir):
        raise NotImplementedError

    @abc.abstractmethod
    def list_dirs(self, fs_path):
        raise NotImplementedError

    @abc.abstractmethod
    def touch(self, fs_path, exist_ok=True):
        raise NotImplementedError


class LocalFS(FS):
    """
    A tool of local file system.

    Examples:
        .. code-block:: python

            from paddle.distributed.fleet.utils import LocalFS

            client = LocalFS()
            subdirs, files = client.ls_dir("./")
    """

    def ls_dir(self, fs_path):
        """	
        List directorys and files under `fs_path` .

        Args:
            fs_path(str): The local file path.

        Returns:
            Tuple: Return a 2-tuple, the first is a list of all its subdirectories, 
            and the second is a list of all its subfiles, e.g. ([subdirname1, subdirname1, ...], [filename1, filename2, ...]).

        Examples:
            .. code-block:: python

                from paddle.distributed.fleet.utils import LocalFS

                client = LocalFS()
                subdirs, files = client.ls_dir("./")
        """
        if not self.is_exist(fs_path):
            return [], []

        dirs = []
        files = []
        for f in os.listdir(fs_path):
            if os.path.isdir(fs_path + "/" + f):
                dirs.append(f)
            else:
                files.append(f)

        return dirs, files

    def mkdirs(self, fs_path):
        """
        Create a local directory.

        Args:
            fs_path(str): The local directory path.

        Examples:
            .. code-block:: python

                from paddle.distributed.fleet.utils import LocalFS

                client = LocalFS()
                client.mkdirs("test_mkdirs")
                client.delete("test_mkdirs")
        """
        assert not os.path.isfile(fs_path), "{} is already a file".format(
            fs_path)
        os.system("mkdir -p {}".format(fs_path))

    def rename(self, fs_src_path, fs_dst_path):
        """
        Rename the file.

        Args:
            fs_src_path(str): The actual name of the file or directory
            fs_dst_path(str): The new name of the file or directory.

        Examples:
            .. code-block:: python

                from paddle.distributed.fleet.utils import LocalFS

                client = LocalFS()
                client.touch("test_rename_src")
                print(client.is_exists("test_rename_src")) # True
                client.rename("test_rename_src", "test_rename_dst")
                print(client.is_exists("test_rename_src")) # False
                print(client.is_exists("test_rename_dst")) # True
                client.delete("test_rename_dst")
        """
        os.rename(fs_src_path, fs_dst_path)

    def _rmr(self, fs_path):
        shutil.rmtree(fs_path)

    def _rm(self, fs_path):
        os.remove(fs_path)

    def delete(self, fs_path):
        """
        Delete the local file path, whether it's a file or directory.

        Args:
            fs_path(str): The local file path.

        Examples:
            .. code-block:: python

                from paddle.distributed.fleet.utils import LocalFS

                client = LocalFS()
                client.mkdirs("test_localFS_mkdirs")
                client.delete("test_localFS_mkdirs")
        """
        if not self.is_exist(fs_path):
            return

        if os.path.isfile(fs_path):
            return self._rm(fs_path)

        return self._rmr(fs_path)

    def need_upload_download(self):
        return False

    def is_file(self, fs_path):
        """
        Whether the local file path is a file.

        Args:
            fs_path(str): The local file path.

        Returns:
            Bool: Return true if the path exists and it's a file, otherwise return false.

        Examples:
            .. code-block:: python

                from paddle.distributed.fleet.utils import LocalFS

                client = LocalFS()
                client.touch("test_is_file")
                print(client.is_file("test_is_file")) # True
                client.delete("test_is_file")
        """
        return os.path.isfile(fs_path)

    def is_dir(self, fs_path):
        """
        Whether the local file path is a directory.

        Args:
            fs_path(str): The local file path.

        Returns:
            Bool: Return true if the path exists and it's a directory, otherwise return false.

        Examples:
            .. code-block:: python

                from paddle.distributed.fleet.utils import LocalFS

                client = LocalFS()
                client.mkdirs("test_is_dir")
                print(client.is_dir("test_is_file")) # True
                client.delete("test_is_dir")
        """
        return os.path.isdir(fs_path)

    def is_exist(self, fs_path):
        """
        Whether the local file path exists.

        Args:
            fs_path(str): The local file path.

        Returns:
            Bool: Wheter it's a file or directory, return true if the path exists, 
            otherwise return false.

        Examples:
            .. code-block:: python

                from paddle.distributed.fleet.utils import LocalFS

                client = LocalFS()
                ret = local_fs.is_exist("test_is_exist")
        """
        return os.path.exists(fs_path)

    def touch(self, fs_path, exist_ok=True):
        """
        Create a local file.

        Args:
            fs_path(str): The local file path.
            exist_ok(bool): When `fs_path` exists, if `exist_ok` is set false,
            program will throw an Exception. Default is true.

        Examples:
            .. code-block:: python

                from paddle.distributed.fleet.utils import LocalFS

                client = LocalFS()
                client.touch("test_touch")
                client.delete("test_touch")
        """
        if self.is_exist(fs_path):
            if exist_ok:
                return
            raise FSFileExistsError

        os.system("touch {}".format(fs_path))

    def mv(self, src_path, dst_path, overwrite=False, test_exists=False):
        """
        Move a local file or directory from `src_path` to `dst_path` .

        Args:
            src_path(str):  Name of the file or directory, that's needed to be moved.
            dst_path(str):  Name of the file or directory to which to move to.
            overwrite(bool): Whether to re-write `dst_path` if that exists. Default is False.

        Examples:
            .. code-block:: python

                from paddle.distributed.fleet.utils import LocalFS

                client = LocalFS()
                client.touch("test_mv_src")
                client.mv("test_mv_src", "test_mv_dst")
                client.delete("test_mv_dst")
        """
        if not self.is_exist(src_path):
            raise FSFileNotExistsError

        if overwrite and self.is_exist(dst_path):
            self.delete(dst_path)

        if self.is_exist(dst_path):
            raise FSFileExistsError

        return self.rename(src_path, dst_path)

    def list_dirs(self, fs_path):
        """	
        Only list directorys under `fs_path` .

        Args:
            fs_path(str): The local file path.

        Returns:
            List: A list of all its subdirectories, e.g. [subdirname1, subdirname1, ...].

        Examples:
            .. code-block:: python

                from paddle.distributed.fleet.utils import LocalFS

                client = LocalFS()
                subdirs = client.list_dirs("./")
        """
        if not self.is_exist(fs_path):
            return []

        dirs = [
            f for f in os.listdir(fs_path) if os.path.isdir(fs_path + "/" + f)
        ]

        return dirs


def _handle_errors(max_time_out=None):
    def decorator(f):
        @functools.wraps(f)
        def handler(*args, **kwargs):
            o = args[0]
            time_out = max_time_out
            if time_out is None:
                time_out = float(o._time_out) / 1000.0
            else:
                time_out /= 1000.0
            inter = float(o._sleep_inter) / 1000.0

            start = time.time()
            last_print_time = start
            while True:
                try:
                    return f(*args, **kwargs)
                # important: only ExecuteError need to retry
                except ExecuteError as e:
                    if time.time() - start >= time_out:
                        raise FSTimeOut("args:{} timeout:{}".format(
                            args, time.time() - start))

                    time.sleep(inter)

                if time.time() - last_print_time > 30:
                    print("hadoop operator timeout:args:{} timeout:{}".format(
                        args, time.time() - start))
                    last_print_time = time.time()

        return handler

    return decorator


class HDFSClient(FS):
    """
    A tool of HDFS.

    Args:
        hadoop_home(str): Hadoop home. 
        configs(dict): Hadoop config. It is a dictionary and needs to contain the
            keys: "fs.default.name" and "hadoop.job.ugi".

    Examples:

        .. code-block:: text

            from paddle.distributed.fleet.utils import HDFSClient
            hadoop_home = "/home/client/hadoop-client/hadoop/"

            configs = {
                "fs.default.name": "hdfs://xxx.hadoop.com:54310",
                "hadoop.job.ugi": "hello,hello123"
            }

            client = HDFSClient(hadoop_home, configs)
            client.ls_dir("hdfs:/test_hdfs_client")
    """

    def __init__(
            self,
            hadoop_home,
            configs,
            time_out=5 * 60 * 1000,  # ms
            sleep_inter=1000):  # ms
        self.pre_commands = []
        hadoop_bin = '%s/bin/hadoop' % hadoop_home
        self.pre_commands.append(hadoop_bin)
        dfs = 'fs'
        self.pre_commands.append(dfs)

        if configs:
            for k, v in six.iteritems(configs):
                config_command = '-D%s=%s' % (k, v)
                self.pre_commands.append(config_command)

        self._time_out = time_out
        self._sleep_inter = sleep_inter
        self._base_cmd = " ".join(self.pre_commands)
        self._bd_err_re = re.compile(
            r'\s?responseErrorMsg\s?\:.*, errorCode\:\s?[0-9]+, path\:')

    def _run_cmd(self, cmd, redirect_stderr=False):
        exe_cmd = "{} -{}".format(self._base_cmd, cmd)
        ret, output = core.shell_execute_cmd(exe_cmd, 0, 0, redirect_stderr)
        ret = int(ret)
        if ret == 134:
            raise FSShellCmdAborted(cmd)
        return ret, output.splitlines()

    @_handle_errors()
    def list_dirs(self, fs_path):
        """	
        Only list directorys under `fs_path` .

        Args:
            fs_path(str): The HDFS file path.

        Returns:
            List: A list of all its subdirectories, e.g. [subdirname1, subdirname1, ...].

        Examples:

            .. code-block:: text

                from paddle.distributed.fleet.utils import HDFSClient

                hadoop_home = "/home/client/hadoop-client/hadoop/"
                configs = {
                    "fs.default.name": "hdfs://xxx.hadoop.com:54310",
                    "hadoop.job.ugi": "hello,hello123"
                }

                client = HDFSClient(hadoop_home, configs)
                subdirs = client.list_dirs("hdfs:/test_hdfs_client")
        """
        if not self.is_exist(fs_path):
            return []

        dirs, files = self._ls_dir(fs_path)
        return dirs

    @_handle_errors()
    def ls_dir(self, fs_path):
        """	
        List directorys and files under `fs_path` .

        Args:
            fs_path(str): The HDFS file path.

        Returns:
            Tuple: Return a 2-tuple, the first element is the list of all its subdirectories, 
            and the second one is the list of all its subfiles, e.g. ([subdirname1, subdirname1, ...], [filename1, filename2, ...]).

        Examples:

            .. code-block:: text

                from paddle.distributed.fleet.utils import HDFSClient

                hadoop_home = "/home/client/hadoop-client/hadoop/"
                configs = {
                    "fs.default.name": "hdfs://xxx.hadoop.com:54310",
                    "hadoop.job.ugi": "hello,hello123"
                }

                client = HDFSClient(hadoop_home, configs)
                subdirs, files = client.ls_dir("hdfs:/test_hdfs_client")
        """
        if not self.is_exist(fs_path):
            return [], []

        return self._ls_dir(fs_path)

    def _ls_dir(self, fs_path):
        cmd = "ls {}".format(fs_path)
        ret, lines = self._run_cmd(cmd)

        if ret != 0:
            raise ExecuteError(cmd)

        dirs = []
        files = []
        for line in lines:
            arr = line.split()
            if len(arr) != 8:
                continue

            p = os.path.basename(arr[7])
            if arr[0][0] == 'd':
                dirs.append(p)
            else:
                files.append(p)

        return dirs, files

    def _test_match(self, lines):
        for l in lines:
            m = self._bd_err_re.match(l)
            if m != None:
                return m

        return None

    @_handle_errors()
    def is_dir(self, fs_path):
        """
        Whether the remote HDFS path is a directory.

        Args:
            fs_path(str): The HDFS file path.

        Returns:
            Bool: Return true if the path exists and it's a directory, otherwise return false.

        Examples:

            .. code-block:: text

                from paddle.distributed.fleet.utils import HDFSClient

                hadoop_home = "/home/client/hadoop-client/hadoop/"
                configs = {
                    "fs.default.name": "hdfs://xxx.hadoop.com:54310",
                    "hadoop.job.ugi": "hello,hello123"
                }

                client = HDFSClient(hadoop_home, configs)
                ret = client.is_file("hdfs:/test_hdfs_client")
        """
        if not self.is_exist(fs_path):
            return False

        return self._is_dir(fs_path)

    def _is_dir(self, fs_path):
        cmd = "test -d {}".format(fs_path, redirect_stderr=True)
        ret, lines = self._run_cmd(cmd)
        if ret:
            # other error
            if self._test_match(lines):
                raise ExecuteError(cmd)

            return False

        return True

    def is_file(self, fs_path):
        """
        Whether the remote HDFS path is a file.

        Args:
            fs_path(str): The HDFS file path.

        Returns:
            Bool: Return true if the path exists and it's a file, otherwise return false.

        Examples:

            .. code-block:: text

                from paddle.distributed.fleet.utils import HDFSClient

                hadoop_home = "/home/client/hadoop-client/hadoop/"
                configs = {
                    "fs.default.name": "hdfs://xxx.hadoop.com:54310",
                    "hadoop.job.ugi": "hello,hello123"
                }

                client = HDFSClient(hadoop_home, configs)
                ret = client.is_file("hdfs:/test_hdfs_client")
        """
        if not self.is_exist(fs_path):
            return False

        return not self._is_dir(fs_path)

    @_handle_errors()
    def is_exist(self, fs_path):
        """
        Whether the remote HDFS path exists.

        Args:
            fs_path(str): The hdfs file path.

        Returns:
            Bool: Whether it's is file or directory, return true if the path exists,
            otherwise return false.

        Examples:

            .. code-block:: text

                from paddle.distributed.fleet.utils import HDFSClient

                hadoop_home = "/home/client/hadoop-client/hadoop/"
                configs = {
                    "fs.default.name": "hdfs://xxx.hadoop.com:54310",
                    "hadoop.job.ugi": "hello,hello123"
                }

                client = HDFSClient(hadoop_home, configs)
                ret = client.is_exist("hdfs:/test_hdfs_client")
        """
        cmd = "ls {} ".format(fs_path)
        ret, out = self._run_cmd(cmd, redirect_stderr=True)
        if ret != 0:
            for l in out:
                if "No such file or directory" in l:
                    return False
            raise ExecuteError(cmd)

        return True

    # can't retry
    def upload(self, local_path, fs_path):
        """
        Upload the local path to remote HDFS.

        Args:
            local_path(str): The local path.
            fs_path(str): The HDFS path.

        Examples:

            .. code-block:: text

                from paddle.distributed.fleet.utils import HDFSClient

                hadoop_home = "/home/client/hadoop-client/hadoop/"
                configs = {
                    "fs.default.name": "hdfs://xxx.hadoop.com:54310",
                    "hadoop.job.ugi": "hello,hello123"
                }

                client = HDFSClient(hadoop_home, configs)
                client.upload("test_hdfs_client", "hdfs:/test_hdfs_client")
        """
        if self.is_exist(fs_path):
            raise FSFileExistsError("{} exists".format(fs_path))

        local = LocalFS()
        if not local.is_exist(local_path):
            raise FSFileNotExistsError("{} not exists".format(local_path))

        return self._try_upload(local_path, fs_path)

    @_handle_errors()
    def _try_upload(self, local_path, fs_path):
        cmd = "put {} {}".format(local_path, fs_path)
        ret = 0
        try:
            ret, lines = self._run_cmd(cmd)
            if ret != 0:
                raise ExecuteError(cmd)
        except Exception as e:
            self.delete(fs_path)
            raise e

    # can't retry
    def download(self, fs_path, local_path):
        """
        Download remote HDFS path to the local.

        Args:
            fs_path(str):  The HDFS path.
            local_path(str): The local path.

        Examples:

            .. code-block:: text

                from paddle.distributed.fleet.utils import HDFSClient

                hadoop_home = "/home/client/hadoop-client/hadoop/"
                configs = {
                    "fs.default.name": "hdfs://xxx.hadoop.com:54310",
                    "hadoop.job.ugi": "hello,hello123"
                }

                client = HDFSClient(hadoop_home, configs)
                client.download("hdfs:/test_hdfs_client", "./")
        """
        if not self.is_exist(fs_path):
            raise FSFileNotExistsError("{} not exits".format(fs_path))

        return self._try_download(fs_path, local_path)

    @_handle_errors()
    def _try_download(self, fs_path, local_path):
        cmd = "get {} {}".format(fs_path, local_path)
        ret = 0
        try:
            ret, lines = self._run_cmd(cmd)
            if ret != 0:
                raise ExecuteError(cmd)
        except Exception as e:
            local_fs = LocalFS()
            local_fs.delete(local_path)
            raise e

    @_handle_errors()
    def mkdirs(self, fs_path):
        """
        Create a remote HDFS directory.

        Args:
            fs_path(str): The HDFS directory path.

        Examples:

            .. code-block:: text

                from paddle.distributed.fleet.utils import HDFSClient

                hadoop_home = "/home/client/hadoop-client/hadoop/"
                configs = {
                    "fs.default.name": "hdfs://xxx.hadoop.com:54310",
                    "hadoop.job.ugi": "hello,hello123"
                }

                client = HDFSClient(hadoop_home, configs)
                client.mkdirs("hdfs:/test_hdfs_client")
        """
        if self.is_exist(fs_path):
            return

        out_hdfs = False

        cmd = "mkdir {} ".format(fs_path)
        ret, out = self._run_cmd(cmd, redirect_stderr=True)
        if ret != 0:
            for l in out:
                if "No such file or directory" in l:
                    out_hdfs = True
                    break
            if not out_hdfs:
                raise ExecuteError(cmd)

        if out_hdfs and not self.is_exist(fs_path):
            cmd = "mkdir -p {}".format(fs_path)
            ret, lines = self._run_cmd(cmd)
            if ret != 0:
                raise ExecuteError(cmd)

    def mv(self, fs_src_path, fs_dst_path, overwrite=False, test_exists=True):
        """
        Move a remote HDFS file or directory from `fs_src_path` to `fs_dst_path` .

        Args:
            fs_src_path(str):  Name of the file or directory, that's needed to be moved.
            fs_dst_path(str):  Name of the file or directory to which to move to.
            overwrite(bool): Whether to re-write `fs_dst_path` if that exists. Default is False.
            test_exists(bool): Check the existence of `fs_src_path` and `fs_dst_path` . When `test_exists` is set true, if `fs_src_path` doesn't exist or `fs_dst_path` exists, program will throw an Excetption. 

        Examples:

            .. code-block:: text

                from paddle.distributed.fleet.utils import HDFSClient

                hadoop_home = "/home/client/hadoop-client/hadoop/"
                configs = {
                    "fs.default.name": "hdfs://xxx.hadoop.com:54310",
                    "hadoop.job.ugi": "hello,hello123"
                }

                client = HDFSClient(hadoop_home, configs)
                client.mv("hdfs:/test_hdfs_client", "hdfs:/test_hdfs_client2")
        """
        if overwrite and self.is_exist(fs_dst_path):
            self.delete(fs_dst_path)

        if test_exists:
            if not self.is_exist(fs_src_path):
                raise FSFileNotExistsError("{} is not exists".format(
                    fs_src_path))

            if self.is_exist(fs_dst_path):
                raise FSFileExistsError("{} exists already".format(fs_dst_path))

        return self._try_mv(fs_src_path, fs_dst_path)

    @_handle_errors()
    def _try_mv(self, fs_src_path, fs_dst_path):
        cmd = "mv {} {}".format(fs_src_path, fs_dst_path)
        ret = 0
        try:
            ret, _ = self._run_cmd(cmd)
            if ret != 0:
                raise ExecuteError(cmd)
        except Exception as e:
            if not self.is_exist(fs_src_path) and \
                    self.is_exist(fs_dst_path):
                return
            raise e

    def _rmr(self, fs_path):
        cmd = "rmr {}".format(fs_path)
        ret, _ = self._run_cmd(cmd)
        if ret != 0:
            raise ExecuteError(cmd)

    def _rm(self, fs_path):
        cmd = "rm {}".format(fs_path)
        ret, _ = self._run_cmd(cmd)
        if ret != 0:
            raise ExecuteError(cmd)

    @_handle_errors()
    def delete(self, fs_path):
        """
        Delete a remote HDFS path, whether it's a file or directory.

        Args:
            fs_path(str): The HDFS file path.

        Examples:

            .. code-block:: text

                from paddle.distributed.fleet.utils import HDFSClient

                hadoop_home = "/home/client/hadoop-client/hadoop/"
                configs = {
                    "fs.default.name": "hdfs://xxx.hadoop.com:54310",
                    "hadoop.job.ugi": "hello,hello123"
                }

                client = HDFSClient(hadoop_home, configs)
                client.delete("hdfs:/test_hdfs_client")
        """
        if not self.is_exist(fs_path):
            return

        is_dir = self._is_dir(fs_path)
        if is_dir:
            return self._rmr(fs_path)

        return self._rm(fs_path)

    def touch(self, fs_path, exist_ok=True):
        """
        Create a remote HDFS file.

        Args:
            fs_path(str): The HDFS file path.
            exist_ok(bool): When `fs_path` exists, if `exist_ok` is set false,
            program will throw an Exception. Default is true.

        Examples:

            .. code-block:: text

                from paddle.distributed.fleet.utils import HDFSClient

                hadoop_home = "/home/client/hadoop-client/hadoop/"
                configs = {
                    "fs.default.name": "hdfs://xxx.hadoop.com:54310",
                    "hadoop.job.ugi": "hello,hello123"
                }

                client = HDFSClient(hadoop_home, configs)
                client.touch("hdfs:/test_hdfs_client")
        """
        if self.is_exist(fs_path):
            if exist_ok:
                return
            raise FSFileExistsError

        return self._touchz(fs_path)

    @_handle_errors()
    def _touchz(self, fs_path):
        cmd = "touchz {}".format(fs_path)
        ret, _ = self._run_cmd(cmd)
        if ret != 0:
            raise ExecuteError

    def need_upload_download(self):
        return True
