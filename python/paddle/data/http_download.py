# /usr/bin/env python
# -*- coding:utf-8 -*-

# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

########################################################################
#
# Funciton for data download,it use the urllib urlretrieve and we can
# see the download process when downloading the source.
#
# download process like: - Download progress:10%
#
########################################################################

import os
import sys
import shutil
import zipfile
import tarfile
import stat
from six.moves import urllib


def download_with_urlretrieve(url, filename=None, with_progress=True):
    """
    Download each file with urlretrieve,and the download process can be seen.

    :param url: the url for data download.
    :type url: basestring
    :param filename: Output file name. None if use default file name.
    :type filename: basestring
    :param with_progress: with progress bar or not. Default is true.
    :type with_progress: bool
    :return: the downloaded filename
    :rtype: basestring
    """

    def check_download_progress(count, block_size, total_size):
        percent = float(count * block_size) / total_size
        msg = "\r- Downloading {1} progress: {0:.1%}".format(percent, filename)
        sys.stdout.write(msg)
        sys.stdout.flush()

    hook = None
    if with_progress:
        hook = check_download_progress

    return urllib.request.urlretrieve(url, filename, reporthook=hook)[0]


def data_download(download_dir, source_url):
    """
    Download data according to the url for source_name.
    when downloading,it can see each download process.

    Args:
        download_dir:the directory for data download.
        source_url:the url for data download.

    Returns:
        the path after data downloaded.
    """
    src_file = source_url.strip().split('/')[-1]
    file_path = os.path.join(download_dir, src_file)

    print file_path
    if not os.path.exists(file_path):
        temp_file_name = download_with_urlretrieve(source_url)
        temp_file_path = os.getcwd()
        os.rename(temp_file_name, src_file)
        shutil.move(src_file, download_dir)
        print("Download finished, Extracting files.")

        if 'zip' in src_file:
            tar = zipfile.ZipFile(file_path, 'r')
            infos = tar.infolist()
            for file in infos:
                tar.extract(file, download_dir)
                fpath = os.path.join(download_dir, file.filename)
                if 'master' in src_file:
                    os.chmod(fpath, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
            os.remove(file_path)
        elif src_file in ['.json.gz', 'txt', 'emb', 'python.tar.gz']:
            pass
        elif src_file.split('.')[-1] is 'gz':
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)
            os.remove(file_path)
        print("Unpacking done!")
    else:
        if 'zip' in src_file:
            tar = zipfile.ZipFile(file_path, 'r')
            infos = tar.infolist()
            for file in infos:
                tar.extract(file, download_dir)
                fpath = os.path.join(download_dir, file.filename)
                if 'master' in src_file:
                    os.chmod(fpath, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
            os.remove(file_path)
        elif src_file in ['.json.gz', 'txt', 'emb']:
            pass
        elif src_file.split('.')[-1] is 'gz':
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)
            os.remove(file_path)
        print("Data has been already downloaded and unpacked!")

    return download_dir


download = download_with_urlretrieve
