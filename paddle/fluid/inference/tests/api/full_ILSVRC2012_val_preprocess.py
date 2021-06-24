#   copyright (c) 2019 paddlepaddle authors. all rights reserved.
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.
import hashlib
import unittest
import os
import io
import numpy as np
import time
import sys
import random
import functools
import contextlib
from PIL import Image
import math
from paddle.dataset.common import download
import tarfile
import argparse
import shutil

np.random.seed(0)

DATA_DIM = 224
SIZE_FLOAT32 = 4
SIZE_INT64 = 8
FULL_SIZE_BYTES = 30106000008
FULL_IMAGES = 50000
FOLDER_NAME = "ILSVRC2012/"
VALLIST_TAR_NAME = "ILSVRC2012/val_list.txt"

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def process_image(img):
    img = resize_short(img, target_size=256)
    img = crop_image(img, target_size=DATA_DIM, center=True)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std
    return img


def download_concat(cache_folder, zip_path):
    data_urls = []
    data_md5s = []
    data_urls.append(
        'https://paddle-inference-dist.bj.bcebos.com/int8/ILSVRC2012_img_val.tar.gz.partaa'
    )
    data_md5s.append('60f6525b0e1d127f345641d75d41f0a8')
    data_urls.append(
        'https://paddle-inference-dist.bj.bcebos.com/int8/ILSVRC2012_img_val.tar.gz.partab'
    )
    data_md5s.append('1e9f15f64e015e58d6f9ec3210ed18b5')
    file_names = []
    print("Downloading full ImageNet Validation dataset ...")
    for i in range(0, len(data_urls)):
        download(data_urls[i], cache_folder, data_md5s[i])
        file_name = os.path.join(cache_folder, data_urls[i].split('/')[-1])
        file_names.append(file_name)
        print("Downloaded part {0}\n".format(file_name))
    with open(zip_path, "wb") as outfile:
        for fname in file_names:
            shutil.copyfileobj(open(fname, 'rb'), outfile)


def print_processbar(done_percentage):
    done_filled = done_percentage * '='
    empty_filled = (100 - done_percentage) * ' '
    sys.stdout.write("\r[%s%s]%d%%" %
                     (done_filled, empty_filled, done_percentage))
    sys.stdout.flush()


def convert_Imagenet_tar2bin(tar_file, output_file):
    print('Converting 50000 images to binary file ...\n')
    tar = tarfile.open(name=tar_file, mode='r:gz')

    print_processbar(0)

    dataset = {}
    for tarInfo in tar:
        if tarInfo.isfile() and tarInfo.name != VALLIST_TAR_NAME:
            dataset[tarInfo.name] = tar.extractfile(tarInfo).read()
    with open(output_file, "w+b") as ofs:
        ofs.seek(0)
        num = np.array(int(FULL_IMAGES)).astype('int64')
        ofs.write(num.tobytes())

        per_percentage = FULL_IMAGES // 100

        val_info = tar.getmember(VALLIST_TAR_NAME)
        val_list = tar.extractfile(val_info).read().decode("utf-8")
        lines = val_list.splitlines()
        idx = 0
        for imagedata in dataset.values():
            img = Image.open(io.BytesIO(imagedata))
            img = process_image(img)
            np_img = np.array(img)
            ofs.write(np_img.astype('float32').tobytes())
            if idx % per_percentage == 0:
                print_processbar(idx // per_percentage)
            idx = idx + 1

        val_dict = {}
        for line_idx, line in enumerate(lines):
            if line_idx == FULL_IMAGES:
                break
            name, label = line.split()
            val_dict[name] = label

        for img_name in dataset.keys():
            remove_len = (len(FOLDER_NAME))
            img_name_prim = img_name[remove_len:]
            label = val_dict[img_name_prim]
            label_int = (int)(label)
            np_label = np.array(label_int)
            ofs.write(np_label.astype('int64').tobytes())
        print_processbar(100)
    tar.close()
    print("Conversion finished.")


def run_convert():
    print('Start to download and convert 50000 images to binary file...')
    cache_folder = os.path.expanduser('~/.cache/paddle/dataset/int8/download')
    zip_path = os.path.join(cache_folder, 'full_imagenet_val.tar.gz.partaa')
    output_file = os.path.join(cache_folder, 'int8_full_val.bin')
    retry = 0
    try_limit = 3

    while not (os.path.exists(output_file) and
               os.path.getsize(output_file) == FULL_SIZE_BYTES):
        if os.path.exists(output_file):
            sys.stderr.write(
                "\n\nThe existing binary file[{}] is broken. Start to generate new one...\n\n".
                format(output_file))
            os.remove(output_file)
        if retry < try_limit:
            retry = retry + 1
        else:
            raise RuntimeError(
                "Can not convert the dataset to binary file with try limit {0}".
                format(try_limit))
        download_concat(cache_folder, zip_path)
        convert_Imagenet_tar2bin(zip_path, output_file)
    print("\nSuccess! The binary file can be found at {0}".format(output_file))


def convert_Imagenet_local2bin(args):
    data_dir = args.data_dir
    label_list_path = os.path.join(args.data_dir, args.label_list)
    bin_file_path = os.path.join(args.data_dir, args.output_file)
    assert data_dir, 'Once set --local, user need to provide the --data_dir'
    with open(label_list_path) as flist:
        lines = [line.strip() for line in flist]
        num_images = len(lines)

        with open(bin_file_path, "w+b") as of:
            of.seek(0)
            num = np.array(int(num_images)).astype('int64')
            of.write(num.tobytes())
            for idx, line in enumerate(lines):
                img_path, label = line.split()
                img_path = os.path.join(data_dir, img_path)
                if not os.path.exists(img_path):
                    continue

                #save image(float32) to file
                img = Image.open(img_path)
                img = process_image(img)
                np_img = np.array(img)
                of.seek(SIZE_INT64 + SIZE_FLOAT32 * DATA_DIM * DATA_DIM * 3 *
                        idx)
                of.write(np_img.astype('float32').tobytes())

                #save label(int64_t) to file
                label_int = (int)(label)
                np_label = np.array(label_int)
                of.seek(SIZE_INT64 + SIZE_FLOAT32 * DATA_DIM * DATA_DIM * 3 *
                        num_images + idx * SIZE_INT64)
                of.write(np_label.astype('int64').tobytes())

        # The bin file should contain
        # number of images + all images data + all corresponding labels
        # so the file target_size should be as follows
        target_size = SIZE_INT64 + num_images * 3 * args.data_dim * args.data_dim * SIZE_FLOAT32 + num_images * SIZE_INT64
        if (os.path.getsize(bin_file_path) == target_size):
            print(
                "Success! The user data output binary file can be found at: {0}".
                format(bin_file_path))
        else:
            print("Conversion failed!")


def main_preprocess_Imagenet(args):
    parser = argparse.ArgumentParser(
        description="Convert the full Imagenet val set or local data to binary file.",
        usage=None,
        add_help=True)
    parser.add_argument(
        '--local',
        action="store_true",
        help="If used, user need to set --data_dir and then convert file")
    parser.add_argument(
        "--data_dir", default="", type=str, help="Dataset root directory")
    parser.add_argument(
        "--label_list",
        type=str,
        default="val_list.txt",
        help="List of object labels with same sequence as denoted in the annotation file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="imagenet_small.bin",
        help="File path of the output binary file")
    parser.add_argument(
        "--data_dim",
        type=int,
        default=DATA_DIM,
        help="Image preprocess with data_dim width and height")

    args = parser.parse_args()
    if args.local:
        convert_Imagenet_local2bin(args)
    else:
        run_convert()


if __name__ == '__main__':
    main_preprocess_Imagenet(sys.argv)
