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

import math
import os
import pickle
import shutil
import subprocess
import multiprocessing
import sys


class SampleCodeCtx:
    def __init__(self, start_line, end_line, content, file_name):
        self.start_line = start_line
        self.end_line = end_line
        self.content = content
        self.file_name = file_name


class SampleCodeGenerator:
    def __init__(self, root_path):
        self.code_blocks = []
        # not extract dirs or files
        self.white_list = [
            "python/paddle/distributed", "python/paddle/fluid/incubate"
        ]
        self.error_blocks = []
        self.root_path = os.path.abspath(root_path)

    def filter(self, file_path):
        for f in self.white_list:
            wf = os.path.join(self.root_path, f)
            if wf == file_path or file_path.startswith(wf):
                return True
        return False

    def _check_indent(self, code_line):
        indent = ""
        for c in code_line:
            if c == '\t':
                indent += '    '
            elif c == ' ':
                indent += ' '
            if c != ' ' and c != '\t':
                break
        return indent

    def _find_all(self, src_str, substr):
        indices = []
        get_one = src_str.find(substr)
        while get_one != -1:
            indices.append(get_one)
            get_one = src_str.find(substr, get_one + 1)
        return indices

    def extract_sample_code(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as srcfile:
            filename = srcfile.name
            srcc = srcfile.read()
            srcfile.seek(0, 0)
            srcls = srcfile.readlines()
            sample_code_begins = self._find_all(srcc, " code-block:: python")
            if len(sample_code_begins) == 0:
                return

            for i in range(0, len(srcls)):
                if srcls[i].find(".. code-block:: python") != -1:
                    content = ""
                    start = i

                    blank_line = 1
                    while srcls[start + blank_line].strip() == '':
                        blank_line += 1

                    startindent = ""
                    # remove indent error
                    if srcls[start + blank_line].find("from") != -1:
                        startindent += srcls[start + blank_line][:srcls[
                            start + blank_line].find("from")]
                    elif srcls[start + blank_line].find("import") != -1:
                        startindent += srcls[start + blank_line][:srcls[
                            start + blank_line].find("import")]
                    else:
                        startindent += self._check_indent(srcls[start +
                                                                blank_line])
                    content += srcls[start + blank_line][len(startindent):]
                    for j in range(start + blank_line + 1, len(srcls)):
                        # planish a blank line
                        if not srcls[j].startswith(startindent) and srcls[
                                j] != '\n':
                            break
                        if srcls[j].find(" code-block:: python") != -1:
                            break
                        content += srcls[j].replace(startindent, "", 1)

                    ctx = SampleCodeCtx(i, j, content, filename)
                    self.code_blocks.append(ctx)
            return

    def runCodeBlocks(self):
        if not os.path.exists("temp"):
            os.mkdir("temp")

        for codes in self.code_blocks:

            fname = codes.file_name.split("/")[-1].split(".py")[0] + (
                "_{}_{}.py".format(codes.start_line, codes.end_line))
            content = "# -*- coding: utf-8 -*-\n" + codes.content
            #print(codes.file_name)
            #print(content)
            print("########## begin to run code blocks ###########\n")
            print("file_name: ", codes.file_name)
            print("start_line: ", codes.start_line)
            print("end_line: ", codes.end_line)

            print("content: \n")
            print(content)

            tmp_f = open("temp/" + fname, 'w')
            tmp_f.write(content)
            tmp_f.close()
            cmd = ["python3.7", "temp/" + fname]

            subprc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, error = subprc.communicate()
            err = "".join(error.decode(encoding='utf-8'))
            if subprc.returncode != 0:
                print("\nSample code error found in ", codes.file_name)

                print(err)
                self.error_blocks.append(codes)
            else:
                print("\nSample code run Successfully! \n")
            os.remove("temp/" + fname)
        shutil.rmtree("temp")

    def run(self):
        for paths, _, filenames in os.walk(self.root_path):
            for file in filenames:
                if file.endswith(".py") or file.endswith(".cc"):
                    file_path = os.path.join(paths, file)

                    if self.filter(file_path):
                        continue

                    self.extract_sample_code(file_path)

        self.runCodeBlocks()
        print("total code blocks: %d" % len(self.code_blocks))
        print("error code blocks: %d" % len(self.error_blocks))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: inadequate number of arguments")
        print("Please input the root path of Paddle")
        sys.exit(1)
    else:
        if not os.path.exists(sys.argv[1]):
            print("Root path of Paddle not found")
            sys.exit(1)

        root_path = sys.argv[1]
        gen = SampleCodeGenerator(root_path)
        gen.run()
