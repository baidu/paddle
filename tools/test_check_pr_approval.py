#! /usr/bin/env python

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
"""
TestCases for check_pr_approval.py
"""
import unittest
import subprocess
import sys


class Test_check_approval(unittest.TestCase):
    def setUp(self):
        self.codeset = 'UTF-8'
        # only key info in it
        self.jsonstr = """
[
  {
    "id": 688077074,
    "node_id": "MDE3OlB1bGxSZXF1ZXN0UmV2aWV3Njg4MDc3MDc0",
    "user": {
      "login": "wadefelix",
      "id": 1306724,
      "type": "User",
      "site_admin": false
    },
    "body": "",
    "state": "COMMENTED",
    "author_association": "CONTRIBUTOR"
  },
  {
    "id": 688092580,
    "node_id": "MDE3OlB1bGxSZXF1ZXN0UmV2aWV3Njg4MDkyNTgw",
    "user": {
      "login": "MingMingShangTian",
      "id": 13469016,
      "type": "User",
      "site_admin": false
    },
    "body": "LGTM",
    "state": "APPROVED",
    "author_association": "CONTRIBUTOR"
  },
  {
    "id": 689175539,
    "node_id": "MDE3OlB1bGxSZXF1ZXN0UmV2aWV3Njg5MTc1NTM5",
    "user": {
      "login": "pangyoki",
      "id": 26408901,
      "type": "User",
      "site_admin": false
    },
    "body": "LGTM",
    "state": "APPROVED",
    "author_association": "CONTRIBUTOR"
  }
]
""".encode(self.codeset)

    def test_ids(self):
        cmd = [sys.executable, 'check_pr_approval.py', '1', '26408901']
        subprc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        output, error = subprc.communicate(input=self.jsonstr)
        self.assertEqual('TRUE', output.decode(self.codeset).rstrip())

    def test_logins(self):
        cmd = [sys.executable, 'check_pr_approval.py', '1', 'pangyoki']
        subprc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        output, error = subprc.communicate(input=self.jsonstr)
        self.assertEqual('TRUE', output.decode(self.codeset).rstrip())

    def test_ids_and_logins(self):
        cmd = [
            sys.executable, 'check_pr_approval.py', '2', 'pangyoki', '13469016'
        ]
        subprc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        output, error = subprc.communicate(input=self.jsonstr)
        #self.assertEqual('', error.rstrip())
        self.assertEqual('TRUE', output.decode(self.codeset).rstrip())

    def test_check_with_required_reviewer_not_approved(self):
        cmd = [
            sys.executable, 'check_pr_approval.py', '2', 'wadefelix',
            ' 13469016'
        ]
        subprc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        output, error = subprc.communicate(input=self.jsonstr)
        self.assertEqual('FALSE', output.decode(self.codeset).rstrip())


if __name__ == '__main__':
    unittest.main()
