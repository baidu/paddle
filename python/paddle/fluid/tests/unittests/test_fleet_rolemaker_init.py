#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Test cloud role maker."""

from __future__ import print_function
import os
import platform
import shutil
import tempfile
import unittest
import paddle
import paddle.distributed.fleet.base.role_maker as role_maker


class TestPSCloudRoleMakerCase1(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMake Parameter Server.
    """

    def setUp(self):
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:4001,127.0.0.1:4002"

    def test_paddle_trainers_num(self):
        # PADDLE_TRAINERS_NUM
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)


class TestPSCloudRoleMakerCase2(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMake Parameter Server.
    """

    def setUp(self):
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:4001,127.0.0.1:4002"
        os.environ["PADDLE_TRAINERS_NUM"] = str(2)

    def test_training_role(self):
        # TRAINING_ROLE
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)


class TestPSCloudRoleMakerCase3(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMake Parameter Server.
    """

    def setUp(self):
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:4001,127.0.0.1:4002"
        os.environ["PADDLE_TRAINERS_NUM"] = str(2)
        os.environ["TRAINING_ROLE"] = 'TRAINER'

    def test_trainer_id(self):
        # PADDLE_TRAINER_ID
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)


class TestPSCloudRoleMakerCase4(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMake Parameter Server.
    """

    def setUp(self):
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:4001,127.0.0.1:4002"
        os.environ["PADDLE_TRAINERS_NUM"] = str(2)
        os.environ["TRAINING_ROLE"] = 'PSERVER'

    def test_ps_port(self):
        # PADDLE_PORT
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)


class TestPSCloudRoleMakerCase5(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMake Parameter Server.
    """

    def setUp(self):
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:4001,127.0.0.1:4002"
        os.environ["PADDLE_TRAINERS_NUM"] = str(2)
        os.environ["TRAINING_ROLE"] = 'PSERVER'
        os.environ["PADDLE_PORT"] = str(4001)

    def test_ps_ip(self):
        # POD_IP
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)


class TestPSCloudRoleMakerCase6(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMake Parameter Server.
    """

    def setUp(self):
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:4001,127.0.0.1:4002"
        os.environ[
            "PADDLE_HETER_TRAINER_IP_PORT_LIST"] = "127.0.0.1:4003,127.0.0.1:4004"
        os.environ["PADDLE_TRAINERS_NUM"] = str(2)
        os.environ["TRAINING_ROLE"] = 'HETER_TRAINER'

    def test_heter_port(self):
        # PADDLE_PORT
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)


class TestPSCloudRoleMakerCase7(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMake Parameter Server.
    """

    def setUp(self):
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:4001,127.0.0.1:4002"
        os.environ[
            "PADDLE_HETER_TRAINER_IP_PORT_LIST"] = "127.0.0.1:4003,127.0.0.1:4004"
        os.environ["PADDLE_TRAINERS_NUM"] = str(2)
        os.environ["TRAINING_ROLE"] = 'HETER_TRAINER'
        os.environ["PADDLE_PORT"] = str(4003)

    def test_heter_ip(self):
        # POD_IP
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)


if __name__ == "__main__":
    unittest.main()
