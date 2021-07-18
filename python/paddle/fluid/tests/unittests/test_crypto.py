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

from paddle.fluid.core import CipherUtils
from paddle.fluid.core import CipherFactory
from paddle.fluid.core import Cipher

import unittest


class CipherUtilsTestCase(unittest.TestCase):
    def test_gen_key(self):
        key1 = CipherUtils.gen_key(256)
        key2 = CipherUtils.gen_key_to_file(256, "/tmp/paddle_aes_test.keyfile")
        self.assertNotEquals(key1, key2)
        key3 = CipherUtils.read_key_from_file("/tmp/paddle_aes_test.keyfile")
        self.assertEqual(key2, key3)
        self.assertEqual(len(key1), 32)
        self.assertEqual(len(key2), 32)


class CipherTestCase(unittest.TestCase):
    def test_aes_cipher(self):
        plaintext = "hello world"
        key = CipherUtils.gen_key(256)
        cipher = CipherFactory.create_cipher()

        ciphertext = cipher.encrypt(plaintext, key)
        cipher.encrypt_to_file(plaintext, key, "paddle_aes_test.ciphertext")

        plaintext1 = cipher.decrypt(ciphertext, key)
        plaintext2 = cipher.decrypt_from_file(key, "paddle_aes_test.ciphertext")

        self.assertEqual(plaintext, plaintext1.decode())
        self.assertEqual(plaintext1, plaintext2)


if __name__ == '__main__':
    unittest.main()
