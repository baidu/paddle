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

from __future__ import print_function

import unittest
import paddle.compat as cpt


class TestCompatible(unittest.TestCase):
    def test_type(self):
        self.assertEqual(cpt.int_type, int)
        self.assertEqual(cpt.long_type, int)

    def test_to_text(self):
        self.assertIsNone(cpt.to_text(None))

        self.assertTrue(isinstance(cpt.to_text(str("")), str))
        self.assertTrue(isinstance(cpt.to_text(str("123")), str))
        self.assertTrue(isinstance(cpt.to_text(b""), str))
        self.assertTrue(isinstance(cpt.to_text(b""), str))
        self.assertTrue(isinstance(cpt.to_text(u""), str))
        self.assertTrue(isinstance(cpt.to_text(u""), str))

        self.assertEqual("", cpt.to_text(str("")))
        self.assertEqual("123", cpt.to_text(str("123")))
        self.assertEqual("", cpt.to_text(b""))
        self.assertEqual("123", cpt.to_text(b"123"))
        self.assertEqual("", cpt.to_text(u""))
        self.assertEqual("123", cpt.to_text(u"123"))

        # check list types, not inplace
        l = [""]
        l2 = cpt.to_text(l)
        self.assertTrue(isinstance(l2, list))
        self.assertFalse(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual([""], l2)
        l = ["", "123"]
        l2 = cpt.to_text(l)
        self.assertTrue(isinstance(l2, list))
        self.assertFalse(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual(["", "123"], l2)
        l = ["", b"123", u"321"]
        l2 = cpt.to_text(l)
        self.assertTrue(isinstance(l2, list))
        self.assertFalse(l is l2)
        self.assertNotEqual(l, l2)
        self.assertEqual(["", "123", "321"], l2)

        # check list types, inplace
        l = [""]
        l2 = cpt.to_text(l, inplace=True)
        self.assertTrue(isinstance(l2, list))
        self.assertTrue(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual([""], l2)
        l = ["", b"123"]
        l2 = cpt.to_text(l, inplace=True)
        self.assertTrue(isinstance(l2, list))
        self.assertTrue(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual(["", "123"], l2)
        l = ["", b"123", u"321"]
        l2 = cpt.to_text(l, inplace=True)
        self.assertTrue(isinstance(l2, list))
        self.assertTrue(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual(["", "123", "321"], l2)
        for i in l2:
            self.assertTrue(isinstance(i, str))

        # check set types, not inplace
        l = set("")
        l2 = cpt.to_text(l, inplace=False)
        self.assertTrue(isinstance(l2, set))
        self.assertFalse(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual(set(""), l2)
        l = set([b"", b"123"])
        l2 = cpt.to_text(l, inplace=False)
        self.assertTrue(isinstance(l2, set))
        self.assertFalse(l is l2)
        self.assertNotEqual(l, l2)
        self.assertEqual(set(["", "123"]), l2)
        l = set(["", b"123", u"321"])
        l2 = cpt.to_text(l, inplace=False)
        self.assertTrue(isinstance(l2, set))
        self.assertFalse(l is l2)
        self.assertNotEqual(l, l2)
        self.assertEqual(set(["", "123", "321"]), l2)

        # check set types, inplace
        l = set("")
        l2 = cpt.to_text(l, inplace=True)
        self.assertTrue(isinstance(l2, set))
        self.assertTrue(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual(set(""), l2)
        l = set([b"", b"123"])
        l2 = cpt.to_text(l, inplace=True)
        self.assertTrue(isinstance(l2, set))
        self.assertTrue(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual(set(["", "123"]), l2)
        l = set(["", b"123", u"321"])
        l2 = cpt.to_text(l, inplace=True)
        self.assertTrue(isinstance(l2, set))
        self.assertTrue(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual(set(["", "123", "321"]), l2)
        for i in l2:
            self.assertTrue(isinstance(i, str))

        # check dict types, not inplace
        l = {"": ""}
        l2 = cpt.to_text(l, inplace=False)
        self.assertTrue(isinstance(l2, dict))
        self.assertFalse(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual({"": ""}, l2)

        # check dict types, inplace
        l = {"": ""}
        l2 = cpt.to_text(l, inplace=True)
        self.assertTrue(isinstance(l2, dict))
        self.assertTrue(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual({"": ""}, l2)

    def test_to_bytes(self):
        self.assertIsNone(cpt.to_bytes(None))

        self.assertTrue(isinstance(cpt.to_bytes(str("")), bytes))
        self.assertTrue(isinstance(cpt.to_bytes(str("123")), bytes))
        self.assertTrue(isinstance(cpt.to_bytes(b""), bytes))
        self.assertTrue(isinstance(cpt.to_bytes(b""), bytes))
        self.assertTrue(isinstance(cpt.to_bytes(u""), bytes))
        self.assertTrue(isinstance(cpt.to_bytes(u""), bytes))

        self.assertEqual(b"", cpt.to_bytes(str("")))
        self.assertEqual(b"123", cpt.to_bytes(str("123")))
        self.assertEqual(b"", cpt.to_bytes(b""))
        self.assertEqual(b"123", cpt.to_bytes(b"123"))
        self.assertEqual(b"", cpt.to_bytes(u""))
        self.assertEqual(b"123", cpt.to_bytes(u"123"))

        # check list types, not inplace
        l = [""]
        l2 = cpt.to_bytes(l)
        self.assertTrue(isinstance(l2, list))
        self.assertFalse(l is l2)
        self.assertNotEqual(l, l2)
        self.assertEqual([b""], l2)
        l = ["", "123"]
        l2 = cpt.to_bytes(l)
        self.assertTrue(isinstance(l2, list))
        self.assertFalse(l is l2)
        self.assertNotEqual(l, l2)
        self.assertEqual([b"", b"123"], l2)
        l = ["", b"123", u"321"]
        l2 = cpt.to_bytes(l)
        self.assertTrue(isinstance(l2, list))
        self.assertFalse(l is l2)
        self.assertNotEqual(l, l2)
        self.assertEqual([b"", b"123", b"321"], l2)

        # check list types, inplace
        l = [""]
        l2 = cpt.to_bytes(l, inplace=True)
        self.assertTrue(isinstance(l2, list))
        self.assertTrue(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual([b""], l2)
        l = ["", b"123"]
        l2 = cpt.to_bytes(l, inplace=True)
        self.assertTrue(isinstance(l2, list))
        self.assertTrue(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual([b"", b"123"], l2)
        l = ["", b"123", u"321"]
        l2 = cpt.to_bytes(l, inplace=True)
        self.assertTrue(isinstance(l2, list))
        self.assertTrue(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual([b"", b"123", b"321"], l2)
        for i in l2:
            self.assertTrue(isinstance(i, bytes))

        # check set types, not inplace
        l = set([""])
        l2 = cpt.to_bytes(l, inplace=False)
        self.assertTrue(isinstance(l2, set))
        self.assertFalse(l is l2)
        self.assertNotEqual(l, l2)
        self.assertEqual(set([b""]), l2)
        l = set([u"", u"123"])
        l2 = cpt.to_bytes(l, inplace=False)
        self.assertTrue(isinstance(l2, set))
        self.assertFalse(l is l2)
        self.assertNotEqual(l, l2)
        self.assertEqual(set([b"", b"123"]), l2)
        l = set(["", b"123", u"321"])
        l2 = cpt.to_bytes(l, inplace=False)
        self.assertTrue(isinstance(l2, set))
        self.assertFalse(l is l2)
        self.assertNotEqual(l, l2)
        self.assertEqual(set([b"", b"123", b"321"]), l2)

        # check set types, inplace
        l = set("")
        l2 = cpt.to_bytes(l, inplace=True)
        self.assertTrue(isinstance(l2, set))
        self.assertTrue(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual(set(b""), l2)
        l = set([u"", u"123"])
        l2 = cpt.to_bytes(l, inplace=True)
        self.assertTrue(isinstance(l2, set))
        self.assertTrue(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual(set([b"", b"123"]), l2)
        l = set(["", b"123", u"321"])
        l2 = cpt.to_bytes(l, inplace=True)
        self.assertTrue(isinstance(l2, set))
        self.assertTrue(l is l2)
        self.assertEqual(l, l2)
        self.assertEqual(set([b"", b"123", b"321"]), l2)
        for i in l2:
            self.assertTrue(isinstance(i, bytes))

    def test_round(self):
        self.assertEqual(3.0, cpt.round(3.4))
        self.assertEqual(4.0, cpt.round(3.5))
        self.assertEqual(0.0, cpt.round(0.1))
        self.assertEqual(0.0, cpt.round(0.0))
        self.assertEqual(-0.0, cpt.round(-0.0))
        self.assertEqual(-0.0, cpt.round(-0.1))
        self.assertEqual(-3.0, cpt.round(-3.4))
        self.assertEqual(-4.0, cpt.round(-3.5))
        self.assertEqual(5.0, cpt.round(5))
        self.assertRaises(TypeError, cpt.round, None)

    def test_floor_division(self):
        self.assertEqual(0.0, cpt.floor_division(3, 4))
        self.assertEqual(1.0, cpt.floor_division(4, 3))
        self.assertEqual(2.0, cpt.floor_division(6, 3))
        self.assertEqual(-2.0, cpt.floor_division(-4, 3))
        self.assertEqual(-2.0, cpt.floor_division(-6, 3))
        self.assertRaises(ZeroDivisionError, cpt.floor_division, 3, 0)
        self.assertRaises(TypeError, cpt.floor_division, None, None)

    def test_get_exception_message(self):
        exception_message = "test_message"
        self.assertRaises(AssertionError, cpt.get_exception_message, None)
        try:
            raise RuntimeError(exception_message)
        except Exception as e:
            self.assertEqual(exception_message, cpt.get_exception_message(e))
            self.assertIsNotNone(e)

        try:
            raise Exception(exception_message)
        except Exception as e:
            self.assertEqual(exception_message, cpt.get_exception_message(e))
            self.assertIsNotNone(e)


if __name__ == "__main__":
    unittest.main()
