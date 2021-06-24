#! python

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

import unittest
import os
import shutil
import re
import sampcd_processor
from sampcd_processor import find_all
from sampcd_processor import get_api_md5
from sampcd_processor import get_incrementapi
from sampcd_processor import sampcd_extract_to_file
from sampcd_processor import extract_code_blocks_from_docstr
from sampcd_processor import execute_samplecode
from sampcd_processor import find_last_future_line_end
from sampcd_processor import insert_codes_into_codeblock
from sampcd_processor import get_test_capacity
from sampcd_processor import is_required_match


class Test_find_all(unittest.TestCase):
    def test_find_none(self):
        self.assertEqual(0, len(find_all('hello', 'world')))

    def test_find_one(self):
        self.assertListEqual([0], find_all('hello', 'hello'))

    def test_find_two(self):
        self.assertListEqual([1, 15],
                             find_all(' hello, world; hello paddle!', 'hello'))


class Test_find_last_future_line_end(unittest.TestCase):
    def test_no_instant(self):
        samplecodes = """
                print(10//3)
        """
        self.assertIsNone(find_last_future_line_end(samplecodes))

    def test_1_instant(self):
        samplecodes = """
                from __future__ import print_function

                print(10//3)
        """
        mo = re.search("print_function\n", samplecodes)
        self.assertIsNotNone(mo)
        self.assertGreaterEqual(
            find_last_future_line_end(samplecodes), mo.end())

    def test_2_instant(self):
        samplecodes = """
                from __future__ import print_function
                from __future__ import division

                print(10//3)
        """
        mo = re.search("division\n", samplecodes)
        self.assertIsNotNone(mo)
        self.assertGreaterEqual(
            find_last_future_line_end(samplecodes), mo.end())


class Test_extract_code_blocks_from_docstr(unittest.TestCase):
    def test_no_samplecode(self):
        docstr = """
        placeholder
        """
        codeblocks = extract_code_blocks_from_docstr(docstr)
        self.assertListEqual([], codeblocks)

    def test_codeblock_before_examples_is_ignored(self):
        docstr = """
            .. code-block:: python

                print(1+1)
        Examples:
        """
        codeblocks = extract_code_blocks_from_docstr(docstr)
        self.assertListEqual(codeblocks, [])

    def test_1_samplecode(self):
        docstr = """
        Examples:
            .. code-block:: python

                print(1+1)
        """
        codeblocks = extract_code_blocks_from_docstr(docstr)
        self.assertListEqual(codeblocks, [{
            'codes': """print(1+1)""",
            'name': None,
            'id': 1,
            'required': None,
        }])

    def test_2_samplecodes(self):
        docstr = """
        placeholder
        Examples:
            .. code-block:: python

                print(1/0)

            .. code-block:: python
               :name: one_plus_one
               :linenos:

                # required: gpu
                print(1+1)
        """
        codeblocks = extract_code_blocks_from_docstr(docstr)
        self.assertListEqual(codeblocks, [{
            'codes': """print(1/0)""",
            'name': None,
            'id': 1,
            'required': None,
        }, {
            'codes': """# required: gpu
print(1+1)""",
            'name': 'one_plus_one',
            'id': 2,
            'required': 'gpu',
        }])


class Test_insert_codes_into_codeblock(unittest.TestCase):
    def test_required_None(self):
        codeblock = {
            'codes': """print(1/0)""",
            'name': None,
            'id': 1,
            'required': None,
        }
        self.assertEqual("""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(1/0)
print("not-specified's sample code (name:None, id:1) is executed successfully!")""",
                         insert_codes_into_codeblock(codeblock))

    def test_required_gpu(self):
        codeblock = {
            'codes': """# required: gpu
print(1+1)""",
            'name': None,
            'id': 1,
            'required': 'gpu',
        }
        self.assertEqual("""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# required: gpu
print(1+1)
print("not-specified's sample code (name:None, id:1) is executed successfully!")""",
                         insert_codes_into_codeblock(codeblock))

    def test_from_future(self):
        codeblock = {
            'codes': """
from __future__ import print_function
from __future__ import division
print(10//3)""",
            'name': 'future',
            'id': 1,
            'required': None,
        }
        self.assertEqual("""
from __future__ import print_function
from __future__ import division

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(10//3)
print("not-specified's sample code (name:future, id:1) is executed successfully!")""",
                         insert_codes_into_codeblock(codeblock))


def clear_capacity():
    sampcd_processor.SAMPLE_CODE_TEST_CAPACITY = set()
    sampcd_processor.RUN_ON_DEVICE = 'cpu'
    if sampcd_processor.ENV_KEY_TEST_CAPACITY in os.environ:
        del os.environ[sampcd_processor.ENV_KEY_TEST_CAPACITY]


class Test_get_test_capacity(unittest.TestCase):
    def setUp(self):
        clear_capacity()
        get_test_capacity()

    def tearDown(self):
        clear_capacity()
        get_test_capacity()

    def test_NoEnvVar(self):
        clear_capacity()
        get_test_capacity()
        self.assertCountEqual(['cpu', ],
                              sampcd_processor.SAMPLE_CODE_TEST_CAPACITY)

    def test_NoEnvVar_RUN_ON_DEVICE_gpu(self):
        clear_capacity()
        sampcd_processor.RUN_ON_DEVICE = 'gpu'
        get_test_capacity()
        self.assertCountEqual(['cpu', 'gpu'],
                              sampcd_processor.SAMPLE_CODE_TEST_CAPACITY)

    def test_EnvVar_gpu(self):
        clear_capacity()
        os.environ[sampcd_processor.ENV_KEY_TEST_CAPACITY] = 'gpu'
        get_test_capacity()
        self.assertCountEqual(['cpu', 'gpu'],
                              sampcd_processor.SAMPLE_CODE_TEST_CAPACITY)

    def test_EnvVar_gpu_and_distributed(self):
        clear_capacity()
        os.environ[sampcd_processor.ENV_KEY_TEST_CAPACITY] = 'gpu,distributed'
        get_test_capacity()
        self.assertCountEqual(['cpu', 'gpu', 'distributed'],
                              sampcd_processor.SAMPLE_CODE_TEST_CAPACITY)


class Test_is_required_match(unittest.TestCase):
    def setUp(self):
        clear_capacity()

    def tearDown(self):
        clear_capacity()
        get_test_capacity()

    def test_alldefault(self):
        clear_capacity()
        get_test_capacity()
        self.assertTrue(is_required_match(''))
        self.assertTrue(is_required_match(None))
        self.assertTrue(is_required_match('cpu'))
        self.assertFalse(is_required_match('gpu'))
        self.assertIsNone(is_required_match('skiptest'))
        self.assertIsNone(is_required_match('skip'))
        self.assertIsNone(is_required_match('cpu,skiptest'))

    def test_gpu_equipped(self):
        clear_capacity()
        os.environ[sampcd_processor.ENV_KEY_TEST_CAPACITY] = 'gpu'
        get_test_capacity()
        self.assertTrue(is_required_match('cpu'))
        self.assertTrue(is_required_match('gpu'))
        self.assertTrue(is_required_match('gpu,cpu'))
        self.assertIsNone(is_required_match('skiptest'))
        self.assertFalse(is_required_match('distributed'))

    def test_gpu_distributed_equipped(self):
        clear_capacity()
        os.environ[sampcd_processor.ENV_KEY_TEST_CAPACITY] = 'gpu,distributed'
        get_test_capacity()
        self.assertTrue(is_required_match('cpu'))
        self.assertTrue(is_required_match('gpu'))
        self.assertTrue(is_required_match('distributed'))
        self.assertFalse(is_required_match('xpu'))
        self.assertIsNone(is_required_match('skiptest'))


class Test_execute_samplecode(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(sampcd_processor.SAMPLECODE_TEMPDIR):
            os.mkdir(sampcd_processor.SAMPLECODE_TEMPDIR)
        self.successSampleCodeFile = os.path.join(
            sampcd_processor.SAMPLECODE_TEMPDIR, 'samplecode_success.py')
        with open(self.successSampleCodeFile, 'w') as f:
            f.write('print(1+1)')
        self.failedSampleCodeFile = os.path.join(
            sampcd_processor.SAMPLECODE_TEMPDIR, 'samplecode_failed.py')
        with open(self.failedSampleCodeFile, 'w') as f:
            f.write('print(1/0)')

    def tearDown(self):
        os.remove(self.successSampleCodeFile)
        os.remove(self.failedSampleCodeFile)

    def test_run_success(self):
        result, tfname, msg, exec_time = execute_samplecode(
            self.successSampleCodeFile)
        self.assertTrue(result)
        self.assertEqual(self.successSampleCodeFile, tfname)
        self.assertIsNotNone(msg)
        self.assertLess(msg.find('skipped'), 0)
        self.assertLess(exec_time, 10)

    def test_run_failed(self):
        result, tfname, msg, exec_time = execute_samplecode(
            self.failedSampleCodeFile)
        self.assertFalse(result)
        self.assertEqual(self.failedSampleCodeFile, tfname)
        self.assertIsNotNone(msg)
        self.assertLess(msg.find('skipped'), 0)
        self.assertLess(exec_time, 10)


def clear_summary_info():
    for k in sampcd_processor.SUMMARY_INFO.keys():
        sampcd_processor.SUMMARY_INFO[k].clear()


class Test_sampcd_extract_to_file(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(sampcd_processor.SAMPLECODE_TEMPDIR):
            os.mkdir(sampcd_processor.SAMPLECODE_TEMPDIR)
        clear_capacity()
        os.environ[sampcd_processor.ENV_KEY_TEST_CAPACITY] = 'gpu,distributed'
        get_test_capacity()

    def tearDown(self):
        shutil.rmtree(sampcd_processor.SAMPLECODE_TEMPDIR)
        clear_capacity()
        get_test_capacity()

    def test_1_samplecode(self):
        comments = """
        Examples:
            .. code-block:: python

                print(1+1)
        """
        funcname = 'one_plus_one'
        sample_code_filenames = sampcd_extract_to_file(comments, funcname)
        self.assertCountEqual([
            os.path.join(sampcd_processor.SAMPLECODE_TEMPDIR,
                         funcname + '_example.py')
        ], sample_code_filenames)

    def test_no_samplecode(self):
        comments = """
        placeholder
        """
        funcname = 'one_plus_one'
        sample_code_filenames = sampcd_extract_to_file(comments, funcname)
        self.assertCountEqual([], sample_code_filenames)

    def test_2_samplecodes(self):
        comments = """
        placeholder
        Examples:
            .. code-block:: python

                print(1/0)

            .. code-block:: python

                print(1+1)
        """
        funcname = 'one_plus_one'
        sample_code_filenames = sampcd_extract_to_file(comments, funcname)
        self.assertCountEqual([
            os.path.join(sampcd_processor.SAMPLECODE_TEMPDIR,
                         funcname + '_example_1.py'),
            os.path.join(sampcd_processor.SAMPLECODE_TEMPDIR,
                         funcname + '_example_2.py')
        ], sample_code_filenames)

    def test_2_samplecodes_has_skipped(self):
        comments = """
        placeholder
        Examples:
            .. code-block:: python

                # required: skiptest
                print(1/0)

            .. code-block:: python

                print(1+1)

            .. code-block:: python

                # required: gpu
                print(1//1)

            .. code-block:: python

                # required: xpu
                print(1//1)

            .. code-block:: python

                # required: distributed
                print(1//1)

            .. code-block:: python

                # required: gpu
                print(1//1)
        """
        funcname = 'one_plus_one'
        clear_summary_info()
        clear_capacity()
        get_test_capacity()

        sample_code_filenames = sampcd_extract_to_file(comments, funcname)
        self.assertCountEqual([
            os.path.join(sampcd_processor.SAMPLECODE_TEMPDIR,
                         funcname + '_example_2.py')
        ], sample_code_filenames)
        self.assertCountEqual(sampcd_processor.SUMMARY_INFO['skiptest'],
                              [funcname + '-1'])
        self.assertCountEqual(sampcd_processor.SUMMARY_INFO['gpu'],
                              [funcname + '-3', funcname + '-6'])
        self.assertCountEqual(sampcd_processor.SUMMARY_INFO['xpu'],
                              [funcname + '-4'])
        self.assertCountEqual(sampcd_processor.SUMMARY_INFO['distributed'],
                              [funcname + '-5'])


class Test_get_api_md5(unittest.TestCase):
    def setUp(self):
        self.api_pr_spec_filename = os.path.abspath(
            os.path.join(os.getcwd(), "..", 'paddle/fluid/API_PR.spec'))
        with open(self.api_pr_spec_filename, 'w') as f:
            f.write("\n".join([
                """paddle.one_plus_one (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55one'))""",
                """paddle.two_plus_two (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55two'))""",
                """paddle.three_plus_three (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6cthree'))""",
                """paddle.four_plus_four (paddle.four_plus_four, ('document', 'ff0f188c95030158cc6398d2a6c5four'))""",
            ]))

    def tearDown(self):
        os.remove(self.api_pr_spec_filename)
        pass

    def test_get_api_md5(self):
        res = get_api_md5('paddle/fluid/API_PR.spec')
        self.assertEqual("ff0f188c95030158cc6398d2a6c55one",
                         res['paddle.one_plus_one'])
        self.assertEqual("ff0f188c95030158cc6398d2a6c55two",
                         res['paddle.two_plus_two'])
        self.assertEqual("ff0f188c95030158cc6398d2a6cthree",
                         res['paddle.three_plus_three'])
        self.assertEqual("ff0f188c95030158cc6398d2a6c5four",
                         res['paddle.four_plus_four'])


class Test_get_incrementapi(unittest.TestCase):
    def setUp(self):
        self.api_pr_spec_filename = os.path.abspath(
            os.path.join(os.getcwd(), "..", 'paddle/fluid/API_PR.spec'))
        with open(self.api_pr_spec_filename, 'w') as f:
            f.write("\n".join([
                """paddle.one_plus_one (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55one'))""",
                """paddle.two_plus_two (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55two'))""",
                """paddle.three_plus_three (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6cthree'))""",
                """paddle.four_plus_four (paddle.four_plus_four, ('document', 'ff0f188c95030158cc6398d2a6c5four'))""",
            ]))
        self.api_dev_spec_filename = os.path.abspath(
            os.path.join(os.getcwd(), "..", 'paddle/fluid/API_DEV.spec'))
        with open(self.api_dev_spec_filename, 'w') as f:
            f.write("\n".join([
                """paddle.one_plus_one (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55one'))""",
            ]))
        self.api_diff_spec_filename = os.path.abspath(
            os.path.join(os.getcwd(), "dev_pr_diff_api.spec"))

    def tearDown(self):
        os.remove(self.api_pr_spec_filename)
        os.remove(self.api_dev_spec_filename)
        os.remove(self.api_diff_spec_filename)

    def test_it(self):
        get_incrementapi()
        with open(self.api_diff_spec_filename, 'r') as f:
            lines = f.readlines()
            self.assertCountEqual([
                "paddle.two_plus_two\n", "paddle.three_plus_three\n",
                "paddle.four_plus_four\n"
            ], lines)


# https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/layers/ops.py
# why? unabled to use the ast module. emmmmm

if __name__ == '__main__':
    unittest.main()
