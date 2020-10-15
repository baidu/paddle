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
import unittest
import paddle.fluid.incubate.fleet.base.role_maker as role_maker


class TestCloudRoleMaker(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMaker.
    """

    def setUp(self):
        """Set up, set envs."""
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001,127.0.0.2:36001"

    def test_pslib_1(self):
        """Test cases for pslib."""
        import sys
        import threading
        import paddle.fluid as fluid
        try:
            from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
            from paddle.fluid.incubate.fleet.parameter_server.pslib import PSLib
            from paddle.fluid.incubate.fleet.base.role_maker import \
                GeneralRoleMaker
            from paddle.distributed.fleet.utils.http_server import KVHandler
            from paddle.distributed.fleet.utils.http_server import KVServer
            from paddle.distributed.fleet.utils.http_server import KVHTTPServer
        except:
            print("warning: no fleet, skip test_pslib_4")
            return

        class FakeStream():
            """
            it is a fake stream only for test.
            """

            def write(self, a):
                """
                write a to stream, do nothing

                Args:
                    a(str): the string to write
                """
                pass

            def read(self, b):
                """
                read data of len b from stream, do nothing

                Args:
                    b(str): the len to read

                Returns:
                    c(str): the result
                """
                if b == 0:
                    raise ValueError("this is only for test")
                return "fake"

        import os

        try:

            class TmpKVHander(KVHandler):
                """
                it is a fake handler only for this test case.
                """

                def __init__(self, server):
                    """Init."""
                    self.path = "a/b/c"
                    self.server = server
                    self.wfile = FakeStream()
                    self.rfile = FakeStream()
                    self.headers = {}
                    self.headers['Content-Length'] = 0

                def address_string(self):
                    """
                    fake address string, it will do nothing.
                    """
                    return "123"

                def send_response(self, code):
                    """
                    fake send response, it will do nothing.

                    Args:
                        code(int): error code
                    """
                    pass

                def send_header(self, a, b):
                    """
                    fake send header, it will do nothing.

                    Args:
                        a(str): some header
                        b(str): some header
                    """
                    pass

                def end_headers(self):
                    """
                    fake end header, it will do nothing.
                    """
                    pass
        except:
            print("warning: no KVHandler, skip test_pslib_4")
            return

        import sys

        try:

            class TmpServer(KVHTTPServer):
                """
                it is a fake server only for this test case.
                """

                def __init__(self):
                    """Init."""
                    self.delete_kv_lock = threading.Lock()
                    self.delete_kv = {}
                    self.kv_lock = threading.Lock()
                    self.kv = {}
        except:
            print("warning: no KVHTTPServer, skip test_pslib_4")
            return

        try:

            class TmpS(KVServer):
                """
                it is a fake server only for this test case.
                """

                def __init__(self):
                    """Init."""
                    self.http_server = TmpServer()
                    self.listen_thread = None
                    self.size = {}
                    self.size["a"] = 999
        except:
            print("warning: no KVServer, skip test_pslib_4")
            return

        s = TmpServer()
        h = TmpKVHander(s)
        h.do_GET()
        h.path = "a/b"
        h.do_GET()
        h.do_PUT()
        h.do_DELETE()
        h.path = "a/b/c"
        s.kv["b"] = {}
        s.kv["b"]["c"] = "456"
        h.do_GET()
        h.path = "a/d/e"
        h.do_PUT()
        h.headers['Content-Length'] = 1
        h.do_PUT()
        h.do_DELETE()
        h.log_message("666")
        s.get_deleted_size("haha")
        s1 = TmpS()
        s1.should_stop()


if __name__ == "__main__":
    unittest.main()
