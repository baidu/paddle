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
"""Test fleet metric."""

from __future__ import print_function
import numpy as np
import paddle
import paddle.fluid as fluid
import os
import unittest
import numpy as np
import paddle.distributed.fleet.metrics.metric as metric
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.base.util_factory import UtilBase

paddle.enable_static()


class TestFleetMetric(unittest.TestCase):
    """Test cases for fleet metric."""

    def setUp(self):
        """Set up, set envs."""

        class FakeUtil(UtilBase):
            def __init__(self, fake_fleet):
                super(FakeUtil, self).__init__()
                self.fleet = fake_fleet

            def all_reduce(self, input, mode="sum", comm_world="worker"):
                input = np.array(input)
                input_shape = input.shape
                input_list = input.reshape(-1).tolist()

                self.fleet._barrier(comm_world)

                ans = self.fleet._all_reduce(input_list, mode)

                output = np.array(ans).reshape(input_shape)
                return output

        class FakeFleet:
            """Fake fleet only for test."""

            def __init__(self):
                """Init."""
                self.gloo = fluid.core.Gloo()
                self.gloo.set_rank(0)
                self.gloo.set_size(1)
                self.gloo.set_prefix("123")
                self.gloo.set_iface("lo")
                self.gloo.set_hdfs_store("./tmp_test_metric", "", "")
                self.gloo.init()

            def _all_reduce(self, input, mode="sum"):
                """All reduce using gloo."""
                ans = self.gloo.all_reduce(input, mode)
                return ans

            def _barrier(self, comm_world="worker"):
                """Fake barrier, do nothing."""
                pass

        self.util = FakeUtil(FakeFleet())
        fleet.util = self.util

    def test_metric_1(self):
        """Test cases for metrics."""
        train = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(train, startup):
            t = fluid.layers.create_global_var(
                shape=[1, 1],
                value=1,
                dtype='int64',
                persistable=True,
                force_cpu=True)
            t1 = fluid.layers.create_global_var(
                shape=[1, 1],
                value=1,
                dtype='int64',
                persistable=True,
                force_cpu=True)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup)
            metric.sum(t, scope, self.util)
            metric.max(t, scope, self.util)
            metric.min(t, scope, self.util)
            metric.auc(t, t1, scope, self.util)
            metric.mae(t, t1, scope, self.util)
            metric.rmse(t, t1, scope, self.util)
            metric.mse(t, t1, scope, self.util)
            metric.acc(t, t1, scope, self.util)
            metric.sum(str(t.name))
            metric.max(str(t.name))
            metric.min(str(t.name))
            metric.auc(str(t1.name), str(t.name))
            metric.mae(str(t1.name), str(t.name))
            metric.rmse(str(t1.name), str(t.name))
            metric.mse(str(t1.name), str(t.name))
            metric.acc(str(t.name), str(t1.name))
        arr = np.array([1, 2, 3, 4])
        metric.sum(arr, util=self.util)
        metric.max(arr, util=self.util)
        metric.min(arr, util=self.util)
        arr1 = np.array([[1, 2, 3, 4]])
        arr2 = np.array([[1, 2, 3, 4]])
        arr3 = np.array([1, 2, 3, 4])
        metric.auc(arr1, arr2, util=self.util)
        metric.mae(arr, arr3, util=self.util)
        metric.rmse(arr, arr3, util=self.util)
        metric.mse(arr, arr3, util=self.util)
        metric.acc(arr, arr3, util=self.util)


if __name__ == "__main__":
    unittest.main()
