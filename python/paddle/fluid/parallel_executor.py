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
import multiprocessing
from . import core
from . import framework
from . import executor
import warnings
import sys
import os

__all__ = ['ParallelExecutor', 'ExecutionStrategy', 'BuildStrategy']

ExecutionStrategy = core.ParallelExecutor.ExecutionStrategy
BuildStrategy = core.ParallelExecutor.BuildStrategy


class ParallelExecutor(object):
    """
    ParallelExecutor can run program in parallel.

    Args:
        use_cuda (bool): Whether to use CUDA or not.
        loss_name (str): The loss name must set in training. Default None.
        main_program (Program): The program that need to run, if not provided,
            then default_main_program will be used. Default None.
        share_vars_from(ParallelExecutor): If provied, it will share variables
            from the specified ParallelExecutor. Default None.
        num_trainers(int): If greater than 1, NCCL will be initialized with
            multiple rank of nodes, each node should have same number of GPUs.
            Distributed training will be enabled then. Default 1.
        trainer_id(int: Must use together with num_trainers. trainer_id is the
            "rank" of current node starts from 0. Default 0.

    Returns:
        ParallelExecutor: The initialized ParallelExecutor object.

    Raises:
        TypeError: If share_vars_from is provided, but not ParallelExecutor object.

    Examples:
        .. code-block:: python

          train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=loss.name)
          test_exe = fluid.ParallelExecutor(use_cuda=True,
                                            main_program=test_program,
                                            share_vars_from=train_exe)

          train_loss, = train_exe.run([loss.name], feed=feed_dict)
          test_loss, = test_exe.run([loss.name], feed=feed_dict)
    """

    def __init__(self,
                 use_cuda,
                 loss_name=None,
                 main_program=None,
                 share_vars_from=None,
                 exec_strategy=None,
                 build_strategy=None,
                 num_trainers=1,
                 trainer_id=0,
                 **kwargs):
        if len(kwargs) != 0:
            err_msg = ""
            for key in kwargs:
                if key in dir(ExecutionStrategy):
                    err_msg += \
                        "Setting {0} by constructor is deprecated. Use " \
                        "strategy=ExecutionStrategy(); strategy.{0}=xxx; " \
                        "pe=ParallelExecutor(exec_strategy=strategy) " \
                        "instead.\n ".format(key)
                elif key in dir(BuildStrategy):
                    err_msg += \
                        "Setting {0} by constructor is deprecated. Use " \
                        "strategy=BuildStrategy(); See help(" \
                        "paddle.fluid.ParallelExecutor.BuildStrategy) \n".format(
                            key)
                else:
                    err_msg += "Setting {0} by constructor is deprecated. Use strategy.\n".format(
                        key)
            raise ValueError(err_msg)

        self._places = []
        self._act_places = []
        if use_cuda:
            for i in range(core.get_cuda_device_count()):
                p = core.Place()
                self._act_places.append(core.CUDAPlace(i))
                p.set_place(self._act_places[-1])
                self._places.append(p)
        else:
            cpu_num = int(
                os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
            for i in range(cpu_num):
                p = core.Place()
                self._act_places.append(core.CPUPlace())
                p.set_place(self._act_places[-1])
                self._places.append(p)
        assert self._places, "no place for execution"

        if exec_strategy is None:
            exec_strategy = ExecutionStrategy()
        exec_strategy.use_cuda = use_cuda

        if exec_strategy.num_threads == 0:
            if use_cuda:
                # Experiments on se-resnext shows that too many threads hurt
                # performance. Worth tunning for other models in the future.
                exec_strategy.num_threads = len(self._places) * 4
            else:
                cpu_num = int(
                    os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
                exec_strategy.num_threads = cpu_num

        if build_strategy is None:
            build_strategy = BuildStrategy()

        main = main_program
        main = main if main else framework.default_main_program()
        scope = executor.global_scope()
        # FIXME(Yancey1989): it's a temporary approach to determinate the distribute
        # train program, call self.bcast_param() at the end of each mini-batch.
        self.is_dist = True if "recv" in [
            op.type for op in main.global_block().ops
        ] else False

        if share_vars_from and not isinstance(share_vars_from,
                                              ParallelExecutor):
            raise TypeError("share_vars_from must be ParallelExecutor.")

        local_scopes = share_vars_from.executor.local_scopes(
        ) if share_vars_from else []

        self.persistable_vars = [
            v.name for v in [
                var for var in main.list_vars()
                if var.persistable and var.type != core.VarDesc.VarType.RAW
            ]
        ]

        self.executor = core.ParallelExecutor(
            self._places,
            set([
                p.name for p in main.global_block().iter_parameters()
                if not p.stop_gradient
            ]),
            set(self.persistable_vars), main.desc, loss_name
            if loss_name else '', scope, local_scopes, exec_strategy,
            build_strategy, num_trainers, trainer_id)
        self.scope = scope

    def run(self, fetch_list, feed=None, feed_dict=None, return_numpy=True):
        """
        Run a parallel executor with fetch_list.

        The feed parameter can be a dict or a list. If feed is a dict, the
        feed data will be split into multiple devices. If feed is a list, we
        assume the data has been splitted into multiple devices, the each
        element in the list will be copied to each device directly.

        For example, if the feed is a dict:

        >>> exe = ParallelExecutor()
        >>> # the image will be splitted into devices. If there is two devices
        >>> # each device will process an image with shape (24, 1, 28, 28)
        >>> exe.run(feed={'image': numpy.random.random(size=(48, 1, 28, 28))})

        For example, if the feed is a list:

        >>> exe = ParallelExecutor()
        >>> # each device will process each element in the list.
        >>> # the 1st device will process an image with shape (48, 1, 28, 28)
        >>> # the 2nd device will process an image with shape (32, 1, 28, 28)
        >>> #
        >>> # you can use exe.device_count to get the device number.
        >>> exe.run(feed=[{"image": numpy.random.random(size=(48, 1, 28, 28))},
        >>>               {"image": numpy.random.random(size=(32, 1, 28, 28))},
        >>>              ])

        Args:
            fetch_list(list): The fetched variable names
            feed(list|dict|None): The feed variables. If the feed is a dict,
                tensors in that dict will be splitted into each devices. If
                the feed is a list, each element of the list will be copied
                to each device. Default None.
            feed_dict: Alias for feed parameter, for backward compatibility.
                This parameter has been deprecated. Default None.
            return_numpy(bool): Whether converts the fetched tensor to numpy.
                Default: True.

        Returns:
            List: The fetched result list.

        Raises:
            ValueError: If the feed is a list, but its length is not equal the
                length of active places, or its element's is not dict.

        NOTES:
            1. If the feed's type is dict, the number of data that feeds to
               ParallelExecutor must be bigger than active places. Otherwise,
               it will throw exception from C++ side. Special attention should be
               paid to check whether the last batch of the dataset is bigger
               than active places.
            2. If active places are more than one, the fetch results for each
               variable is a list, and each element of this list is the variable of
               respective active place.

        Examples:
            .. code-block:: python

                pe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                            loss_name=avg_cost.name,
                                            main_program=fluid.default_main_program())
                loss = pe.run(feed=feeder.feed(cur_batch),
                              fetch_list=[avg_cost.name]))
        """
        if feed is None and feed_dict is not None:
            feed = feed_dict
            print(
                "`feed_dict` is deprecated. Please use `feed=`",
                file=sys.stderr)

        if isinstance(feed, dict):
            feed_tensor_dict = dict()
            for feed_name in feed:
                feed_tensor = feed[feed_name]
                if not isinstance(feed_tensor, core.LoDTensor):
                    feed_tensor = core.LoDTensor()
                    # always set to CPU place, since the tensor need to be splitted
                    # it is fast in CPU
                    feed_tensor.set(feed[feed_name], core.CPUPlace())
                feed_tensor_dict[feed_name] = feed_tensor

            self.executor.feed_and_split_tensor_into_local_scopes(
                feed_tensor_dict)
        elif isinstance(feed, list) or isinstance(feed, tuple):
            if len(feed) != len(self._act_places):
                raise ValueError(
                    "Feed a list of tensor, the list should be the same size as places"
                )

            res = list()

            for i, each in enumerate(feed):
                if not isinstance(each, dict):
                    raise TypeError(
                        "Each element of feed list should be a dict")
                res_dict = dict()
                for feed_name in each:
                    tensor = each[feed_name]
                    if not isinstance(tensor, core.LoDTensor):
                        tmp = core.LoDTensor()
                        tmp.set(tensor, self._act_places[i])
                        tensor = tmp
                    res_dict[feed_name] = tensor
                res.append(res_dict)
            self.executor.feed_tensors_into_local_scopes(res)

        fetch_var_name = '@FETCHED_VAR_NAME@'
        self.executor.run(fetch_list, fetch_var_name)
        arr = self.scope.find_var(fetch_var_name).get_lod_tensor_array()

        if self.is_dist:
            self.bcast_params()

        if return_numpy:
            return executor.as_numpy(arr)

        return [arr[i] for i in range(len(arr))]

    def bcast_params(self):
        """
        Broadcast the parameters to other devices. It is used during
        distributed training.
        """
        self.executor.bcast_params(set(self.persistable_vars))

    @property
    def device_count(self):
        return len(self._act_places)
