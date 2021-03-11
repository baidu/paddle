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

import os
import six
import sys
import paddle
import numpy as np
from collections import namedtuple
from .. import core
from .fetcher import _IterableDatasetFetcher, _MapDatasetFetcher
from ..multiprocess_utils import _cleanup_mmap, CleanupFuncRegistrar, MP_STATUS_CHECK_INTERVAL
from ..framework import in_dygraph_mode

# NOTE: queue has a different name in python2 and python3
if six.PY2:
    import Queue as queue
else:
    import queue

__all__ = ['get_worker_info']


class _IterableDatasetStopIteration(object):
    def __init__(self, worker_id):
        self.worker_id = worker_id


class _DatasetKind(object):
    MAP = 0
    ITER = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collate_batch, collate_fn,
                       drop_last):
        if kind == _DatasetKind.MAP:
            return _MapDatasetFetcher(dataset, auto_collate_batch, collate_fn,
                                      drop_last)
        elif kind == _DatasetKind.ITER:
            return _IterableDatasetFetcher(dataset, auto_collate_batch,
                                           collate_fn, drop_last)
        else:
            raise NotImplementedError("unknown Dataset kind {}".format(kind))


def _worker_loop(dataset, dataset_kind, indices_queue, out_queue, done_event,
                 auto_collate_batch, collate_fn, init_fn, worker_id,
                 num_workers, use_shared_memory):
    try:
        # NOTE: [ mmap files clear ] When the child process exits unexpectedly,
        # some shared memory objects may have been applied for but have not yet
        # been put into the inter-process Queue. This part of the object needs
        # to be cleaned up when the process ends.
        CleanupFuncRegistrar.register(_cleanup_mmap)

        # set signal handler
        core._set_process_signal_handler()

        global _worker_info
        _worker_info = WorkerInfo(
            id=worker_id, num_workers=num_workers, dataset=dataset)

        init_exception = None
        try:
            if init_fn is not None:
                init_fn(worker_id)
            fetcher = _DatasetKind.create_fetcher(
                dataset_kind, dataset, auto_collate_batch, collate_fn, True)
        except:
            init_exception = Exception("init_fn failed in worker {}: " \
                                    "{}".format(worker_id, sys.exc_info()))

        iterator_drained = False
        parent_watch_dog = ParentWatchDog()

        while parent_watch_dog.is_alive():
            try:
                data = indices_queue.get(MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            # None as poison piil, so worker event should be set
            if data is None:
                assert done_event.is_set() or iterator_drained, \
                        "get None when worker done_event set"
                break
            # If worker done event is set but get still get data in
            # indices_queue, remaining data should be get and skipped.
            if done_event.is_set() or iterator_drained:
                continue

            idx, indices = data
            try:
                if init_exception is not None:
                    batch = init_exception
                    init_exception = None
                else:
                    batch = fetcher.fetch(indices)
            except Exception as e:
                if isinstance(
                        e, StopIteration) and dataset_kind == _DatasetKind.ITER:
                    out_queue.put(
                        (idx, _IterableDatasetStopIteration(worker_id)))
                    iterator_drained = True
                else:
                    out_queue.put((idx, e))
            else:
                if use_shared_memory:
                    # FIXME(dkp): _convert_to_tensor_list only support np.array
                    #             list now, should support paddle.Tensor list
                    if isinstance(batch[0], paddle.Tensor):
                        batch = [field.numpy() for field in batch]
                    if isinstance(batch, np.ndarray):
                        batch = [batch]

                    tensor_list = core._convert_to_tensor_list(batch)
                    if in_dygraph_mode():
                        varbase_list = [
                            core.VarBase(
                                t, keep_place=True) for t in tensor_list
                        ]
                        out_queue.put((idx, varbase_list))
                    else:
                        out_queue.put((idx, tensor_list))
                    core._remove_tensor_list_mmap_fds(tensor_list)
                else:
                    out_queue.put((idx, batch))
    except KeyboardInterrupt:
        # NOTE: Main process will raise KeyboardInterrupt anyways, ignore it in child process
        pass
    except:
        six.reraise(*sys.exc_info())
    finally:
        if use_shared_memory:
            _cleanup_mmap()


class ParentWatchDog(object):
    def __init__(self):
        self._parent_pid = os.getppid()
        self._parent_alive = True

    def is_alive(self):
        if self._parent_alive:
            self._parent_alive = os.getppid() == self._parent_pid
        return self._parent_alive


# worker information for each workers, used for splitting data copy
# for IteratorDataset in worker processes.
_worker_info = None


def get_worker_info():
    """
    Get DataLoader worker process information function, this function is
    used to split data copy in worker process for IterableDataset
    (see :code:`paddle.io.IterableDataset`), worker information contains
    following fields:

    :attr:`num_workers`: total worker process number, see `paddle.io.DataLoader`

    :attr:`id`: the worker processs id, count from 0 to :attr:`num_workers - 1`

    :attr:`dataset`: the dataset object in this worker process

    Returns:
        WorkerInfo: an instance of WorkerInfo which contains fields above.

    .. note::
        For mode usage and exampls, please see :code:`paddle.io.IterableDataset`

    Example:

        .. code-block:: python

            import math
            import paddle
            import numpy as np
            from paddle.io import IterableDataset, DataLoader, get_worker_info

            class SplitedIterableDataset(IterableDataset):
                def __init__(self, start, end):
                    self.start = start
                    self.end = end

                def __iter__(self):
                    worker_info = get_worker_info()
                    if worker_info is None:
                        iter_start = self.start
                        iter_end = self.end
                    else:
                        per_worker = int(
                            math.ceil((self.end - self.start) / float(
                                worker_info.num_workers)))
                        worker_id = worker_info.id
                        iter_start = self.start + worker_id * per_worker
                        iter_end = min(iter_start + per_worker, self.end)

                    for i in range(iter_start, iter_end):
                        yield np.array([i])

            place = paddle.CPUPlace()
            dataset = SplitedIterableDataset(start=2, end=9)
            dataloader = DataLoader(
                dataset,
                places=place,
                num_workers=2,
                batch_size=1,
                drop_last=True)

            for data in dataloader:
                print(data)
            # outputs: [2, 5, 3, 6, 4, 7]

    """
    return _worker_info


class WorkerInfo(object):
    __initialized = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__initialized = True

    def __setattr__(self, key, val):
        if self.__initialized:
            raise RuntimeError("Cannot assign attributes to {} objects".format(
                self.__class__.__name__))
        return super(WorkerInfo, self).__setattr__(key, val)
