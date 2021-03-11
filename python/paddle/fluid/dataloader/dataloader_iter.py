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

import os
import six
import sys
import time
import signal
import numbers
import logging
import itertools
import threading
import numpy as np
import multiprocessing

# NOTE: queue has a different name in python2 and python3
if six.PY2:
    import Queue as queue
else:
    import queue

import paddle
from .. import core, layers
from ..framework import in_dygraph_mode, _current_expected_place
from ..multiprocess_utils import _set_SIGCHLD_handler, MP_STATUS_CHECK_INTERVAL
from .worker import ParentWatchDog, get_worker_info, _worker_loop, _DatasetKind, _IterableDatasetStopIteration
from .pin_memory import pin_memory, _pin_memory_loop
from .collate import default_collate_fn, default_convert_fn
from .batch_sampler import _InfiniteIterableSampler

__all__ = ['get_worker_info']


class _DataLoaderIterBase(object):
    """
    Iterator implement of DataLoader, will load and feed mini-batch
    data by setting in given dataloader.

    Args:
        loader(instance of DataLoader): instance of `fluid.io.DataLoader`
    """

    def __init__(self, loader):
        self._dataset = loader.dataset
        self._feed_list = loader.feed_list or []
        self._places = loader.places
        self._return_list = loader.return_list
        self._batch_sampler = loader.batch_sampler
        self._auto_collate_batch = loader.auto_collate_batch
        self._num_workers = loader.num_workers
        self._use_buffer_reader = loader.use_buffer_reader
        self._use_shared_memory = loader.use_shared_memory
        self._timeout = loader.timeout if loader.timeout > 0 else MP_STATUS_CHECK_INTERVAL
        self._worker_init_fn = loader.worker_init_fn
        self._dataset_kind = loader.dataset_kind
        self._pin_memory = loader.pin_memory

        if self._auto_collate_batch:
            self._sampler_iter = iter(loader.batch_sampler)
            self._collate_fn = loader.collate_fn or default_collate_fn
        else:
            if self._dataset_kind == _DatasetKind.MAP:
                self._sampler_iter = iter(list(range(len(self._dataset))))
            else:
                self._sampler_iter = iter(
                    _InfiniteIterableSampler(self._dataset, 1))
            self._collate_fn = loader.collate_fn or default_convert_fn

        # LoDTensorBlockingQueue instance for create_py_reader and a thread
        # to put mini-batch data to self._blocking_queue, mini-batch data
        # will be get from:
        # 1. multi-process mode: get data from workers' result queue
        # 2. single-process mode: read mini-batch data in main process
        # self._blocking_queue = None
        self._thread = None
        self._thread_done_event = threading.Event()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._batch_sampler)


class _DataLoaderIterSingleProcess(_DataLoaderIterBase):
    """
    Single process implement of DataLoaderIter, loading data from
    loader.data in main process
    """

    def __init__(self, loader):
        super(_DataLoaderIterSingleProcess, self).__init__(loader)

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collate_batch,
            self._collate_fn, True)

    def __next__(self):
        indices = next(self._sampler_iter)
        data = self._dataset_fetcher.fetch(indices)
        if self._pin_memory:
            data = pin_memory(data)
        return data

    # python2 compatibility
    def next(self):
        return self.__next__()


class _DataLoaderIterMultiProcess(_DataLoaderIterBase):
    def __init__(self, loader):
        super(_DataLoaderIterMultiProcess, self).__init__(loader)

        assert self._num_workers > 0,  "Multi-process DataLoader " \
                    "invalid num_workers({})".format(self._num_workers)

        # subprocess wrokers' result queue
        self._worker_result_queue = None
        # queue for output data
        self._data_queue = None

        # data get from _worker_result_queue will be reordered by _rcvd_idx
        # for data order keeping, data index not equal _rcvd_idx 
        # will be cached in _task_infos
        self._send_idx = 0
        self._rcvd_idx = 0
        self._batches_outstanding = 0
        self._task_infos = {}

        # indices outstand as _outstanding_capacity at first, and
        # blocking_queue capacity is also _outstanding_capacity.
        # _outstanding_capacity here to make sure each indices_queue
        # has at least 2 indices, and outstanding batch cached
        # output data for at least 2 iterations(Note that len(_places)
        # batches will be composed as an iteration output)
        self._outstanding_capacity = 2 * max(self._num_workers,
                                             len(self._places))

        # init workers and indices queues and put 2 indices in each indices queue
        self._init_workers()
        for _ in range(self._outstanding_capacity):
            self._try_put_indices()

        self._init_pin_memory_thread()

        self._shutdown = False

    def _init_workers(self):
        # multiprocess worker and indice queue list initial as empty
        self._workers = []
        self._worker_status = []
        self._indices_queues = []
        self._workers_idx_cycle = itertools.cycle(range(self._num_workers))

        # create data_queue for workers
        self._worker_result_queue = multiprocessing.Queue()

        # event for workers and thread, thread event is only need 
        # in multi-processing mode
        self._workers_done_event = multiprocessing.Event()
        self._thread_done_event = threading.Event()

        for i in range(self._num_workers):
            indices_queue = multiprocessing.Queue()
            self._indices_queues.append(indices_queue)
            worker = multiprocessing.Process(
                target=_worker_loop,
                args=(self._dataset, self._dataset_kind, indices_queue,
                      self._worker_result_queue, self._workers_done_event,
                      self._auto_collate_batch, self._collate_fn,
                      self._worker_init_fn, i, self._num_workers,
                      self._use_shared_memory))
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
            self._worker_status.append(True)

        core._set_process_pids(id(self), tuple(w.pid for w in self._workers))
        _set_SIGCHLD_handler()

    def _init_pin_memory_thread(self):
        if self._pin_memory:
            self._data_queue = queue.Queue()
            self._pin_memory_thread_done_event = threading.Event()
            self._pin_memory_thread = threading.Thread(
                target=_pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue,
                      self._pin_memory_thread_done_event,
                      _current_expected_place()))
            self._pin_memory_thread.daemon = True
            self._pin_memory_thread.start()
        else:
            self._data_queue = self._worker_result_queue

    def _clear_and_remove_queues(self):
        queues = [self._data_queue]
        if self._pin_memory:
            queues.append(self._worker_result_queue)
        for queue_ in queues:
            if queue is not None:
                while True:
                    try:
                        queue_.get_nowait()
                    except:
                        queue_.cancel_join_thread()
                        queue_.close()
                        break

    def _shutdown_worker(self, worker_id):
        if self._worker_status[worker_id]:
            self._indices_queues[worker_id].put(None)
            self._worker_status[worker_id] = False

    def _try_shutdown_all(self):
        if not self._shutdown:
            try:
                # exit pin_memory thread
                if self._pin_memory and hasattr(self, '_pin_memory_thread'):
                    self._pin_memory_thread_done_event.set()
                    # send something to wake up pin_memory thread to check
                    # done event
                    self._worker_result_queue.put((None, None))
                    self._pin_memory_thread.join()
                    self._worker_result_queue.cancel_join_thread()
                    self._worker_result_queue.close()

                # set _workers_done_event should be set before put None
                # to indices_queue, workers wll exit on reading None from
                # indices_queue
                self._workers_done_event.set()
                for i in range(self._num_workers):
                    self._shutdown_worker(i)

                for w in self._workers:
                    # try to kill worker peacefully, if failed, kill
                    # the worker forcely
                    w.join(timeout=MP_STATUS_CHECK_INTERVAL)
                    if w.is_alive():
                        w.terminate()
                for q in self._indices_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                core._erase_process_pids(id(self))
                self._shutdown = True

    def _try_get_data(self, timeout=MP_STATUS_CHECK_INTERVAL):
        try:
            # [ avoid hang ]: main process may blocking at _reader.read_next when
            # KeyboardInterrupt, we do following tradeoff:
            # 1. get data with timeout, MP_STATUS_CHECK_INTERVAL(5s) as timeout
            #    default, if KeyboardInterrupt blocking, failed workers will be
            #    checked and raise RuntimeError to quit DataLoader in timeout
            #    exception handling.
            # 2. if get data timeout and check workers all alive, continue to
            #    get data again
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            # check failed workers
            failed_workers = []
            for i, w in enumerate(self._workers):
                if self._worker_status[i] and not w.is_alive():
                    failed_workers.append(w)
                    self._shutdown_worker(i)
            if len(failed_workers) > 0:
                pids = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError("DataLoader {} workers exit unexpectedly, " \
                            "pids: {}".format(len(failed_workers), pids))

            # get(timeout) will call _poll(timeout) and may raise IOError
            if isinstance(e, queue.Empty) or isinstance(e, IOError):
                return (False, None)

    def _get_data(self):
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.
                                   format(self._timeout))
        elif self._pin_memory:
            while self._pin_memory_thread.is_alive():
                success, data = self._try_get_data()
                if success:
                    return data
            else:
                raise RuntimeError('Pin memory thread exited unexpectedly')
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _next_data(self):
        while True:
            while self._rcvd_idx < self._send_idx:
                info = self._task_infos[self._rcvd_idx]
                if len(info) == 2 or self._worker_status[info[0]]:
                    break
                del self._task_infos[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # if _rcvd_idx = _send_idx, which mean all send data has
                # been received and yielded and outstanding batch number
                # is 0, this will only occured on epoch end, so raise
                # StopIteration to end this epoch iteration
                self._try_shutdown_all()
                raise StopIteration

            if self._rcvd_idx in self._task_infos and \
                    len(self._task_infos[self._rcvd_idx]) == 2:
                return self._process_data(
                    self._task_infos.pop(self._rcvd_idx)[1])

            assert not self._shutdown and self._batches_outstanding > 0

            idx, data = self._get_data()
            self._batches_outstanding -= 1

            if self._dataset_kind == _DatasetKind.ITER:
                if isinstance(data, _IterableDatasetStopIteration):
                    self._shutdown_worker(data.worker_id)
                    self._try_put_indices()
                    continue

            if idx != self._rcvd_idx:
                # cache inorder data
                self._task_infos[idx] += (data, )
            else:
                del self._task_infos[idx]
                return self._process_data(data)

    def _try_put_indices(self):
        assert self._batches_outstanding <= self._outstanding_capacity, \
                    "too many indices have been put to queue"
        try:
            indices = next(self._sampler_iter)
        except StopIteration:
            return

        for i in range(self._num_workers):
            worker_idx = next(self._workers_idx_cycle)
            if self._worker_status[worker_idx]:
                break
        else:
            return

        self._indices_queues[worker_idx].put((self._send_idx, indices))
        self._task_infos[self._send_idx] = (worker_idx, )
        self._batches_outstanding += 1
        self._send_idx += 1

    def __del__(self):
        self._try_shutdown_all()

    def __next__(self):
        try:
            data = self._next_data()
            return data
        except StopIteration:
            self._try_shutdown_all()
            six.reraise(*sys.exc_info())

    # python2 compatibility
    def next(self):
        return self.__next__()

    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_indices()
        if isinstance(data, Exception):
            raise data
        return data
