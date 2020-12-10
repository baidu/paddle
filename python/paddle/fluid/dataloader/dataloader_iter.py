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
from collections import namedtuple

# NOTE: queue has a different name in python2 and python3
if six.PY2:
    import Queue as queue
else:
    import queue

import paddle
from .. import core, layers
from ..framework import in_dygraph_mode
from ..multiprocess_utils import CleanupFuncRegistrar, _cleanup_mmap, _set_SIGCHLD_handler
from .fetcher import _IterableDatasetFetcher, _MapDatasetFetcher
from .batch_sampler import _InfiniteIterableSampler

__all__ = ['get_worker_info']

# multi-process worker check indices queue interval, avoid
# hanging in subprocess data loading
MP_INDICES_CHECK_INTERVAL = 5

_IterableDatasetStopIteration = namedtuple('_IterableDatasetStopIteration',
                                           ['worker_id'])


def default_collate_fn(batch):
    """
    Default batch collating function for :code:`fluid.io.DataLoader`,
    batch should be a list of samples, and each sample should be a list
    of fields as follows:
    
    [[filed1, filed2, ...], [filed1, filed2, ...], ...]
    
    This default collate function zipped each filed together and stack
    each filed as the batch field as follows:

    [batch_filed1, batch_filed2, ...]

    Args:  
        batch(list of list of numpy array): the batch data, each fields
              should be a numpy array, each sample should be a list of
              fileds, and batch should be a list of sample.
    
    Returns:
        a list of numpy array: collated batch
    """
    sample = batch[0]
    # dataset has only 1 field
    if isinstance(sample, np.ndarray):
        return [np.stack(batch, axis=0)]

    # batch each field
    slots = []
    for items in batch:
        for i, item in enumerate(items):
            if len(slots) < len(items):
                slots.append([item])
            else:
                slots[i].append(item)

    outputs = []
    for slot in slots:
        if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
            tmp = np.stack(slot, axis=0)
            outputs.append(tmp)
        elif isinstance(slot[0], paddle.Tensor):
            tmp = layers.stack(slot, axis=0)
            outputs.append(tmp)
        else:
            raise RuntimeError("Unknown data type {}".format(type(slot[0])))
    return outputs


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
        self._timeout = loader.timeout if loader.timeout > 0 else MP_INDICES_CHECK_INTERVAL
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
            self._collate_fn = loader.collate_fn

        # LoDTensorBlockingQueue instance for create_py_reader and a thread
        # to put mini-batch data to self._blocking_queue, mini-batch data
        # will be get from:
        # 1. multi-process mode: get data from workers' result queue
        # 2. single-process mode: read mini-batch data in main process
        self._blocking_queue = None
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

        # NOTE: len(self._places) batch data compose as an output
        # iteration, set blocking_queue can cache 2 iteration datas
        # at most here
        self._blocking_queue_capacity = 2 * len(self._places)

        self._init_thread()

    def _init_thread(self):
        self._var_names = [v.name for v in self._feed_list]
        self._shapes = [v.shape for v in self._feed_list]
        self._dtypes = [v.dtype for v in self._feed_list]
        self._need_check_feed = [
            v.desc.need_check_feed() for v in self._feed_list
        ]
        # if only 1 place, do not need to keep order
        self._blocking_queue = core.init_lod_tensor_blocking_queue(
            core.Variable(), self._blocking_queue_capacity,
            len(self._places) > 1)
        self._reader = core.create_py_reader(
            self._blocking_queue, self._var_names, self._shapes, self._dtypes,
            self._need_check_feed, self._places, self._use_buffer_reader, True,
            self._pin_memory)

        self._thread = threading.Thread(target=self._thread_loop)
        self._thread.daemon = True
        self._thread.start()

    def _thread_loop(self):
        try:
            for indices in self._sampler_iter:
                # read data from dataset in mini-batch
                batch = self._dataset_fetcher.fetch(indices)

                # pack as LoDTensorArray
                array = core.LoDTensorArray()
                for slot in batch:
                    if not isinstance(slot, core.LoDTensor):
                        self._check_input_array(slot)
                        # FIXME(dkp): blocking_queue only support
                        #             core.LoDTensorArray as input now, read
                        #             numpy data into a LoDTensorArray here,
                        #             should support paddle.Tensor list later
                        if isinstance(slot, paddle.Tensor):
                            slot = slot.numpy()
                        tmp = core.LoDTensor()
                        tmp.set(slot, core.CPUPlace())
                        slot = tmp

                    array.append(slot)

                if not self._blocking_queue.push(array):
                    break

            self._blocking_queue.close()
            self._thread = None
        except StopIteration:
            self._blocking_queue.close()
        except Exception:
            self._blocking_queue.kill()
            self._thread = None
            logging.warning("DataLoader reader thread raised an exception.")
            six.reraise(*sys.exc_info())

    @classmethod
    def _check_input_array(cls, item):
        if isinstance(item, paddle.Tensor):
            return
        arr = np.array(item)
        if arr.dtype == np.object:
            raise TypeError((
                "\n\tFaild to convert input data to a regular ndarray :\n\t* Usually "
                "this means the input data contains nested lists with different lengths. "
                "\n\t* Check the reader function passed to 'decorate_batch_generator'"
                " to locate the data causes this issue.\n\t* Please consider using "
                "'fluid.create_lod_tensor' to convert it to a LoD-Tensor."))

    def __next__(self):
        try:
            if in_dygraph_mode():
                return self._reader.read_next_var_list()
            else:
                if self._return_list:
                    # static graph organized data on multi-device with list, if
                    # place number is 1, there is only 1 device, extra the data
                    # from list for devices to be compatible with dygraph mode
                    if len(self._places) == 1:
                        return self._reader.read_next_list()[0]
                    else:
                        return self._reader.read_next_list()
                else:
                    return self._reader.read_next()
        except StopIteration:
            self._reader.reset()
            six.reraise(*sys.exc_info())

    # python2 compatibility
    def next(self):
        return self.__next__()

    def __del__(self):
        # _blocking_queue in keep order mode holds sub-threads
        # need to release thread resources on unexpected exit
        if self._blocking_queue:
            self._blocking_queue.close()


# NOTE(chenweihang): _worker_loop must be top level method to be pickled
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
                data = indices_queue.get(MP_INDICES_CHECK_INTERVAL)
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
                    out_queue.put(_IterableDatasetStopIteration(worker_id))
                    iterator_drained = True
                else:
                    out_queue.put((idx, e))
            else:
                if use_shared_memory:
                    # FIXME(dkp): _convert_to_tensor_list only support np.array
                    #             list now, should support paddle.Tensor list
                    if isinstance(batch[0][0], paddle.Tensor):
                        np_batch = []
                        for sample in batch:
                            np_batch.append([s.numpy() for s in sample])
                        batch = np_batch

                    tensor_list = core._convert_to_tensor_list(batch)
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


class _DataLoaderIterMultiProcess(_DataLoaderIterBase):
    def __init__(self, loader):
        super(_DataLoaderIterMultiProcess, self).__init__(loader)

        assert self._num_workers > 0,  "Multi-process DataLoader " \
                    "invalid num_workers({})".format(self._num_workers)

        # subprocess wrokers' result queue
        self._data_queue = None

        # data get from _data_queue will be reordered by _rcvd_idx
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

        # see _try_put_indices
        self._thread_lock = threading.Lock()

        # init workers and indices queues and put 2 indices in each indices queue
        self._init_workers()
        for _ in range(self._outstanding_capacity):
            self._try_put_indices()

        self._init_thread()
        self._shutdown = False

    def _init_workers(self):
        # multiprocess worker and indice queue list initial as empty
        self._workers = []
        self._worker_status = []
        self._indices_queues = []
        self._workers_idx_cycle = itertools.cycle(range(self._num_workers))

        # create data_queue for workers
        self._data_queue = multiprocessing.Queue()

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
                      self._data_queue, self._workers_done_event,
                      self._auto_collate_batch, self._collate_fn,
                      self._worker_init_fn, i, self._num_workers,
                      self._use_shared_memory))
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
            self._worker_status.append(True)

        core._set_process_pids(id(self), tuple(w.pid for w in self._workers))
        _set_SIGCHLD_handler()

    def _clear_and_remove_data_queue(self):
        if self._data_queue is not None:
            while True:
                try:
                    self._data_queue.get_nowait()
                except:
                    self._data_queue.cancel_join_thread()
                    self._data_queue.close()
                    break

    def _init_thread(self):
        self._var_names = [v.name for v in self._feed_list]
        self._shapes = [v.shape for v in self._feed_list]
        self._dtypes = [v.dtype for v in self._feed_list]
        self._need_check_feed = [
            v.desc.need_check_feed() for v in self._feed_list
        ]
        # if only 1 place, do not need to keep order
        self._blocking_queue = core.init_lod_tensor_blocking_queue(
            core.Variable(), self._outstanding_capacity, len(self._places) > 1)
        self._reader = core.create_py_reader(
            self._blocking_queue, self._var_names, self._shapes, self._dtypes,
            self._need_check_feed, self._places, self._use_buffer_reader, True,
            self._pin_memory)

        self._thread_done_event = threading.Event()
        self._thread = threading.Thread(target=self._thread_loop)
        self._thread.daemon = True
        self._thread.start()

    def _shutdown_worker(self, worker_id):
        if self._worker_status[worker_id]:
            self._indices_queues[worker_id].put(None)
            self._worker_status[worker_id] = False

    def _try_shutdown_all(self):
        if not self._shutdown:
            try:
                self._exit_thread_expectedly()
                self._clear_and_remove_data_queue()

                # set _workers_done_event should be set before put None
                # to indices_queue, workers wll exit on reading None from
                # indices_queue
                self._workers_done_event.set()
                for i in range(self._num_workers):
                    self._shutdown_worker(i)

                for w in self._workers:
                    w.join()
                for q in self._indices_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                core._erase_process_pids(id(self))
                self._shutdown = True

    def _exit_thread_expectedly(self):
        self._thread_done_event.set()
        self._blocking_queue.close()

    def _exit_thread_unexpectedly(self):
        self._thread_done_event.set()
        self._blocking_queue.kill()
        logging.error("DataLoader reader thread raised an exception!")

    def _thread_loop(self):
        while not self._thread_done_event.is_set():
            batch = self._get_data()
            if not self._thread_done_event.is_set():
                if batch is None:
                    self._exit_thread_expectedly()
                elif isinstance(batch, Exception):
                    self._exit_thread_unexpectedly()
                else:
                    try:
                        # pack as LoDTensorArray
                        array = core.LoDTensorArray()
                        if self._use_shared_memory:
                            for tensor in batch:
                                array.append(tensor)
                        else:
                            # LoDTensor not in shared memory is not
                            # serializable, cannot be create in workers
                            for slot in batch:
                                if not isinstance(slot, core.LoDTensor):
                                    tmp = core.LoDTensor()
                                    tmp.set(slot, core.CPUPlace())
                                    slot = tmp
                                array.append(slot)

                        if not self._blocking_queue.push(array):
                            self._blocking_queue.close()
                    except:
                        self._exit_thread_unexpectedly()
                        six.reraise(*sys.exc_info())
                    finally:
                        self._rcvd_idx += 1

    def _get_data(self):
        while not self._thread_done_event.is_set():
            # For IterableDataset, batch indices is generated infinitely
            # for each worker to raise StopIteration, but a StopIteration
            # raising process will discard a batch indices which is count
            # in _send_idx but will not increase _rcvd_idx, so we check 
            # whether the worker is still alive here to skip the discarded
            # batch indices and increase _rcvd_idx
            if self._dataset_kind == _DatasetKind.ITER:
                while self._rcvd_idx < self._send_idx:
                    info = self._task_infos[self._rcvd_idx]
                    if len(info) == 2 or self._worker_status[info[0]]:
                        break
                    del self._task_infos[self._rcvd_idx]
                    self._rcvd_idx += 1
                    self._batches_outstanding -= 1
                else:
                    # NOTE: _rcvd_idx and _send_idx only record batches among
                    #       workers, if batches among workers drained, there
                    #       may also be data in blocking queue
                    if self._batches_outstanding < len(self._places):
                        return None
                    continue

            if self._rcvd_idx in self._task_infos and \
                    len(self._task_infos[self._rcvd_idx]) == 2:
                return self._task_infos.pop(self._rcvd_idx)[1]

            try:
                # [ avoid hang ]: main process may blocking at _reader.read_next when
                # KeyboardInterrupt, we do following tradeoff:
                # 1. get data with timeout, MP_INDICES_CHECK_INTERVAL(5s) as timeout
                #    default, if KeyboardInterrupt blocking, failed workers will be
                #    checked and raise RuntimeError to quit DataLoader in timeout
                #    exception handling.
                # 2. if get data timeout and check workers all alive, continue to
                #    get data again
                data = self._data_queue.get(timeout=self._timeout)
            except Exception as e:
                # check if thread done event set when waiting data
                if self._thread_done_event.is_set():
                    continue

                # check failed workers
                failed_workers = []
                for i, w in enumerate(self._workers):
                    if self._worker_status[i] and not w.is_alive():
                        failed_workers.append(w)
                        self._shutdown_worker(i)
                if len(failed_workers) > 0:
                    self._exit_thread_unexpectedly()
                    pids = ', '.join(str(w.pid) for w in failed_workers)
                    raise RuntimeError("DataLoader {} workers exit unexpectedly, " \
                                "pids: {}".format(len(failed_workers), pids))

                # get(timeout) will call _poll(timeout) and may raise IOError
                if isinstance(e, queue.Empty) or isinstance(e, IOError):
                    # continue on timeout to keep getting data from queue
                    continue

                self._exit_thread_unexpectedly()
                logging.error("DataLoader reader thread failed({}) to read data from " \
                              "workers' result queue.".format(e))
                six.reraise(*sys.exc_info())
            else:
                if self._dataset_kind == _DatasetKind.ITER and isinstance(
                        data, _IterableDatasetStopIteration):
                    # if a worker get StopIteraion, we shutdown this worker,
                    # note that this batch indices to trigger StopIteration
                    # is discard, outstanding batch number should be decrease
                    # and another indices should be put for other workers
                    # may still working.
                    self._shutdown_worker(data.worker_id)
                    self._batches_outstanding -= 1
                    self._try_put_indices()
                    continue

                idx, batch = data
                if idx == self._rcvd_idx:
                    del self._task_infos[idx]
                    return batch
                else:
                    self._task_infos[idx] += (batch, )
                    continue

    def _try_put_indices(self):
        assert self._batches_outstanding <= self._outstanding_capacity, \
                    "too many indices have been put to queue"
        # In multi-process mode for IterableDataset, _try_put_indices will
        # be called both in main process(for our implement has blocking queue,
        # and blocking queue read is in main process) and thread, which may
        # cause error following error
        #   1. "ValueError: generator already executing" in next(self._sampler_iter)
        #   2. re-enter in increase _send_idx
        # add a lock for threading save, for _try_put_indices is only a slight
        # function which is not in data reading pipeline, this lock almost no
        # influence on performance
        with self._thread_lock:
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
            # _batches_outstanding here record the total batch data number
            # in 'from after _try_put_indices to beforeoutput data', this
            # value should be _outstanding_capacity if data is not drained,
            # if _batches_outstanding is less than _places number, there are
            # no enough data to generate next output, close blocking_queue and
            # set _thread_done_event here, py_reader will raise StopIteration,
            # end workers and indices_queues in StopIteration handling
            if self._batches_outstanding < len(self._places):
                self._thread_done_event.set()
                self._blocking_queue.close()

            if in_dygraph_mode():
                data = self._reader.read_next_var_list()
            else:
                if self._return_list:
                    data = self._reader.read_next_list()
                    # static graph organized data on multi-device with list, if
                    # place number is 1, there is only 1 device, extra the data
                    # from list for devices to be compatible with dygraph mode
                    if len(self._places) == 1:
                        data = data[0]
                else:
                    data = self._reader.read_next()
            self._on_output_batch()
            return data
        except StopIteration:
            self._reader.reset()
            self._try_shutdown_all()
            six.reraise(*sys.exc_info())

    # python2 compatibility
    def next(self):
        return self.__next__()

    def _on_output_batch(self):
        for _ in range(len(self._places)):
            self._batches_outstanding -= 1
            self._try_put_indices()
