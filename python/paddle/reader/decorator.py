# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from threading import Thread
import subprocess
import multiprocessing
import six
import sys
import warnings

from six.moves.queue import Queue
from six.moves import zip_longest
from six.moves import map
from six.moves import zip
import itertools
import random
import zlib

import paddle.compat as cpt
from paddle.fluid.reader import QUEUE_GET_TIMEOUT

__all__ = []

# On macOS, the 'spawn' start method is now the default in Python3.8 multiprocessing,
# Paddle is currently unable to solve this, so forces the process to start using 
# the 'fork' start method.
#
# TODO: This solution is not good, because the fork start method could lead to 
# crashes of the subprocess. Figure out how to make 'spawn' work.
#
# For more details, please refer to
# https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
# https://bugs.python.org/issue33725
if sys.version_info >= (3, 8) and sys.platform == 'darwin':
    fork_context = multiprocessing.get_context('fork')
else:
    fork_context = multiprocessing


def cache(reader):
    """
    Cache the reader data into memory. 

    Be careful that this method may take long time to process, 
    and consume lots of memory. :code:`reader()` would only 
    call once. 

    Args:
        reader (generator): a reader object which yields 
            data each time.

    Returns:
        generator: a decorated reader object which yields data from cached memory.
    
    Examples:
        .. code-block:: python

            import paddle
            
            def reader():
                for i in range(3):
                    yield i
            
            # All data is cached into memory
            cached_reader = paddle.io.cache(reader)
            
            # Output: 0 1 2
            for i in cached_reader():
                print(i)
    """
    all_data = tuple(reader())

    def __impl__():
        for item in all_data:
            yield item

    return __impl__


def map_readers(func, *readers):
    """
    Creates a data reader that outputs return value of function using
    output of each data reader as arguments.

    If input readers output the following data entries: 2 3,
    and the input func is mul(x, y),
    the output of the resulted reader will be 6.


    Args:
        func: a function to read data and compute result, the output of this function 
              will be set as the output of the resulted data reader.
        readers (Reader|list of Reader): list of readers whose outputs will be used as arguments of func.
 
    Returns:
        the resulted data reader (Reader)

    Examples:

        .. code-block:: python

         import paddle.reader
         d = {"h": 0, "i": 1}
         def func(x):
             return d[x]
         def reader():
             yield "h"
             yield "i"
         map_reader_result = paddle.reader.map_readers(func, reader)
    """

    def reader():
        rs = []
        for r in readers:
            rs.append(r())
        for e in map(func, *rs):
            yield e

    return reader


def shuffle(reader, buf_size):
    """
    paddle.fluid.io.shuffle ( :ref:`api_fluid_io_shuffle` ) is recommended to use,
    and paddle.reader.shuffle is an alias.

    This API creates a decorated reader that outputs the shuffled data.

    The output data from the origin reader will be saved into a buffer, 
    and then shuffle the data. The size of buffer is determined by argument buf_size.
 
    Args:
        reader(callable): the original reader whose data will be shuffled.
        buf_size(int): the size of shuffled buffer.

    Returns:
        callable: a decorated reader.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            def reader():
                for i in range(5):
                    yield i
            shuffled_reader = fluid.io.shuffle(reader, 3)
            for e in shuffled_reader():
                print(e)
            # outputs are 0~4 unordered arrangement
    """

    def data_reader():
        buf = []
        for e in reader():
            buf.append(e)
            if len(buf) >= buf_size:
                random.shuffle(buf)
                for b in buf:
                    yield b
                buf = []

        if len(buf) > 0:
            random.shuffle(buf)
            for b in buf:
                yield b

    return data_reader


class XmapEndSignal():
    pass


def xmap_readers(mapper, reader, process_num, buffer_size, order=False):
    """
    Use multi-threads to map samples from reader by a mapper defined by user.

    Args:
        mapper (callable): a function to map the data from reader.
        reader (callable): a data reader which yields the data. 
        process_num (int): thread number to handle original sample.
        buffer_size (int): size of the queue to read data in. 
        order (bool): whether to keep the data order from original reader. 
            Default False.

    Returns:
        callable: a decorated reader with data mapping. 
    """
    end = XmapEndSignal()

    # define a worker to read samples from reader to in_queue
    def read_worker(reader, in_queue):
        for i in reader():
            in_queue.put(i)
        in_queue.put(end)

    # define a worker to read samples from reader to in_queue with order flag
    def order_read_worker(reader, in_queue):
        in_order = 0
        for i in reader():
            in_queue.put((in_order, i))
            in_order += 1
        in_queue.put(end)

    # define a worker to handle samples from in_queue by mapper
    # and put mapped samples into out_queue
    def handle_worker(in_queue, out_queue, mapper):
        sample = in_queue.get()
        while not isinstance(sample, XmapEndSignal):
            r = mapper(sample)
            out_queue.put(r)
            sample = in_queue.get()
        in_queue.put(end)
        out_queue.put(end)

    # define a worker to handle samples from in_queue by mapper
    # and put mapped samples into out_queue by order
    def order_handle_worker(in_queue, out_queue, mapper, out_order):
        ins = in_queue.get()
        while not isinstance(ins, XmapEndSignal):
            order, sample = ins
            r = mapper(sample)
            while order != out_order[0]:
                pass
            out_queue.put(r)
            out_order[0] += 1
            ins = in_queue.get()
        in_queue.put(end)
        out_queue.put(end)

    def xreader():
        in_queue = Queue(buffer_size)
        out_queue = Queue(buffer_size)
        out_order = [0]
        # start a read worker in a thread
        target = order_read_worker if order else read_worker
        t = Thread(target=target, args=(reader, in_queue))
        t.daemon = True
        t.start()
        # start several handle_workers
        target = order_handle_worker if order else handle_worker
        args = (in_queue, out_queue, mapper, out_order) if order else (
            in_queue, out_queue, mapper)
        workers = []
        for i in range(process_num):
            worker = Thread(target=target, args=args)
            worker.daemon = True
            workers.append(worker)
        for w in workers:
            w.start()

        sample = out_queue.get()
        while not isinstance(sample, XmapEndSignal):
            yield sample
            sample = out_queue.get()
        finish = 1
        while finish < process_num:
            sample = out_queue.get()
            if isinstance(sample, XmapEndSignal):
                finish += 1
            else:
                yield sample

    return xreader
