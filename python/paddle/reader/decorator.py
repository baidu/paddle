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

__all__ = [
    'map_readers', 'buffered', 'compose', 'chain', 'shuffle',
    'ComposeNotAligned', 'firstn', 'xmap_readers', 'PipeReader'
]

from threading import Thread
import subprocess

from six.moves.queue import Queue
from six.moves import zip_longest
import itertools
import random
import zlib


def map_readers(func, *readers):
    """
    Creates a data reader that outputs return value of function using
    output of each data readers as arguments.

    :param func: function to use. The type of func should be (Sample) => Sample
    :type: callable
    :param readers: readers whose outputs will be used as arguments of func.
    :return: the created data reader.
    :rtype: callable
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
    Creates a data reader whose data output is shuffled.

    Output from the iterator that created by original reader will be
    buffered into shuffle buffer, and then shuffled. The size of shuffle buffer
    is determined by argument buf_size.

    :param reader: the original reader whose output will be shuffled.
    :type reader: callable
    :param buf_size: shuffle buffer size.
    :type buf_size: int

    :return: the new reader whose output is shuffled.
    :rtype: callable
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


def chain(*readers):
    """
    Creates a data reader whose output is the outputs of input data
    readers chained together.

    If input readers output following data entries:
    [0, 0, 0]
    [1, 1, 1]
    [2, 2, 2]
    The chained reader will output:
    [0, 0, 0, 1, 1, 1, 2, 2, 2]

    :param readers: input readers.
    :return: the new data reader.
    :rtype: callable
    """

    def reader():
        rs = []
        for r in readers:
            rs.append(r())

        for e in itertools.chain(*rs):
            yield e

    return reader


class ComposeNotAligned(ValueError):
    pass


def compose(*readers, **kwargs):
    """
    Creates a data reader whose output is the combination of input readers.

    If input readers output following data entries:
    (1, 2)    3    (4, 5)
    The composed reader will output:
    (1, 2, 3, 4, 5)

    :param readers: readers that will be composed together.
    :param check_alignment: if True, will check if input readers are aligned
        correctly. If False, will not check alignment and trailing outputs
        will be discarded. Defaults to True.
    :type check_alignment: bool

    :return: the new data reader.

    :raises ComposeNotAligned: outputs of readers are not aligned.
        Will not raise when check_alignment is set to False.
    """
    check_alignment = kwargs.pop('check_alignment', True)

    def make_tuple(x):
        if isinstance(x, tuple):
            return x
        else:
            return (x, )

    def reader():
        rs = []
        for r in readers:
            rs.append(r())
        if not check_alignment:
            for outputs in zip(*rs):
                yield sum(list(map(make_tuple, outputs)), ())
        else:
            for outputs in zip_longest(*rs):
                for o in outputs:
                    if o is None:
                        # None will be not be present if compose is aligned
                        raise ComposeNotAligned(
                            "outputs of readers are not aligned.")
                yield sum(list(map(make_tuple, outputs)), ())

    return reader


def buffered(reader, size):
    """
    Creates a buffered data reader.

    The buffered data reader will read and save data entries into a
    buffer. Reading from the buffered data reader will proceed as long
    as the buffer is not empty.

    :param reader: the data reader to read from.
    :type reader: callable
    :param size: max buffer size.
    :type size: int

    :returns: the buffered data reader.
    """

    class EndSignal():
        pass

    end = EndSignal()

    def read_worker(r, q):
        for d in r:
            q.put(d)
        q.put(end)

    def data_reader():
        r = reader()
        q = Queue(maxsize=size)
        t = Thread(
            target=read_worker, args=(
                r,
                q, ))
        t.daemon = True
        t.start()
        e = q.get()
        while e != end:
            yield e
            e = q.get()

    return data_reader


def firstn(reader, n):
    """
    Limit the max number of samples that reader could return.

    :param reader: the data reader to read from.
    :type reader: callable
    :param n: the max number of samples that return.
    :type n: int
    :return: the decorated reader.
    :rtype: callable
    """

    # TODO(yuyang18): Check if just drop the reader, could clean the opened
    # resource or not?

    def firstn_reader():
        for i, item in enumerate(reader()):
            if i == n:
                break
            yield item

    return firstn_reader


class XmapEndSignal():
    pass


def xmap_readers(mapper, reader, process_num, buffer_size, order=False):
    """
    Use multiprocess to map samples from reader by a mapper defined by user.
    And this function contains a buffered decorator.
    :param mapper:  a function to map sample.
    :type mapper: callable
    :param reader: the data reader to read from
    :type reader: callable
    :param process_num: process number to handle original sample
    :type process_num: int
    :param buffer_size: max buffer size
    :type buffer_size: int
    :param order: keep the order of reader
    :type order: bool
    :return: the decarated reader
    :rtype: callable
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


def _buf2lines(buf, line_break="\n"):
    # FIXME: line_break should be automatically configured.
    lines = buf.split(line_break)
    return lines[:-1], lines[-1]


class PipeReader:
    """
        PipeReader read data by stream from a command, take it's
        stdout into a pipe buffer and redirect it to the parser to
        parse, then yield data as your desired format.

        You can using standard linux command or call another program
        to read data, from HDFS, Ceph, URL, AWS S3 etc:

        .. code-block:: python
           cmd = "hadoop fs -cat /path/to/some/file"
           cmd = "cat sample_file.tar.gz"
           cmd = "curl http://someurl"
           cmd = "python print_s3_bucket.py"

        An example:

        .. code-block:: python

           def example_reader():
               for f in myfiles:
                   pr = PipeReader("cat %s"%f)
                   for l in pr.get_line():
                       sample = l.split(" ")
                       yield sample
    """

    def __init__(self, command, bufsize=8192, file_type="plain"):
        if not isinstance(command, str):
            raise TypeError("left_cmd must be a string")
        if file_type == "gzip":
            self.dec = zlib.decompressobj(
                32 + zlib.MAX_WBITS)  # offset 32 to skip the header
        self.file_type = file_type
        self.bufsize = bufsize
        self.process = subprocess.Popen(
            command.split(" "), bufsize=bufsize, stdout=subprocess.PIPE)

    def get_line(self, cut_lines=True, line_break="\n"):
        """
        :param cut_lines: cut buffer to lines
        :type cut_lines: bool
        :param line_break: line break of the file, like \n or \r
        :type line_break: string

        :return: one line or a buffer of bytes
        :rtype: string
        """
        remained = ""
        while True:
            buff = self.process.stdout.read(self.bufsize)
            if buff:
                if self.file_type == "gzip":
                    decomp_buff = self.dec.decompress(buff)
                elif self.file_type == "plain":
                    decomp_buff = buff
                else:
                    raise TypeError("file_type %s is not allowed" %
                                    self.file_type)

                if cut_lines:
                    lines, remained = _buf2lines(''.join(
                        [remained, decomp_buff]), line_break)
                    for line in lines:
                        yield line
                else:
                    yield decomp_buff
            else:
                break
