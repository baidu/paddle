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

import os
import contextlib
from . import core
__all__ = [
    'convert_reader_to_recordio_file', 'convert_reader_to_recordio_files'
]


@contextlib.contextmanager
def create_recordio_writer(filename,
                           compressor=core.RecordIOWriter.Compressor.Snappy,
                           max_num_records=1000):
    writer = core.RecordIOWriter(filename, compressor, max_num_records)
    yield writer
    writer.close()


def convert_reader_to_recordio_file(
        filename,
        reader_creator,
        feeder,
        compressor=core.RecordIOWriter.Compressor.Snappy,
        max_num_records=1000,
        feed_order=None):
    """
    Convert a Python Reader to a recordio file.

    Please see :ref:`api_guide_python_reader` and :ref:`api_guide_reader_op` for
    details.

    Examples:

        >>> import paddle.fluid as fluid
        >>> import paddle.dataset.mnist as mnist
        >>> import paddle
        >>>
        >>> tmp_program = fluid.Program()
        >>> with fluid.program_guard(tmp_program):
        >>>     img = fluid.layers.data(name='img', shape=[784])
        >>>     label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        >>> feeder = fluid.DataFeeder(feed_list=[img, label], place=fluid.CPUPlace())
        >>> # mnist.recordio will be generated in current directory
        >>> fluid.recordio_writer.convert_reader_to_recordio_file(
        >>>                     filename="mnist.recordio",
        >>>                     reader_creator=paddle.batch(mnist.train(), batch_size=32),
        >>>                     feeder=feeder)

    Args:
        filename(str): The recordio filename.
        reader_creator(callable): The Python Reader Creator. See
            :ref:`api_guide_python_reader`.
        feeder(DataFeeder): The DataFeeder instance. Used to convert
            :code:`reader_creator` to :code: `lod_tensor`
        compressor: Must in fluid.core.RecordIOWriter.Compressor.Snappy or
            fluid.core.RecordIOWriter.Compressor.NoCompress. Use :code:`Snappy`
            by default.
        max_num_records(int): Maximum number of records in one chuck. Each record
            is each return value from reader function
        feed_order(list): The order of variable names that the reader returns

    Returns:
        int: the number of record that saved.
    """
    if feed_order is None:
        feed_order = feeder.feed_names
    counter = 0
    with create_recordio_writer(filename, compressor,
                                max_num_records) as writer:
        for batch in reader_creator():
            res = feeder.feed(batch)
            for each in feed_order:
                writer.append_tensor(res[each])
            writer.complete_append_tensor()
            counter += 1
    return counter


def convert_reader_to_recordio_files(
        filename,
        batch_per_file,
        reader_creator,
        feeder,
        compressor=core.RecordIOWriter.Compressor.Snappy,
        max_num_records=1000,
        feed_order=None):
    """
    convert a python reader to many recordio files.

    This API is basically same as :code:`convert_reader_to_recordio_file`,
    instead of it will create many recordio files. Each file contains at
    most :code:`batch_per_file` records.

    Please reference
    :ref:`api_fluid_recordio_writer_convert_reader_to_recordio_file` for more
    details.
    """
    if feed_order is None:
        feed_order = feeder.feed_names
    f_name, f_ext = os.path.splitext(filename)
    assert (f_ext == ".recordio")

    lines = []
    f_idx = 0
    counter = 0
    for idx, batch in enumerate(reader_creator()):
        lines.append(batch)
        if idx >= batch_per_file and idx % batch_per_file == 0:
            filename = "%s-%05d%s" % (f_name, f_idx, f_ext)
            with create_recordio_writer(filename, compressor,
                                        max_num_records) as writer:
                for l in lines:
                    res = feeder.feed(l)
                    for each in feed_order:
                        writer.append_tensor(res[each])
                    writer.complete_append_tensor()
                    counter += 1
                lines = []
                f_idx += 1
    return counter
