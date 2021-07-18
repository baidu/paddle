# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from . import core
import sys
import six
import numpy as np
import threading
import paddle
from .framework import Program, Variable, program_guard, default_main_program, default_startup_program, in_dygraph_mode, cpu_places, _current_expected_place
from .executor import global_scope
from .data_feeder import DataFeeder, BatchedTensorProvider
from .multiprocess_utils import multiprocess_queue_set, CleanupFuncRegistrar, _cleanup_mmap, _cleanup, _set_SIGCHLD_handler
from .dataloader import BatchSampler, Dataset
from .dataloader.dataloader_iter import _DataLoaderIterSingleProcess, _DataLoaderIterMultiProcess, default_collate_fn
from .layers.io import monkey_patch_reader_methods, _copy_reader_var_, double_buffer
from .unique_name import UniqueNameGenerator
import logging
from .dataset import DatasetBase, InMemoryDataset

from .incubate.checkpoint import dataloader_auto_checkpoint as dacp
from .incubate.checkpoint import auto_checkpoint as acp
from . import unique_name

### Dygraph DataLoader configs ###
import os
import multiprocessing
import signal

# NOTE: queue has a different name in python2 and python3
if six.PY2:
    import Queue as queue
else:
    import queue

# NOTE: [ avoid hanging & failed quickly ] These value is used in getting data from another process
QUEUE_GET_TIMEOUT = 60

__all__ = ['PyReader', 'DataLoader', 'default_collate_fn']

data_loader_unique_name_generator = UniqueNameGenerator()

KEEP_DATA_LOADER_ORDER = True


def keep_data_loader_order(*args):
    global KEEP_DATA_LOADER_ORDER
    if len(args) == 0:
        return KEEP_DATA_LOADER_ORDER
    else:
        assert len(args) == 1 and isinstance(args[0], bool)
        KEEP_DATA_LOADER_ORDER = args[0]


def _convert_places(places):
    if not isinstance(places, (list, tuple)):
        places = [places]

    ret = []
    for p in places:
        if not isinstance(p, core.Place):
            tmp = core.Place()
            tmp.set_place(p)
            p = tmp

        ret.append(p)
    return ret


class DataLoaderBase(object):
    def __init__(self):
        self._places = None

    def __call__(self):
        return self

    def next(self):
        '''
        Get the next item in the DataLoader object. This method    
        should not be called by users directly. It is used for
        implementing iterator protocol of Python 2.x inside
        PaddlePaddle framework.
        '''
        return self.__next__()

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()

    @classmethod
    def _check_input_array(cls, item):
        arr = np.asarray(item)
        if arr.dtype == np.object:
            raise TypeError(
                "\n\tFaild to convert input data to a regular ndarray :\n\t* Usually "
                "this means the input data contains nested lists with different lengths. "
                "\n\t* Check the reader function passed to 'decorate_batch_generator'"
                " to locate the data causes this issue.\n\t* Please consider using "
                "'fluid.create_lod_tensor' to convert it to a LoD-Tensor.")
        return arr


class DataLoader(object):
    """
    DataLoader prodives an iterator which iterates given dataset
    once by the batch_sampler.

    DataLoader supports single-process and multi-prcess data loading,
    multi-process workers will be used to load data asynchronously if
    :attr:`num_workers` is set as a positive number.

    DataLoader only supports map-style dataset(can get a sample from
    dataset with a given index) currently, for a map-style dataset,
    please see :code:`paddle.io.Dataset`.

    batch_sampler please see :code:`paddle.io.BatchSampler`

    Args:  
        dataset(Dataset): the dataset to load data from, should be an
            instance of subclass of :code:`paddle.io.Dataset`.
        feed_list (list(Variable)|tuple(Variable)): feed variable list.
            The variables should be created by :code:`fluid.data()`.
            :attr:`feed_list` must be set if :attr:`return_list` is
            False. Default None.
        places(list(Place)|tuple(Place)): a list of Place, to put data
            onto, :attr:`places` must be set in both static graph and 
            dynamic graph mode, in dynamic graph mode, place number must
            be 1. Default None.
        return_list (bool): whether the return value on each device is 
            presented as a list. If :attr:`return_list=False`, the return
            value on each device would be a dict of str -> LoDTensor, where
            the key of the dict is the name of each fed variables. If 
            :attr:`return_list=True`, the return value on each device would
            be a list(LoDTensor). :attr:`return_list` can only be True
            in dynamic graph mode. Default False.
        batch_sampler(BatchSampler): an instance of `paddle.io.BatchSampler`
            to generate batch indices to draw samples from :attr:`dataset`
            and combine a batch. Default None.
        batch_size(int): sample number in a mini-batch, a substitution
            parameter for :attr:`batch_sampler`, if :attr:`batch_sampler`
            is not set, a default `paddle.io.BatchSampler` will be used
            and initialize by :attr:`batch_size`, :attr:`shuffle` and
            :attr:`drop_last`. Default 1.
        shuffle(bool): whther to shuffle indices order before genrate
            batch indices, a substitution parameter for :attr:`batch_sampler`
            see :attr:`batch_size`. Default False.
        drop_last(bool): whether drop the last incomplete batch dataset size
            is not divisible by the batch size, a substitution parameter
            for :attr:`batch_sampler`, see :attr:`batch_size`. Default False
        collate_fn(callable): function to generate mini-batch data by merging
            the sample list, None for only stack each fields of sample in axis
            0(same as :attr::`np.stack(..., axis=0)`). Default None
        num_workers(int): the number of subprocess to load data, 0 for no
            subprocess used and loading data in main process. Default 0
        use_buffer_reader (bool): whether to use bufferred reader. 
            If use_buffer_reader=True, the DataLoader would prefetch next 
            batch data asynchronously, so it would speed up data feeding 
            and occupies a little more CPU or GPU memory, i.e., the memory
            of one batch input data. Default True.
        use_shared_memory (bool): whether to use shared memory to speed up
            putting data into inter-process queue, set :attr:`use_shared_memory`
            as True only when the shared memory space on your machine(e.g.
            space of '/dev/shm' on Linux operating sysytem) is large enough.
            Shared memory will only be enabled in multi-process mode(num_workers
            > 0). Default True.
        timeout(int): the timeout value for getting data form output queue
            of subprocesses. Default 0.
        worker_init_fn(callable): init function which will be called with
            worker id on each subproces starting if not set as None. Default
            None.

    Returns:
        DataLoader: an iterable object for data iterating

    Examples:
        
        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid
            from paddle.io import Dataset, BatchSampler, DataLoader

            BATCH_NUM = 20
            BATCH_SIZE = 16
            EPOCH_NUM = 4

            IMAGE_SIZE = 784
            CLASS_NUM = 10

            USE_GPU = False # whether use GPU to run model

            # define a random dataset
            class RandomDataset(Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples

                def __getitem__(self, idx):
                    image = np.random.random([IMAGE_SIZE]).astype('float32')
                    label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
                    return image, label

                def __len__(self):
                    return self.num_samples

            # get places
            places = fluid.cuda_places() if USE_GPU else fluid.cpu_places()

            # -------------------- static graph ---------------------

            def simple_net(image, label):
                fc_tmp = fluid.layers.fc(image, size=CLASS_NUM, act='softmax')
                cross_entropy = fluid.layers.softmax_with_cross_entropy(image, label)
                loss = fluid.layers.reduce_mean(cross_entropy)
                sgd = fluid.optimizer.SGD(learning_rate=1e-3)
                sgd.minimize(loss)
                return loss

            image = fluid.data(name='image', shape=[None, IMAGE_SIZE], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')

            loss = simple_net(image, label)

            exe = fluid.Executor(places[0])
            exe.run(fluid.default_startup_program())

            prog = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(loss_name=loss.name)

            dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)

            loader = DataLoader(dataset,
                                feed_list=[image, label],
                                places=places,
                                batch_size=BATCH_SIZE, 
                                shuffle=True,
                                drop_last=True,
                                num_workers=2)

            for e in range(EPOCH_NUM):
                for i, data in enumerate(loader()):
                    l = exe.run(prog, feed=data, fetch_list=[loss], return_numpy=True)
                    print("Epoch {} batch {}: loss = {}".format(e, i, l[0][0]))

            # -------------------------------------------------------
                
            # --------------------- dygraph mode --------------------

            class SimpleNet(fluid.dygraph.Layer):
                def __init__(self):
                    super(SimpleNet, self).__init__()
                    self.fc = fluid.dygraph.nn.Linear(IMAGE_SIZE, CLASS_NUM, act='softmax')

                def forward(self, image, label=None):
                    return self.fc(image)

            with fluid.dygraph.guard(places[0]):
                simple_net = SimpleNet()
                opt = fluid.optimizer.SGD(learning_rate=1e-3,
                                          parameter_list=simple_net.parameters())

                loader = DataLoader(dataset,
                                    places=places[0],
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=2)

                for e in range(EPOCH_NUM):
                    for i, (image, label) in enumerate(loader()):
                        out = simple_net(image)
                        loss = fluid.layers.cross_entropy(out, label)
                        avg_loss = fluid.layers.reduce_mean(loss)
                        avg_loss.backward()
                        opt.minimize(avg_loss)
                        simple_net.clear_gradients()
                        print("Epoch {} batch {}: loss = {}".format(e, i, np.mean(loss.numpy())))

            # -------------------------------------------------------

    """

    def __init__(self,
                 dataset,
                 feed_list=None,
                 places=None,
                 return_list=False,
                 batch_sampler=None,
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 collate_fn=None,
                 num_workers=0,
                 use_buffer_reader=True,
                 use_shared_memory=True,
                 timeout=0,
                 worker_init_fn=None):
        self.return_list = return_list
        self.collate_fn = collate_fn
        self.use_buffer_reader = use_buffer_reader
        self.worker_init_fn = worker_init_fn

        assert isinstance(dataset, Dataset), \
            "dataset should be subclass instance of paddle.io.Dataset"
        self.dataset = dataset

        if not return_list and not in_dygraph_mode():
            assert feed_list is not None, \
                    "feed_list should be set when return_list=False"
        self.feed_list = feed_list

        assert places is not None, "places cannot be None"
        self.places = _convert_places(places)
        if in_dygraph_mode():
            assert len(self.places) == 1, \
                    "Number of places must be 1 in dygraph mode"

        assert num_workers >= 0, "num_workers should be a non-negative value"
        if num_workers > 0 and (sys.platform == 'darwin' or
                                sys.platform == 'win32'):
            logging.warning(
                "multi-process mode not support MacOs and Windows currently." \
                " use signle-process with num_workers = 0 instead")
            num_workers = 0
        self.num_workers = num_workers

        self.use_shared_memory = use_shared_memory
        if use_shared_memory and num_workers == 0:
            self.use_shared_memory = False

        assert timeout >= 0, "timeout should be a non-negative value"
        self.timeout = timeout

        if batch_sampler is not None:
            assert isinstance(batch_sampler, BatchSampler), \
                "batch_sampler should be None or subclass instance " \
                "of paddle.io.BatchSampler"
            assert batch_size == 1 and not shuffle and not drop_last, \
                "batch_size/shuffle/drop_last should not be set when " \
                "batch_sampler is given"
            self.batch_sampler = batch_sampler
        else:
            assert batch_size is not None and batch_size > 0, \
                "batch_size should be a positive value when " \
                "batch_sampler is not given"
            self.batch_sampler = BatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last)

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        if self.num_workers == 0:
            return _DataLoaderIterSingleProcess(self)
        else:
            return _DataLoaderIterMultiProcess(self)

    def __call__(self):
        return self.__iter__()

    @staticmethod
    def from_generator(feed_list=None,
                       capacity=None,
                       use_double_buffer=True,
                       iterable=True,
                       return_list=False,
                       use_multiprocess=False,
                       drop_last=True):
        """
        .. note::
          **The framework ensures that the data loading order of DataLoader is exactly the same as the user-defined data source.**

        Create a DataLoader object for loading data from Python generator. 
        Data would be prefetched using Python thread and be pushed
        into a queue asynchronously.

        The created DataLoader object provides 3 methods to set the data source
        :code:`set_sample_generator` , :code:`set_sample_list_generator` and 
        :code:`set_batch_generator` . Please see the following example codes
        to know their usages.
        
        If iterable = True, the created DataLoader object is a Python generator
        object, which is iterable using for-range loop.

        If iterable = False, the created DataLoader object provides 
        :code:`start()` and :code:`reset()` method to control the data reading
        process. This mode is designed to be compatible with the 
        :code:`fluid.layers.py_reader` interface. Users can migrate the codes   
        from :code:`fluid.layers.py_reader` to :code:`fluid.io.DataLoader` 
        easily when using iterable=False. 

        Args:  
            feed_list (list(Variable)|tuple(Variable)): feed variable list.
                The variables should be created by :code:`fluid.data()`.
            capacity (int): capacity of the queue maintained in DataLoader.
                The unit is batch number. Set larger capacity if your reader 
                is fast. 
            use_double_buffer (bool): whether to use double_buffer_reader. 
                If use_double_buffer=True, the DataLoader would prefetch next 
                batch data asynchronously, so it would speed up data feeding 
                and occupies a little more CPU or GPU memory, i.e., the memory
                of one batch input data. 
            iterable (bool): whether the created DataLoader is iterable. 
            return_list (bool): whether the return value on each device is 
                presented as a list. It is only valid when iterable=True. 
                If return_list=False, the return value on each device would 
                be a dict of str -> LoDTensor, where the key of the dict is 
                the name of each fed variables. If return_list=True, the 
                return value on each device would be a list(LoDTensor). It is
                recommended to use return_list=False in static graph mode and
                use return_list=True in dygraph mode.  
            use_multiprocess (bool): whether to use multi-process to speed up
                the data loading process in dygraph. Note: this parameter only
                can be used in the dygraph mode. In the static graph mode,
                whether this parameter is set or not has no effect.
                The Default value is False.
            drop_last (bool): whether to drop the last batches whose number is
                less than the CPU core/GPU card number. The default value is 
                True. In training phase, users should not set drop_last=False,
                because all CPU cores/GPU cards must read data from DataLoader. 
                In inference phase, users can set drop_last=False, so that the
                last batches whose number is less than the CPU core/GPU card
                number can be tested. 

        Returns:
            loader (DataLoader): the created DataLoader object.

        Examples 1:
            
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                BATCH_NUM = 10 
                BATCH_SIZE = 16
                EPOCH_NUM = 4

                CLASS_NUM = 10

                ITERABLE = True # whether the created DataLoader object is iterable
                USE_GPU = False # whether to use GPU

                DATA_FORMAT = 'batch_generator' # data format of data source user provides 

                def simple_net(image, label):
                    fc_tmp = fluid.layers.fc(image, size=CLASS_NUM)
                    cross_entropy = fluid.layers.softmax_with_cross_entropy(image, label)
                    loss = fluid.layers.reduce_mean(cross_entropy)
                    sgd = fluid.optimizer.SGD(learning_rate=1e-3)
                    sgd.minimize(loss)
                    return loss

                def get_random_images_and_labels(image_shape, label_shape):
                    image = np.random.random(size=image_shape).astype('float32')
                    label = np.random.random(size=label_shape).astype('int64')
                    return image, label

                # If the data generator yields one sample each time,
                # use DataLoader.set_sample_generator to set the data source.
                def sample_generator_creator(): 
                    def __reader__():
                        for _ in range(BATCH_NUM * BATCH_SIZE):
                            image, label = get_random_images_and_labels([784], [1])
                            yield image, label

                    return __reader__

                # If the data generator yield list of samples each time,
                # use DataLoader.set_sample_list_generator to set the data source.
                def sample_list_generator_creator():
                    def __reader__():
                        for _ in range(BATCH_NUM): 
                            sample_list = []
                            for _ in range(BATCH_SIZE):
                                image, label = get_random_images_and_labels([784], [1])
                                sample_list.append([image, label])

                            yield sample_list

                    return __reader__ 

                # If the data generator yields a batch each time, 
                # use DataLoader.set_batch_generator to set the data source.
                def batch_generator_creator():
                    def __reader__():
                        for _ in range(BATCH_NUM):
                            batch_image, batch_label = get_random_images_and_labels([BATCH_SIZE, 784], [BATCH_SIZE, 1]) 
                            yield batch_image, batch_label

                    return __reader__

                # If DataLoader is iterable, use for loop to train the network 
                def train_iterable(exe, prog, loss, loader):
                    for _ in range(EPOCH_NUM):
                        for data in loader():
                            exe.run(prog, feed=data, fetch_list=[loss])

                # If DataLoader is not iterable, use start() and reset() method to control the process 
                def train_non_iterable(exe, prog, loss, loader):
                    for _ in range(EPOCH_NUM):
                        loader.start() # call DataLoader.start() before each epoch starts
                        try:
                            while True:
                                exe.run(prog, fetch_list=[loss])
                        except fluid.core.EOFException:
                            loader.reset() # call DataLoader.reset() after catching EOFException 

                def set_data_source(loader, places):
                    if DATA_FORMAT == 'sample_generator':
                        loader.set_sample_generator(sample_generator_creator(), batch_size=BATCH_SIZE, drop_last=True, places=places)
                    elif DATA_FORMAT == 'sample_list_generator':
                        loader.set_sample_list_generator(sample_list_generator_creator(), places=places)
                    elif DATA_FORMAT == 'batch_generator':
                        loader.set_batch_generator(batch_generator_creator(), places=places)
                    else:
                        raise ValueError('Unsupported data format')

                image = fluid.data(name='image', shape=[None, 784], dtype='float32')
                label = fluid.data(name='label', shape=[None, 1], dtype='int64')

                # Define DataLoader 
                loader = fluid.io.DataLoader.from_generator(feed_list=[image, label], capacity=16, iterable=ITERABLE)

                # Define network
                loss = simple_net(image, label)

                # Set data source of DataLoader
                #
                # If DataLoader is iterable, places must be given and the number of places must be the same with device number.  
                #  - If you are using GPU, call `fluid.cuda_places()` to get all GPU places. 
                #  - If you are using CPU, call `fluid.cpu_places()` to get all CPU places. 
                # 
                # If DataLoader is not iterable, places can be None.
                places = fluid.cuda_places() if USE_GPU else fluid.cpu_places()
                set_data_source(loader, places)

                exe = fluid.Executor(places[0])
                exe.run(fluid.default_startup_program())

                prog = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(loss_name=loss.name)

                if loader.iterable:
                    train_iterable(exe, prog, loss, loader)
                else:
                    train_non_iterable(exe, prog, loss, loader)


                '''
                Users can use return_list = True in dygraph mode. 
                '''
                with fluid.dygraph.guard(places[0]):
                    loader = fluid.io.DataLoader.from_generator(capacity=2, return_list=True)
                    set_data_source(loader, places[0]) 
                    for image, label in loader():
                        relu = fluid.layers.relu(image)
                        assert image.shape == [BATCH_SIZE, 784] 
                        assert label.shape == [BATCH_SIZE, 1]
                        assert relu.shape == [BATCH_SIZE, 784]

        Examples 2:

            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np
                import os

                # We use 2 CPU cores to run inference network 
                os.environ['CPU_NUM'] = '2'

                # The data source has only 3 batches, which can not be
                # divided evenly to each CPU core
                def batch_generator():  
                    for i in range(3):
                        yield np.array([i+1]).astype('float32'), 

                x = fluid.data(name='x', shape=[None], dtype='float32')  
                y = x * x

                def run_inference(drop_last): 
                    loader = fluid.io.DataLoader.from_generator(feed_list=[x],
                            capacity=8, drop_last=drop_last)
                    loader.set_batch_generator(batch_generator, fluid.cpu_places())

                    exe = fluid.Executor(fluid.CPUPlace())
                    prog = fluid.CompiledProgram(fluid.default_main_program())
                    prog = prog.with_data_parallel()

                    result = []
                    for data in loader():
                        each_ret, = exe.run(prog, feed=data, fetch_list=[y])
                        result.extend(each_ret)
                    return result

                # Set drop_last to True, so that the last batch whose
                # number is less than CPU core number would be discarded.
                print(run_inference(drop_last=True)) # [1.0, 4.0]

                # Set drop_last to False, so that the last batch whose
                # number is less than CPU core number can be tested.
                print(run_inference(drop_last=False)) # [1.0, 4.0, 9.0]
        """
        if in_dygraph_mode():
            return DygraphGeneratorLoader(feed_list, capacity,
                                          use_double_buffer, iterable,
                                          return_list, use_multiprocess)
        else:
            return GeneratorLoader(feed_list, capacity, use_double_buffer,
                                   iterable, return_list, drop_last)

    @staticmethod
    def from_dataset(dataset, places, drop_last=True):
        """
        Create an iterable DataLoader object for loading data from Dataset.    
        Dataset is only supported in Linux system currently.

        Args:
            dataset (InMemoryDataset|QueueDataset): the dataset object.
            places (list(CUDAPlace)|list(CPUPlace)): places where the result 
                data should be converted.   
            drop_last (bool): whether to drop the last batch whose sample 
                number is less than batch size. If drop_last = True, they
                would be dropped. If drop_last = False, they would be kept. 

        Returns:
            loader (DataLoader): the created DataLoader object, which can be 
                treated as a Python generator.   

        Examples:

            .. code-block:: python

                import paddle.fluid as fluid

                image = fluid.data(name='image', shape=[None, 784], dtype='float32')
                label = fluid.data(name='label', shape=[None, 1], dtype='int64')

                dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
                dataset.set_batch_size(32)
                dataset.set_filelist(['a.txt', 'b.txt', 'c.txt'])
                dataset.set_use_var([image, label])
                dataset.set_pipe_command('cat') 

                loader = fluid.io.DataLoader.from_dataset(dataset, fluid.cpu_places())
        """
        return DatasetLoader(dataset, places, drop_last)


class DygraphGeneratorLoader(DataLoaderBase):
    """
    The GeneratorLoader of dygraph

    The multiprocess dygraph GeneratorLoader's most functions are different from 
    static graph GeneratorLoader, Separate implementation to keep code readable.
    """

    def __init__(self,
                 feed_list=None,
                 capacity=None,
                 use_double_buffer=True,
                 iterable=True,
                 return_list=True,
                 use_multiprocess=False):
        self._batch_reader = None
        self._places = None
        self._feed_list = feed_list

        if not capacity:
            raise ValueError("Please give value to capacity.")
        self._capacity = capacity
        self._use_double_buffer = use_double_buffer

        if not iterable:
            logging.warning(
                "Please NOTE: imperative mode can support iterable mode only. Change to iterable mode."
            )
        self._iterable = True
        if not return_list:
            logging.warning(
                "Please NOTE: imperative mode can support return as list only. Change to return as list."
            )
        self._return_list = True

        # NOTE: the multiprocessing in different platform is incompatible, we will solve it later
        self._use_multiprocess = use_multiprocess
        if self._use_multiprocess and (sys.platform == 'darwin' or
                                       sys.platform == 'win32'):
            logging.warning(
                "NOTE: The multiprocess mode does not currently support MacOs and Windows."
            )
            self._use_multiprocess = False

        if self._use_multiprocess:
            # NOTE: the multiprocessing.Queue used to save loading data in self._process
            self._data_queue = None
            # NOTE: this process is used to load data asynchronously from self._batch_reader
            self._process = None

        # NOTE: the C++ LoDTensorBlockingQueue instance
        self._blocking_queue = None
        # NOTE: 1. In multiprocess mode, this thread is used to get next batch data from
        # self._data_queue, then push it into self._blocking_queue; 2. In singleprocess
        # mode, this thread is used to get next batch data from self._batch_reader, then 
        # push it into self._blocking_queue
        self._thread = None

    @property
    def queue(self):
        return self._blocking_queue

    @property
    def iterable(self):
        return self._iterable

    def _clear_and_remove_data_queue(self):
        if self._data_queue is not None:
            while True:
                try:
                    self._data_queue.get_nowait()
                except queue.Empty:
                    break
            global multiprocess_queue_set
            multiprocess_queue_set.remove(self._data_queue)

    def _wait_thread_ends(self):
        thread = self._thread
        if thread is not None:
            self._blocking_queue.close()
            thread.join()

    def _wait_process_ends(self):
        process = self._process
        if process is not None:
            process.join()
            # erase process id
            core._erase_process_pids(id(self))

    def _init_iterable(self):
        self._wait_thread_ends()
        if self._use_multiprocess:
            self._wait_process_ends()
        self._var_names = []
        self._shapes = []
        self._dtypes = []
        self._need_check_feed = []
        self._blocking_queue = core.init_lod_tensor_blocking_queue(
            core.Variable(), self._capacity, False)
        self._reader = None
        self._reader = core.create_py_reader(
            self.queue, self._var_names, self._shapes, self._dtypes,
            self._need_check_feed, self._places, self._use_double_buffer, True)

    def _start(self):
        if self._use_multiprocess:
            # clear old _data_queue and remove it from multiprocess_queue_set
            self._clear_and_remove_data_queue()
            # set data_queue and process
            self._data_queue = multiprocessing.Queue(self._capacity)
            # add _data_queue into global queue set
            global multiprocess_queue_set
            multiprocess_queue_set.add(self._data_queue)
            self._process = multiprocessing.Process(
                target=self._reader_process_loop)
            self._process.daemon = True
            self._process.start()

            # Set child process signal handler
            # NOTE: [ avoiding hang ] 1. if the child process dies due to bus error/segfault
            # or just hang, the main process will hang waiting for data, so here need to deal 
            # with SIGSEGV and SIGBUS of child process; 2. if the main process end before child
            # process, it shuts the all its daemonic children down with a SIGTERM (instead of 
            # joining them without a timeout), so here nedd to deal with SIGTERM.
            core._set_process_pids(id(self), [self._process.pid])
            _set_SIGCHLD_handler()

            # Set reader_thread
            self._thread_done_event = threading.Event()
            self._thread = threading.Thread(
                target=self._reader_thread_loop_for_multiprocess)
            self._thread.daemon = True
            self._thread.start()
        else:
            self._thread = threading.Thread(
                target=self._reader_thread_loop_for_singleprocess)
            self._thread.daemon = True
            self._thread.start()

    def _reset(self):
        self._reader.reset()
        self._wait_thread_ends()
        if self._use_multiprocess:
            self._wait_process_ends()

    def __iter__(self):
        assert self.iterable, "DataLoader is not iterable"
        assert self._batch_reader is not None, \
            "Data source of DataLoader has not set yet"

        self._init_iterable()
        self._start()
        return self

    def __next__(self):
        try:
            return self._reader.read_next_var_list()
        except StopIteration:
            self._reset()
            six.reraise(*sys.exc_info())

    def _exit_thread_expectedly(self):
        self._thread_done_event.set()
        self._blocking_queue.close()

    def _exit_thread_unexpectedly(self):
        self._thread_done_event.set()
        self._blocking_queue.kill()
        logging.error("DataLoader reader thread raised an exception!")

    def _reader_process_loop(self):
        try:
            # set signal handler
            core._set_process_signal_handler()

            # NOTE: [ mmap files clear ] When the child process exits unexpectedly,
            # some shared memory objects may have been applied for but have not yet
            # been put into the inter-process Queue. This part of the object needs
            # to be cleaned up when the process ends.
            CleanupFuncRegistrar.register(_cleanup_mmap)

            for batch in self._batch_reader():
                tensor_list = core._convert_to_tensor_list(batch)
                self._data_queue.put(tensor_list)
                core._remove_tensor_list_mmap_fds(tensor_list)
            self._data_queue.put(None)
        except KeyboardInterrupt:
            # NOTE: Main process will raise KeyboardInterrupt anyways, ignore it in child process
            pass
        except:
            six.reraise(*sys.exc_info())

    def _reader_thread_loop_for_multiprocess(self):
        while not self._thread_done_event.is_set():
            try:
                # NOTE: [ avoid hanging ] Even with carefully designed data dependencies 
                # (i.e., a put() always corresponding to a get()), hanging on get() can 
                # still happen when data in queue is corrupted (e.g., due to 
                # Queue.cancel_join_thread or unexpected exit). So we set a timeout whenever 
                # we try to get data from `data_queue`
                # NOTE: [ avoid failed quickly ] Here, the time setting of QUEUE_GET_TIMEOUT
                # is relatively long, currently it is 60 seconds, because in some models,
                # if the reader child process starts with a heavy burden, the child process
                # has no enough time to put the data in the queue when the main process
                # start trying to get data from queue. At this time, the child thread needs
                # to wait slightly longer
                tensor_list = self._data_queue.get(timeout=QUEUE_GET_TIMEOUT)
            except:
                # NOTE [ avoid handing ] After adding the shared memory mechanism, not only
                # the queue.Empty exception will occur here, but other exceptions will also
                # occur, such as mmap failure. If it is not handled here, it will hang.
                self._exit_thread_unexpectedly()
                logging.error(
                    "DataLoader reader thread failed to read data from the multiprocessing.Queue."
                )
                six.reraise(*sys.exc_info())

            if not self._thread_done_event.is_set():
                if tensor_list is not None:
                    try:
                        array = core.LoDTensorArray()
                        for tensor in tensor_list:
                            array.append(tensor)
                        if not self._blocking_queue.push(array):
                            self._blocking_queue.close()
                    except:
                        self._exit_thread_unexpectedly()
                        six.reraise(*sys.exc_info())
                else:
                    self._exit_thread_expectedly()

    def _reader_thread_loop_for_singleprocess(self):
        try:
            for sample in self._batch_reader():
                array = core.LoDTensorArray()
                for item in sample:
                    if not isinstance(item, core.LoDTensor):
                        item = self._check_input_array(item)
                        tmp = core.LoDTensor()
                        tmp.set(item, core.CPUPlace())
                        item = tmp

                    array.append(item)

                if not self._blocking_queue.push(array):
                    break

            self._blocking_queue.close()
            self._thread = None
        except Exception:
            self._blocking_queue.kill()
            self._thread = None
            logging.warning(
                "DygraphDataLoader reader thread raised an exception.")
            six.reraise(*sys.exc_info())

    def set_sample_generator(self,
                             reader,
                             batch_size,
                             drop_last=True,
                             places=None):
        assert batch_size > 0, "batch_size must be larger than 0"
        self.set_sample_list_generator(
            paddle.batch(
                reader, batch_size=batch_size, drop_last=drop_last),
            places=places)
        return self

    def set_sample_list_generator(self, reader, places=None):
        def __batch_reader_impl__():
            for batch in reader():
                slots = []
                for items in batch:
                    for i, item in enumerate(items):
                        if len(slots) < len(items):
                            slots.append([item])
                        else:
                            slots[i].append(item)
                yield slots

        self.set_batch_generator(__batch_reader_impl__, places)
        return self

    def set_batch_generator(self, reader, places=None):
        self._batch_reader = reader
        if places is None:
            places = _current_expected_place()
        self._places = _convert_places(places)
        assert len(self._places) == 1, \
            "Number of places must be 1 in imperative mode"
        return self


class GeneratorLoader(DataLoaderBase):
    def __init__(self,
                 feed_list=None,
                 capacity=None,
                 use_double_buffer=True,
                 iterable=True,
                 return_list=False,
                 drop_last=True):
        self._tensor_reader = None
        self._places = None
        self._thread = None
        self._queue = None
        self._feed_list = feed_list
        self._exited = False
        self._drop_last = drop_last
        self._keep_order = keep_data_loader_order()
        if not capacity:
            raise ValueError("Please give value to capacity.")
        self._iterable = iterable
        self._return_list = return_list
        if not self._feed_list:
            raise Exception("Feed list must be given under static mode.")
        self._use_double_buffer = use_double_buffer
        self._capacity = capacity
        if not self._iterable:
            self._init_non_iterable()

        self._auto_checkpoint_name = unique_name.generate(
            "__auto_checkpoint_dataloader__")
        self._ignore_epoch = False

    def _wait_thread_ends(self):
        # Get self._thread first to prevent data race, because __thread_main__
        # would set self._thread be None at the end
        thread = self._thread
        if thread is not None and self._iterable:
            self._queue.close()
            thread.join()

    def _init_iterable(self):
        self._wait_thread_ends()
        self._var_names = [v.name for v in self._feed_list]
        self._shapes = [v.shape for v in self._feed_list]
        self._dtypes = [v.dtype for v in self._feed_list]
        self._need_check_feed = [
            v.desc.need_check_feed() for v in self._feed_list
        ]
        self._queue = core.init_lod_tensor_blocking_queue(
            core.Variable(), self._capacity, self._keep_order)
        self._reader = None
        self._reader = core.create_py_reader(
            self.queue, self._var_names, self._shapes, self._dtypes,
            self._need_check_feed, self._places, self._use_double_buffer,
            self._drop_last)

    def _init_non_iterable(self):
        lod_levels = []
        dtypes = []
        shape_concat = []
        ranks = []
        shapes = []
        need_check_feed = []

        for feed_data in self._feed_list:
            dtypes.append(feed_data.dtype)
            shape_concat.extend(feed_data.shape)
            ranks.append(len(feed_data.shape))
            shapes.append(feed_data.shape)
            lod_levels.append(feed_data.lod_level)
            need_check_feed.append(int(feed_data.desc.need_check_feed()))

        queue_name = data_loader_unique_name_generator(
            'lod_tensor_blocking_queue')
        reader_name = data_loader_unique_name_generator('create_py_reader')
        double_buffer_name = data_loader_unique_name_generator('double_buffer')

        var = global_scope().var(queue_name)
        self._queue = core.init_lod_tensor_blocking_queue(var, self._capacity,
                                                          self._keep_order)

        if self._keep_order:
            block = default_main_program().current_block()
        else:
            block = default_startup_program().current_block()

        reader_var = block.create_var(name=reader_name)

        dtype_int = [int(t) for t in dtypes]
        block.append_op(
            type='create_py_reader',
            inputs={'blocking_queue': [queue_name]},
            outputs={'Out': [reader_var]},
            attrs={
                'shape_concat': shape_concat,
                'lod_levels': lod_levels,
                'dtypes': dtype_int,
                'need_check_feed': need_check_feed,
                'ranks': ranks
            })

        reader_var.desc.set_dtypes(dtypes)
        reader_var.persistable = True
        reader_var.stop_gradient = True

        if self._keep_order:
            main_prog_var = reader_var
            reader = main_prog_var
            reader.reset = self._queue.reset
        else:
            main_prog_var = _copy_reader_var_(
                default_main_program().current_block(), reader_var)

            main_prog_var.stop_gradient = True
            main_prog_var.persistable = True

            reader = monkey_patch_reader_methods(main_prog_var)

        if self._use_double_buffer:
            double_buffer_reader = double_buffer(
                reader, name=double_buffer_name)
            # we return a double buffer reader. However, the reset method comes from
            # py_reader.
            double_buffer_reader.reset = reader.reset
            reader = double_buffer_reader

        self._reader = reader

        default_main_program().current_block().append_op(
            type='read',
            inputs={'Reader': [self._reader]},
            outputs={'Out': self._feed_list},
            attrs={'drop_last': self._drop_last})

    @property
    def queue(self):
        return self._queue

    @property
    def iterable(self):
        return self._iterable

    def __iter__(self):
        assert self.iterable, "DataLoader is not iterable"
        assert self._tensor_reader is not None, \
            "Data source of DataLoader has not set yet"

        self._ignore_epoch = dacp._ignore_epoch(self._auto_checkpoint_name)
        if self._ignore_epoch:
            return self

        self._init_iterable()
        self._start()
        return self

    def __next__(self):
        if self._ignore_epoch:
            raise StopIteration

        try:
            if self._return_list:
                return self._reader.read_next_list()
            else:
                return self._reader.read_next()
        except StopIteration:
            dacp._end(self._auto_checkpoint_name)
            self._ignore_epoch = False

            self._queue.close()
            self._reset()
            six.reraise(*sys.exc_info())

    def start(self):
        assert not self._iterable, "start() cannot be called when DataLoader is iterable"
        self._start()

    def reset(self):
        assert not self._iterable, "reset() cannot be called when DataLoader is iterable"
        self._reset()

    def _start(self):
        def __thread_main__():
            try:
                while not self._queue.wait_for_inited(1):
                    if self._exited:
                        return

                for tensors in self._tensor_reader():
                    array = core.LoDTensorArray()
                    for item in tensors:
                        if not isinstance(item, core.LoDTensor):
                            item = self._check_input_array(item)
                            tmp = core.LoDTensor()
                            tmp.set(item, core.CPUPlace())
                            item = tmp

                        array.append(item)

                    if not self._queue.push(array):
                        break

                self._queue.close()
                self._thread = None
            except Exception as ex:
                self._queue.kill()
                self._thread = None
                logging.warn('Your reader has raised an exception!')
                six.reraise(*sys.exc_info())

        self._thread = threading.Thread(target=__thread_main__)
        self._thread.daemon = True
        self._thread.start()

    def _reset(self):
        self._queue.close()
        self._exited = True
        thread = self._thread
        if thread is not None:
            thread.join()

        self._exited = False
        self._reader.reset()

    def set_sample_generator(self,
                             reader,
                             batch_size,
                             drop_last=True,
                             places=None):
        assert batch_size > 0, "batch_size must be larger than 0"
        has_lod = False
        for f in self._feed_list:
            if f.lod_level != 0:
                has_lod = True
                break

        if has_lod:
            self.set_sample_list_generator(
                paddle.batch(
                    reader, batch_size=batch_size, drop_last=drop_last),
                places=places)
        else:
            reader = BatchedTensorProvider(
                feed_list=self._feed_list,
                place=core.CPUPlace(),
                batch_size=batch_size,
                generator=reader,
                drop_last=drop_last)
            self.set_batch_generator(reader, places=places)
        return self

    def set_sample_list_generator(self, reader, places=None):
        with program_guard(Program(), Program()):
            feeder = DataFeeder(
                feed_list=self._feed_list, place=core.CPUPlace())
            paddle_reader = feeder.decorate_reader(reader, multi_devices=False)

        def __tensor_reader_impl__():
            for slots in paddle_reader():
                yield [slots[var.name] for var in self._feed_list]

        self.set_batch_generator(__tensor_reader_impl__, places)
        return self

    def set_batch_generator(self, reader, places=None):
        self._tensor_reader = reader
        if self._iterable:
            assert places is not None, "Places cannot be None when DataLoader is iterable"
            self._places = _convert_places(places)
        else:
            if places is not None:
                logging.info(
                    'places would be ommited when DataLoader is not iterable')
        return self


class PyReader(DataLoaderBase):
    """
    Create a reader object for data feeding in Python. 
    Data would be prefetched using Python thread and be pushed
    into a queue asynchronously. Data in the queue would be extracted 
    automatically when `Executor.run(...)` is called.

    Args:  
        feed_list (list(Variable)|tuple(Variable)): feed variable list.
            The variables should be created by :code:`fluid.layers.data()`.
        capacity (int): capacity of the queue maintained in PyReader.
            The unit is batch number. Set larger capacity if your reader 
            is fast. 
        use_double_buffer (bool): whether to use double_buffer_reader. 
            If use_double_buffer=True, PyReader would prefetch next 
            batch data asynchronously, so it would speed up data feeding 
            and occupies a little more CPU or GPU memory, i.e., the memory
            of one batch input data. 
        iterable (bool): whether the created PyReader is iterable. 
        return_list (bool): whether the return value on each device is 
            presented as a list. It is only valid when iterable=True. 
            If return_list=False, the return value on each device would 
            be a dict of str -> LoDTensor, where the key of the dict is 
            the name of each fed variables. If return_list=True, the 
            return value on each device would be a list(LoDTensor). It is
            recommended to use return_list=False in static graph mode and
            use return_list=True in dygraph mode. 

    Returns:
        the created reader object.

    Return type:
        reader(Reader)

    Examples:
        1. If iterable = False, the created PyReader object is almost the
           same as :code:`fluid.layers.py_reader()`. Operators would be 
           inserted into the program. User should call :code:`start()` 
           before each epoch and catch :code:`fluid.core.EOFException`
           thrown by :code:`Executor.run()` when epoch ends. Once the 
           exception is caught, user should call :code:`reset()` to reset 
           the reader manually.

        .. code-block:: python

           import paddle
           import paddle.fluid as fluid
           import numpy as np

           EPOCH_NUM = 3
           ITER_NUM = 5
           BATCH_SIZE = 3
           
           def network(image, label):
               # User-defined network, here is an example of softmax regression.
               predict = fluid.layers.fc(input=image, size=10, act='softmax')           
               return fluid.layers.cross_entropy(input=predict, label=label)

           def reader_creator_random_image_and_label(height, width):
               def reader():
                   for i in range(ITER_NUM):
                       fake_image = np.random.uniform(low=0,
                                                      high=255,
                                                      size=[height, width])
                       fake_label = np.ones([1])
                       yield fake_image, fake_label
               return reader

           image = fluid.data(name='image', shape=[None, 784, 784], dtype='float32')
           label = fluid.data(name='label', shape=[None, 1], dtype='int64')

           reader = fluid.io.PyReader(feed_list=[image, label],
                                      capacity=4,
                                      iterable=False)

           user_defined_reader = reader_creator_random_image_and_label(784, 784)
           reader.decorate_sample_list_generator(
               paddle.batch(user_defined_reader, batch_size=BATCH_SIZE))
           loss = network(image, label)
           executor = fluid.Executor(fluid.CPUPlace())
           executor.run(fluid.default_startup_program())
           for i in range(EPOCH_NUM):
               reader.start()
               while True:
                   try:
                       executor.run(feed=None)
                   except fluid.core.EOFException:
                       reader.reset()
                       break

 
        2. If iterable=True, the created PyReader object is decoupled with
           the program. No operator would be inserted into the program. 
           In this case, the created reader is a Python generator, which 
           is iterable. User should feed the data yielded from PyReader 
           object into :code:`Executor.run(feed=...)`.  

        .. code-block:: python

           import paddle
           import paddle.fluid as fluid
           import numpy as np

           EPOCH_NUM = 3
           ITER_NUM = 5
           BATCH_SIZE = 10

           def network(image, label):
               # User-defined network, here is an example of softmax regression.
               predict = fluid.layers.fc(input=image, size=10, act='softmax')           
               return fluid.layers.cross_entropy(input=predict, label=label)

           def reader_creator_random_image(height, width):
               def reader():
                   for i in range(ITER_NUM):
                       fake_image = np.random.uniform(low=0, high=255, size=[height, width])
                       fake_label = np.ones([1])
                       yield fake_image, fake_label 
               return reader

           image = fluid.data(name='image', shape=[None, 784, 784], dtype='float32')
           label = fluid.data(name='label', shape=[None, 1], dtype='int64')
           reader = fluid.io.PyReader(feed_list=[image, label], capacity=4, iterable=True, return_list=False)

           user_defined_reader = reader_creator_random_image(784, 784)
           reader.decorate_sample_list_generator(
               paddle.batch(user_defined_reader, batch_size=BATCH_SIZE),
                   fluid.core.CPUPlace())
           
           loss = network(image, label)
           executor = fluid.Executor(fluid.CPUPlace())
           executor.run(fluid.default_startup_program())
           
           for _ in range(EPOCH_NUM):
               for data in reader():
                   executor.run(feed=data, fetch_list=[loss])


        3. If return_list=True, the return values would be presented as list instead of dict. 
           This is usually used in dygraph mode.

        .. code-block:: python

           import paddle
           import paddle.fluid as fluid
           import numpy as np

           ITER_NUM = 5
           BATCH_SIZE = 10

           def reader_creator_random_image(height, width):
               def reader():
                   for i in range(ITER_NUM):
                       yield np.random.uniform(low=0, high=255, size=[height, width]), \
                           np.random.random_integers(low=0, high=9, size=[1])
               return reader

           place = fluid.CPUPlace()
           with fluid.dygraph.guard(place):
               py_reader = fluid.io.PyReader(capacity=2, return_list=True)
               user_defined_reader = reader_creator_random_image(784, 784)
               py_reader.decorate_sample_list_generator(
                   paddle.batch(user_defined_reader, batch_size=BATCH_SIZE),
                   place)
               for image, label in py_reader():
                   relu = fluid.layers.relu(image)
    """

    def __init__(self,
                 feed_list=None,
                 capacity=None,
                 use_double_buffer=True,
                 iterable=True,
                 return_list=False):
        self._loader = DataLoader.from_generator(
            feed_list, capacity, use_double_buffer, iterable, return_list)

    @property
    def queue(self):
        return self._loader.queue

    @property
    def iterable(self):
        return self._loader.iterable

    def __iter__(self):
        return self._loader.__iter__()

    def __next__(self):
        return self._loader.__next__()

    def start(self):
        '''
        Start the data feeding thread. 
        Can only call when the reader object is not iterable.  
        
	Example:
	    .. code-block:: python
    
                import paddle
                import paddle.fluid as fluid
                import numpy as np

                BATCH_SIZE = 10

                def generator():
                    for i in range(5):
                        yield np.random.uniform(low=0, high=255, size=[784, 784]),

                image = fluid.data(name='image', shape=[None, 784, 784], dtype='float32')
                reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=False)
                reader.decorate_sample_list_generator(
                    paddle.batch(generator, batch_size=BATCH_SIZE))

                executor = fluid.Executor(fluid.CPUPlace())
                executor.run(fluid.default_startup_program())
                for i in range(3):
                    reader.start()
                    while True:
                        try:
                            executor.run(feed=None)
                        except fluid.core.EOFException:
                            reader.reset()
                            break

	    '''
        self._loader.start()

    def reset(self):
        '''
        Reset the reader object when :code:`fluid.core.EOFException` raises. 
        Can only call when the reader object is not iterable.
        
        Example:
            .. code-block:: python

                import paddle
                import paddle.fluid as fluid
                import numpy as np

                BATCH_SIZE = 10

                def generator():
                    for i in range(5):
                        yield np.random.uniform(low=0, high=255, size=[784, 784]),

                image = fluid.data(name='image', shape=[None, 784, 784], dtype='float32')
                reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=False)
                reader.decorate_sample_list_generator(
                    paddle.batch(generator, batch_size=BATCH_SIZE))

                executor = fluid.Executor(fluid.CPUPlace())
                executor.run(fluid.default_startup_program())
                for i in range(3):
                    reader.start()
                    while True:
                        try:
                            executor.run(feed=None)
                        except fluid.core.EOFException:
                            reader.reset()
                            break        

        '''
        self._loader.reset()

    def decorate_sample_generator(self,
                                  sample_generator,
                                  batch_size,
                                  drop_last=True,
                                  places=None):
        '''
        Set the data source of the PyReader object.
        
        The provided :code:`sample_generator` should be a Python generator,
        which yields list(numpy.ndarray)-typed data of each sample.

        :code:`places` must be set when the PyReader object is iterable.

        If all inputs have no lods, this method is faster than 
        :code:`decorate_sample_list_generator(paddle.batch(sample_generator, ...))` .

        Args:
            sample_generator (generator): Python generator that yields
                list(numpy.ndarray)-typed sample data.
            batch_size (int): batch size. Must be larger than 0.
            drop_last (bool): Whether to drop the last batch when sample number
                is less than batch_size. 
            places (None|list(CUDAPlace)|list(CPUPlace)): place list. Must
                be provided when PyReader is iterable.

        Example:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                EPOCH_NUM = 3
                ITER_NUM = 15
                BATCH_SIZE = 3
        
                def network(image, label):
                    # User-defined network, here is an example of softmax regression.
                    predict = fluid.layers.fc(input=image, size=10, act='softmax')           
                    return fluid.layers.cross_entropy(input=predict, label=label)

                def random_image_and_label_generator(height, width):
                    def generator():
                        for i in range(ITER_NUM):
                            fake_image = np.random.uniform(low=0,
                                                           high=255,
                                                           size=[height, width])
                            fake_label = np.array([1])
                            yield fake_image, fake_label
                    return generator

                image = fluid.data(name='image', shape=[None, 784, 784], dtype='float32')
                label = fluid.data(name='label', shape=[None, 1], dtype='int64')
                reader = fluid.io.PyReader(feed_list=[image, label], capacity=4, iterable=True)

                user_defined_generator = random_image_and_label_generator(784, 784)
                reader.decorate_sample_generator(user_defined_generator,
                                                 batch_size=BATCH_SIZE,
                                                 places=[fluid.CPUPlace()])
                loss = network(image, label)
                executor = fluid.Executor(fluid.CPUPlace())
                executor.run(fluid.default_startup_program())

                for _ in range(EPOCH_NUM):
                    for data in reader():
                        executor.run(feed=data, fetch_list=[loss])
    
        '''
        self._loader.set_sample_generator(sample_generator, batch_size,
                                          drop_last, places)

    def decorate_sample_list_generator(self, reader, places=None):
        '''
        Set the data source of the PyReader object. 

        The provided :code:`reader` should be a Python generator,
        which yields list(numpy.ndarray) typed batched data. 
        
        :code:`places` must be set when the PyReader object is iterable.

        Args:
            reader (generator): Python generator that yields 
                list(numpy.ndarray)-typed batched data. 
            places (None|list(CUDAPlace)|list(CPUPlace)): place list. Must
                be provided when PyReader is iterable.
        
        Example:
            .. code-block:: python

                import paddle
                import paddle.fluid as fluid
                import numpy as np

                EPOCH_NUM = 3
                ITER_NUM = 15
                BATCH_SIZE = 3

                def network(image, label):
                    # User-defined network, here is an example of softmax regression.
                    predict = fluid.layers.fc(input=image, size=10, act='softmax')           
                    return fluid.layers.cross_entropy(input=predict, label=label)

                def random_image_and_label_generator(height, width):
                    def generator():
                        for i in range(ITER_NUM):
                            fake_image = np.random.uniform(low=0,
                                                           high=255,
                                                           size=[height, width])
                            fake_label = np.ones([1])
                            yield fake_image, fake_label
                    return generator

                image = fluid.data(name='image', shape=[None, 784, 784], dtype='float32')
                label = fluid.data(name='label', shape=[None, 1], dtype='int64')
                reader = fluid.io.PyReader(feed_list=[image, label], capacity=4, iterable=True)

                user_defined_generator = random_image_and_label_generator(784, 784)
                reader.decorate_sample_list_generator(
                    paddle.batch(user_defined_generator, batch_size=BATCH_SIZE),
                    fluid.core.CPUPlace())
                
                loss = network(image, label)
                executor = fluid.Executor(fluid.core.CPUPlace())
                executor.run(fluid.default_startup_program())

                for _ in range(EPOCH_NUM):
                    for data in reader():
                        executor.run(feed=data, fetch_list=[loss])
                 
        '''
        self._loader.set_sample_list_generator(reader, places)

    def decorate_batch_generator(self, reader, places=None):
        '''
        Set the data source of the PyReader object.

        The provided :code:`reader` should be a Python generator,
        which yields numpy.ndarray-typed or LoDTensor-typed batched data.

        :code:`places` must be set when the PyReader object is iterable.

        Args:
            reader (generator): Python generator that yields LoDTensor-typed
                batched data.
            places (None|list(CUDAPlace)|list(CPUPlace)): place list. Must
                be provided when PyReader is iterable.

        Example:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                EPOCH_NUM = 3
                ITER_NUM = 15
                BATCH_SIZE = 3
               
                def network(image, label):
                    # User-defined network, here is an example of softmax regression.
                    predict = fluid.layers.fc(input=image, size=10, act='softmax')           
                    return fluid.layers.cross_entropy(input=predict, label=label)

                def random_image_and_label_generator(height, width):
                    def generator():
                        for i in range(ITER_NUM):
                            batch_image = np.random.uniform(low=0,
                                                            high=255,
                                                            size=[BATCH_SIZE, height, width])
                            batch_label = np.ones([BATCH_SIZE, 1])
                            batch_image = batch_image.astype('float32')
                            batch_label = batch_label.astype('int64')
                            yield batch_image, batch_label
                    return generator

                image = fluid.data(name='image', shape=[None, 784, 784], dtype='float32')
                label = fluid.data(name='label', shape=[None, 1], dtype='int64')
                reader = fluid.io.PyReader(feed_list=[image, label], capacity=4, iterable=True)

                user_defined_generator = random_image_and_label_generator(784, 784)
                reader.decorate_batch_generator(user_defined_generator, fluid.CPUPlace())
                
                loss = network(image, label)
                executor = fluid.Executor(fluid.CPUPlace())
                executor.run(fluid.default_startup_program())

                for _ in range(EPOCH_NUM):
                    for data in reader():
                        executor.run(feed=data, fetch_list=[loss])

        '''
        self._loader.set_batch_generator(reader, places)


class DatasetLoader(DataLoaderBase):
    def __init__(self, dataset, places, drop_last):
        assert isinstance(dataset,
                          DatasetBase), "dataset must be type of DatasetBase"
        assert not in_dygraph_mode(
        ), "DatasetLoader is not supported in dygraph mode yet"

        thread_num = len(places)

        assert len(dataset.filelist) >= thread_num, \
            "Filelist number of dataset {} must be not less than place number {}".format(len(dataset.filelist), thread_num)

        if dataset.thread_num != 0 and dataset.thread_num != thread_num:
            logging.warn('thread_num {} which is set in Dataset is ignored'.
                         format(dataset.thread_num))

        dataset.set_thread(thread_num)

        if isinstance(dataset,
                      InMemoryDataset) and dataset.queue_num > thread_num:
            logging.warn("queue_num {} which is set in Dataset is ignored".
                         format(dataset.queue_num))
            dataset.set_queue_num(thread_num)

        self._dataset = dataset
        use_slots = [
            slot.name for slot in dataset.proto_desc.multi_slot_desc.slots
            if slot.is_used
        ]

        self._iterable_dataset = core.IterableDatasetWrapper(
            dataset.dataset, use_slots,
            _convert_places(places), dataset.proto_desc.batch_size, drop_last)

    def __iter__(self):
        self._dataset._finish_to_run()
        self._dataset._prepare_to_run()
        self._iterable_dataset._start()
        return self

    def __next__(self):
        return self._iterable_dataset._next()
