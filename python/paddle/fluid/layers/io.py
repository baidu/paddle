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
import contextlib
import multiprocessing
import threading

from ..data_feeder import DataFeeder
from .control_flow import BlockGuard
from .layer_function_generator import templatedoc
from .. import core
from ..executor import global_scope
from ..framework import convert_np_dtype_to_dtype_, default_main_program, \
    default_startup_program, program_guard, Program
from ..layer_helper import LayerHelper
from ..unique_name import generate as unique_name

__all__ = [
    'data', 'open_recordio_file', 'open_files', 'read_file', 'shuffle', 'batch',
    'double_buffer', 'random_data_generator', 'py_reader', 'Preprocessor',
    'load'
]


def data(name,
         shape,
         append_batch_size=True,
         dtype='float32',
         lod_level=0,
         type=core.VarDesc.VarType.LOD_TENSOR,
         stop_gradient=True):
    """
    **Data Layer**

    This function takes in the input and based on whether data has
    to be returned back as a minibatch, it creates the global variable by using
    the helper functions. The global variables can be accessed by all the
    following operators in the graph.

    All the input variables of this function are passed in as local variables
    to the LayerHelper constructor.

    Args:
       name(str): The name/alias of the function
       shape(list): Tuple declaring the shape.
       append_batch_size(bool): Whether or not to append the data as a batch.
       dtype(int|float): The type of data : float32, float_16, int etc
       type(VarType): The output type. By default it is LOD_TENSOR.
       lod_level(int): The LoD Level. 0 means the input data is not a sequence.
       stop_gradient(bool): A boolean that mentions whether gradient should flow.

    Returns:
        Variable: The global variable that gives access to the data.

    Examples:
        .. code-block:: python

          data = fluid.layers.data(name='x', shape=[784], dtype='float32')
    """
    helper = LayerHelper('data', **locals())
    shape = list(shape)
    for i in range(len(shape)):
        if shape[i] is None:
            shape[i] = -1
            append_batch_size = False
        elif shape[i] < 0:
            append_batch_size = False

    if append_batch_size:
        shape = [-1] + shape  # append batch size as -1

    data_var = helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        type=type,
        stop_gradient=stop_gradient,
        lod_level=lod_level,
        is_data=True)
    return data_var


class BlockGuardServ(BlockGuard):
    """
    BlockGuardServ class.

    BlockGuardServ class is used to create an op with a block in a program.
    """

    def __init__(self, server):
        if not (isinstance(server, ListenAndServ)):
            raise TypeError("BlockGuardServ takes a ListenAndServ")
        super(BlockGuardServ, self).__init__(server.helper.main_program)
        self.server = server

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False

        self.server.complete_op()
        return super(BlockGuardServ, self).__exit__(exc_type, exc_val, exc_tb)


class ListenAndServ(object):
    """
    **ListenAndServ Layer**

    ListenAndServ is used to create a rpc server bind and listen
    on specific TCP port, this server will run the sub-block when
    received variables from clients.

    Args:
        endpoint(string): IP:port string which the server will listen on.
        inputs(list): a list of variables that the server will get from clients.
        fan_in(int): how many client are expected to report to this server, default: 1.
        optimizer_mode(bool): whether to run the server as a parameter server, default: True.

    Examples:
        .. code-block:: python

            with fluid.program_guard(main):
                serv = layers.ListenAndServ(
                    "127.0.0.1:6170", ["X"], optimizer_mode=False)
                with serv.do():
                    x = layers.data(
                        shape=[32, 32],
                        dtype='float32',
                        name="X",
                        append_batch_size=False)
                    fluid.initializer.Constant(value=1.0)(x, main.global_block())
                    layers.scale(x=x, scale=10.0, out=out_var)

            exe = fluid.Executor(place)
            exe.run(main)
    """

    def __init__(self, endpoint, inputs, fan_in=1, optimizer_mode=True):
        self.helper = LayerHelper("listen_and_serv")
        self.inputs = inputs
        self.outputs = []
        self.endpoint = endpoint
        self.fan_in = fan_in
        # FIXME(typhoonzero): add optimizer_mode is stupid, should make it more
        # general.
        self.optimizer_mode = optimizer_mode

    def do(self):
        return BlockGuardServ(self)

    def get_params_and_grads(self):
        main_program = self.helper.main_program
        current_block = main_program.current_block()
        parent_block = self.parent_block()
        # params and grads in the same order.
        params = list()
        grads = list()
        for op in current_block.ops:
            # FIXME(typhoonzero): op.inputs is None if it's cloned.
            if self.optimizer_mode:
                if "Grad" in op.inputs and "Param" in op.inputs:
                    params.append(op.inputs["Param"].name)
                    grads.append(op.inputs["Grad"].name)
            else:
                # simple recv mode, recv operators inputs.
                for iname in op.input_names:
                    for in_var_name in op.input(iname):
                        params.append(parent_block.var(in_var_name))
                        grads.append(parent_block.var(in_var_name))

        return params, grads

    def parent_block(self):
        prog = self.helper.main_program
        parent_idx = prog.current_block().parent_idx
        assert parent_idx >= 0
        parent_block = prog.block(parent_idx)
        return parent_block

    def complete_op(self):
        main_program = self.helper.main_program
        current_block = main_program.current_block()
        parent_block = self.parent_block()

        parent_block.append_op(
            type='listen_and_serv',
            inputs={"X": self.inputs},
            outputs={},
            attrs={
                'endpoint': self.endpoint,
                'Fanin': self.fan_in,
                'optimize_blocks': [
                    current_block
                ],  # did not support multiple optimize blocks in layers
                'sync_mode': True,  # did not support async now in layers
                'grad_to_block_id': [""]
            })


def Send(endpoints, send_vars, sync=True):
    """
    Send variables to the server side, and get vars from server
    side when server have finished running server side program.

    Args:
        endpoints (str): comma seperated IP:PORT pairs in the order
                   of send_vars to send
        send_vars (list): variables to send to server
        sync (bool): whether to wait the request finish

    """
    assert (type(send_vars) == list)

    epmap = endpoints.split(",")
    endpoints = list(set(epmap))

    helper = LayerHelper("Send", **locals())
    rpc_op_role_name = core.op_proto_and_checker_maker.kOpRoleAttrName()

    helper.append_op(
        type="send",
        inputs={"X": send_vars},
        attrs={
            "endpoints": endpoints,
            "epmap": epmap,
            rpc_op_role_name: core.op_proto_and_checker_maker.OpRole.RPC
        })
    if sync:
        helper.append_op(type="send_barrier", attrs={"endpoints": endpoints})


def Recv(endpoints, get_vars, sync=True):
    """
    Receive variables from server side

    Args:
        endpoints (str): comma seperated IP:PORT pairs in the order
                   of send_vars to send
        get_vars (list): vars to get from server after send completes.
        sync (bool): whether to wait the request finish

    Returns:
        list: list of received variables
    """
    assert (type(get_vars) == list)

    epmap = endpoints.split(",")
    endpoints = list(set(epmap))

    helper = LayerHelper("Recv", **locals())
    helper.append_op(
        type="recv",
        inputs={"X": get_vars},
        outputs={"Out": get_vars},
        attrs={"endpoints": endpoints,
               "epmap": epmap})
    if sync:
        helper.append_op(type="fetch_barrier", attrs={"endpoints": endpoints})
    return get_vars


def monkey_patch_reader_methods(reader):
    def __get_reader__():
        scope = global_scope()
        var = scope.find_var(reader.name)
        return var.get_reader()

    def reset():
        return __get_reader__().reset()

    reader.reset = reset
    reader.stop_gradient = True
    reader.persistable = True
    return reader


def _copy_reader_var_(block, var):
    new_var = block.create_var(name=var.name, type=core.VarDesc.VarType.READER)
    new_var.desc.set_shapes(var.desc.shapes())
    new_var.desc.set_dtypes(var.desc.dtypes())
    new_var.persistable = True
    return new_var


def _copy_reader_create_op_(block, op):
    input_param_names = op.input_names
    new_input_map = {}
    for param_name in input_param_names:
        new_input_map[param_name] = []
        arg_names = op.input(param_name)
        for arg_name in arg_names:
            new_input_map[param_name].append(block.var(arg_name))

    output_param_names = op.output_names
    new_output_map = {}
    for param_name in output_param_names:
        new_output_map[param_name] = []
        arg_names = op.output(param_name)
        for arg_name in arg_names:
            new_output_map[param_name].append(block.var(arg_name))

    new_op = block.append_op(
        type=op.type,
        inputs=new_input_map,
        outputs=new_output_map,
        attrs=op.all_attrs())
    return new_op


@templatedoc(op_type='create_recordio_file_reader')
def open_recordio_file(filename,
                       shapes,
                       lod_levels,
                       dtypes,
                       pass_num=1,
                       for_parallel=True):
    """
    ${comment}

    Args:
       filename(${filename_type}): ${filename_comment}.
       shapes(list): List of tuples which declaring data shapes.
       lod_levels(${lod_levels_type}): ${lod_levels_comment}.
       dtypes(list): List of strs which declaring data type.
       pass_num(int): Number of passes to run.
       for_parallel(Bool): Set it as True if you are going to run
            subsequent operators in parallel.

    Returns:
       ${out_comment}.

    Examples:

        >>> import paddle.fluid as fluid
        >>> reader = fluid.layers.io.open_recordio_file(
        >>>                               filename='./data.recordio',
        >>>                               shapes=[(3,224,224), (1)],
        >>>                               lod_levels=[0, 0],
        >>>                               dtypes=['float32', 'int64'])
        >>> # Via the reader, we can use 'read_file' layer to get data:
        >>> image, label = fluid.layers.io.read_file(reader)
    """
    dtypes = [convert_np_dtype_to_dtype_(dt) for dt in dtypes]
    shape_concat = []
    ranks = []

    for shape in shapes:
        shape_concat.extend(shape)
        ranks.append(len(shape))

    var_name = unique_name('open_recordio_file')

    startup_blk = default_startup_program().current_block()
    startup_var = startup_blk.create_var(name=var_name)
    startup_blk.append_op(
        type='create_recordio_file_reader',
        outputs={'Out': [startup_var]},
        attrs={
            'shape_concat': shape_concat,
            'lod_levels': lod_levels,
            'filename': filename,
            'ranks': ranks
        })

    startup_var.desc.set_dtypes(dtypes)
    startup_var.persistable = True
    main_prog_var = _copy_reader_var_(default_main_program().current_block(),
                                      startup_var)

    if pass_num > 1:
        main_prog_var = multi_pass(reader=main_prog_var, pass_num=pass_num)

    return monkey_patch_reader_methods(main_prog_var)


def random_data_generator(low, high, shapes, lod_levels, for_parallel=True):
    """
    Create a uniform random data generator

    This layer returns a Reader Variable.
    Instead of opening a file and reading data from it, this
    Reader Variable generates float uniform random data by itself.
    It can be used as a dummy reader to test a network without
    opening a real file.

    Args:
       low(float): The lower bound of data's uniform distribution.
       high(float): The upper bound of data's uniform distribution.
       shapes(list): List of tuples which declaring data shapes.
       lod_levels(list): List of ints which declaring data lod_level.
       for_parallel(Bool): Set it as True if you are going to run
            subsequent operators in parallel.

    Returns:
       Variable: A Reader Variable from which we can get random data.

    Examples:

        .. code-block:: python

            reader = fluid.layers.random_data_generator(
                                             low=0.0,
                                             high=1.0,
                                             shapes=[[3,224,224], [1]],
                                             lod_levels=[0, 0])
            # Via the reader, we can use 'read_file' layer to get data:
            image, label = fluid.layers.read_file(reader)
    """
    dtypes = [core.VarDesc.VarType.FP32] * len(shapes)
    shape_concat = []
    ranks = []

    for shape in shapes:
        shape_concat.extend(shape)
        ranks.append(len(shape))

    var_name = unique_name('random_data_generator')

    startup_blk = default_startup_program().current_block()
    startup_var = startup_blk.create_var(name=var_name)
    startup_blk.append_op(
        type='create_random_data_generator',
        outputs={'Out': [startup_var]},
        attrs={
            'low': low,
            'high': high,
            'shape_concat': shape_concat,
            'lod_levels': lod_levels,
            'ranks': ranks
        })

    startup_var.desc.set_dtypes(dtypes)
    startup_var.persistable = True
    main_prog_var = _copy_reader_var_(default_main_program().current_block(),
                                      startup_var)

    if for_parallel:
        main_prog_var = parallel(reader=main_prog_var)

    return monkey_patch_reader_methods(main_prog_var)


def py_reader(capacity,
              shapes,
              dtypes,
              lod_levels=None,
              name=None,
              use_double_buffer=True):
    """
    Create a Python reader for data feeding in Python

    This layer returns a Reader Variable.
    The Reader provides :code:`decorate_paddle_reader()` and
    :code:`decorate_tensor_provider()` to set a Python generator as the data
    source in Python side. When :code:`Executor::Run()` is invoked in C++
    side, the data from the generator would be read automatically. Unlike
    :code:`DataFeeder.feed()`, the data reading process and
    :code:`Executor::Run()` process can run in parallel using
    :code:`py_reader`. The :code:`start()` method of the Reader should be
    called when each pass begins, while the :code:`reset()` method should be
    called when the pass ends and :code:`fluid.core.EOFException` raises.
    Note that :code:`Program.clone()` method cannot clone :code:`py_reader`.

    Args:
       capacity(int): The buffer capacity maintained by :code:`py_reader`.
       shapes(list|tuple): List of tuples which declaring data shapes.
       dtypes(list|tuple): List of strs which declaring data type.
       lod_levels(list|tuple): List of ints which declaring data lod_level.
       name(basestring): The prefix Python queue name and Reader name. None will
            be generated automatically.
       use_double_buffer(bool): Whether use double buffer or not.

    Returns:
       Variable: A Reader from which we can get feeding data.

    Examples:

        1. The basic usage of :code:`py_reader` is as follows:

        >>> import paddle.v2
        >>> import paddle.fluid as fluid
        >>> import paddle.dataset.mnist as mnist
        >>>
        >>> reader = fluid.layers.py_reader(capacity=64,
        >>>                                 shapes=[(-1,3,224,224), (-1,1)],
        >>>                                 dtypes=['float32', 'int64'])
        >>> reader.decorate_paddle_reader(
        >>>     paddle.v2.reader.shuffle(paddle.batch(mnist.train())
        >>>
        >>> img, label = fluid.layers.read_file(reader)
        >>> loss = network(img, label) # some network definition
        >>>
        >>> fluid.Executor(fluid.CUDAPlace(0)).run(fluid.default_startup_program())
        >>>
        >>> exe = fluid.ParallelExecutor(use_cuda=True, loss_name=loss.name)
        >>> for epoch_id in range(10):
        >>>     reader.start()
        >>>     try:
        >>>         while True:
        >>>             exe.run(fetch_list=[loss.name])
        >>>     except fluid.core.EOFException:
        >>>         reader.reset()

        2. When training and testing are both performed, two different
        :code:`py_reader` should be created with different names, e.g.:

        >>> import paddle.v2
        >>> import paddle.fluid as fluid
        >>> import paddle.dataset.mnist as mnist
        >>>
        >>> def network(reader):
        >>>     img, label = fluid.layers.read_file(reader)
        >>>     # Here, we omitted the network definition
        >>>     return loss
        >>>
        >>> train_reader = fluid.layers.py_reader(capacity=64,
        >>>                                       shapes=[(-1,3,224,224), (-1,1)],
        >>>                                       dtypes=['float32', 'int64'],
        >>>                                       name='train_reader')
        >>> train_reader.decorate_paddle_reader(
        >>>     paddle.v2.reader.shuffle(paddle.batch(mnist.train())
        >>>
        >>> test_reader = fluid.layers.py_reader(capacity=32,
        >>>                                      shapes=[(-1,3,224,224), (-1,1)],
        >>>                                      dtypes=['float32', 'int64'],
        >>>                                      name='test_reader')
        >>> test_reader.decorate_paddle_reader(paddle.batch(mnist.test(), 512))
        >>>
        >>> # Create train_main_prog and train_startup_prog
        >>> train_main_prog = fluid.Program()
        >>> train_startup_prog = fluid.Program()
        >>> with fluid.program_guard(train_main_prog, train_startup_prog):
        >>>     # Use fluid.unique_name.guard() to share parameters with test program
        >>>     with fluid.unique_name.guard():
        >>>         train_loss = network(train_reader) # some network definition
        >>>         adam = fluid.optimizer.Adam(learning_rate=0.01)
        >>>         adam.minimize(loss)
        >>>
        >>> # Create test_main_prog and test_startup_prog
        >>> test_main_prog = fluid.Program()
        >>> test_startup_prog = fluid.Program()
        >>> with fluid.program_guard(test_main_prog, test_startup_prog):
        >>>     # Use fluid.unique_name.guard() to share parameters with train program
        >>>     with fluid.unique_name.guard():
        >>>         test_loss = network(test_reader)
        >>>
        >>> fluid.Executor(fluid.CUDAPlace(0)).run(train_startup_prog)
        >>> fluid.Executor(fluid.CUDAPlace(0)).run(test_startup_prog)
        >>>
        >>> train_exe = fluid.ParallelExecutor(use_cuda=True,
        >>>                 loss_name=train_loss.name, main_program=train_main_prog)
        >>> test_exe = fluid.ParallelExecutor(use_cuda=True,
        >>>                 loss_name=test_loss.name, main_program=test_main_prog)
        >>> for epoch_id in range(10):
        >>>     train_reader.start()
        >>>     try:
        >>>         while True:
        >>>             train_exe.run(fetch_list=[train_loss.name])
        >>>     except fluid.core.EOFException:
        >>>         train_reader.reset()
        >>>
        >>>     test_reader.start()
        >>>     try:
        >>>         while True:
        >>>             test_exe.run(fetch_list=[test_loss.name])
        >>>     except fluid.core.EOFException:
        >>>         test_reader.reset()
    """
    dtypes = [convert_np_dtype_to_dtype_(dt) for dt in dtypes]
    shape_concat = []
    ranks = []

    for shape in shapes:
        shape_concat.extend(shape)
        ranks.append(len(shape))

    if lod_levels is None:
        lod_levels = [0] * len(shapes)

    if name is None:
        queue_name = unique_name('lod_tensor_blocking_queue')
        reader_name = unique_name('create_py_reader')
        double_buffer_name = unique_name('double_buffer')
    else:
        queue_name = "_".join([name, "queue"])
        reader_name = "_".join([name, "reader"])
        double_buffer_name = "_".join([name, "double_buffer"])

    var = global_scope().var(queue_name)
    feed_queue = core.init_lod_tensor_blocking_queue(var, capacity, shapes)

    startup_blk = default_startup_program().current_block()
    startup_var = startup_blk.create_var(name=reader_name)
    startup_blk.append_op(
        type='create_py_reader',
        inputs={'blocking_queue': [queue_name]},
        outputs={'Out': [startup_var]},
        attrs={
            'shape_concat': shape_concat,
            'lod_levels': lod_levels,
            'ranks': ranks
        })

    startup_var.desc.set_dtypes(dtypes)
    startup_var.persistable = True

    main_prog_var = _copy_reader_var_(default_main_program().current_block(),
                                      startup_var)

    reader = monkey_patch_reader_methods(main_prog_var)
    if use_double_buffer:
        double_buffer_reader = double_buffer(reader, name=double_buffer_name)
        # we return a double buffer reader. However, the reset method comes from
        # py_reader.
        double_buffer_reader.reset = reader.reset
        reader = double_buffer_reader

    # monkey patch py_reader special methods
    reader.queue = feed_queue
    current_reset_method = reader.reset
    reader.thread = None
    reader.tensor_provider = None
    reader.exited = False

    def start_provide_thread(func):
        def __provider_thread__():
            for tensors in func():
                array = core.LoDTensorArray()
                for item in tensors:
                    if not isinstance(item, core.LoDTensor):
                        tmp = core.LoDTensor()
                        tmp.set(item, core.CPUPlace())
                        item = tmp

                    array.append(item)

                if reader.exited:
                    break
                feed_queue.push(array)
                if reader.exited:
                    break
            feed_queue.close()

        reader.thread = threading.Thread(target=__provider_thread__)
        reader.thread.daemon = True
        reader.thread.start()

    def __set_tensor_provider__(func):
        reader.tensor_provider = func

    def __set_paddle_reader__(paddle_reader):
        with program_guard(Program(), Program()):
            feed_list = []
            counter = 0
            for dtype, shape, lod_level in zip(dtypes, shapes, lod_levels):
                name = str(counter)
                feed_list.append(
                    data(
                        name=name,
                        dtype=dtype,
                        shape=shape,
                        lod_level=lod_level))
                counter += 1

            feeder = DataFeeder(feed_list=feed_list, place=core.CPUPlace())
            paddle_reader = feeder.decorate_reader(
                paddle_reader, multi_devices=False)

        def __tensor_provider__():
            for slots in paddle_reader():
                yield [slots[str(idx)] for idx in xrange(counter)]

        __set_tensor_provider__(__tensor_provider__)

    def __reset__():
        current_reset_method()
        if reader.thread is not None and reader.tensor_provider is not None:
            reader.exited = True
            reader.thread.join()
            reader.exited = False

    def __start__():
        start_provide_thread(reader.tensor_provider)

    reader.reset = __reset__
    reader.decorate_tensor_provider = __set_tensor_provider__
    reader.decorate_paddle_reader = __set_paddle_reader__
    reader.start = __start__

    return reader


def open_files(filenames,
               shapes,
               lod_levels,
               dtypes,
               thread_num=None,
               buffer_size=None,
               pass_num=1,
               is_test=None):
    """
    Open files

    This layer takes a list of files to read from and returns a Reader Variable.
    Via the Reader Variable, we can get data from given files. All files must
    have name suffixs to indicate their formats, e.g., '*.recordio'.

    Args:
       filenames(list): The list of file names.
       shapes(list): List of tuples which declaring data shapes.
       lod_levels(list): List of ints which declaring data lod_level.
       dtypes(list): List of strs which declaring data type.
       thread_num(None): The number of thread to read files.
            Default: min(len(filenames), cpu_number).
       buffer_size(None): The buffer size of reader. Default: 3 * thread_num
       pass_num(int): Number of passes to run.
       is_test(bool|None): Whether `open_files` used for testing or not. If it
            is used for testing, the order of data generated is same as the file
            order. Otherwise, it is not guaranteed the order of data is same
            between every epoch. [Default: False].

    Returns:
       Variable: A Reader Variable via which we can get file data.

    Examples:
       .. code-block:: python

         reader = fluid.layers.io.open_files(filenames=['./data1.recordio',
                                                     './data2.recordio'],
                                             shapes=[(3,224,224), (1)],
                                             lod_levels=[0, 0],
                                             dtypes=['float32', 'int64'])

         # Via the reader, we can use 'read_file' layer to get data:
         image, label = fluid.layers.io.read_file(reader)
    """
    if thread_num is None:
        thread_num = min(len(filenames), multiprocessing.cpu_count())
    else:
        thread_num = int(thread_num)

    if buffer_size is None:
        buffer_size = 3 * thread_num
    else:
        buffer_size = int(buffer_size)

    if isinstance(filenames, basestring):
        filenames = [filenames]
    dtypes = [convert_np_dtype_to_dtype_(dt) for dt in dtypes]
    shape_concat = []
    ranks = []

    for shape in shapes:
        shape_concat.extend(shape)
        ranks.append(len(shape))

    multi_file_reader_name = unique_name('multi_file_reader')
    startup_blk = default_startup_program().current_block()
    startup_reader = startup_blk.create_var(name=multi_file_reader_name)
    attrs = {
        'shape_concat': shape_concat,
        'lod_levels': lod_levels,
        'ranks': ranks,
        'file_names': filenames,
        'thread_num': thread_num,
        'buffer_size': buffer_size
    }
    if is_test is not None:
        attrs['is_test'] = is_test
    startup_blk.append_op(
        type='open_files', outputs={'Out': [startup_reader]}, attrs=attrs)

    startup_reader.desc.set_dtypes(dtypes)
    startup_reader.persistable = True
    main_prog_reader = _copy_reader_var_(default_main_program().current_block(),
                                         startup_reader)
    if pass_num > 1:
        main_prog_reader = multi_pass(
            reader=main_prog_reader, pass_num=pass_num)

    return monkey_patch_reader_methods(main_prog_reader)


def __create_shared_decorated_reader__(op_type, reader, attrs):
    var_name = unique_name(op_type)
    startup_blk = default_startup_program().current_block()
    startup_var = startup_blk.create_var(name=var_name)
    startop_op = startup_blk.append_op(
        type=op_type,
        inputs={'UnderlyingReader': reader},
        outputs={'Out': [startup_var]},
        attrs=attrs)
    startup_var.persistable = True
    main_prog_block = default_main_program().current_block()
    main_prog_var = _copy_reader_var_(main_prog_block, startup_var)
    _copy_reader_create_op_(main_prog_block, startop_op)
    return monkey_patch_reader_methods(main_prog_var)


def __create_unshared_decorated_reader__(op_type, reader, attrs, name=None):
    new_reader_name = name if name is not None else unique_name(op_type)
    main_blk = default_main_program().current_block()
    new_reader = main_blk.create_var(name=new_reader_name)
    main_blk.append_op(
        type=op_type,
        inputs={'UnderlyingReader': reader},
        outputs={'Out': [new_reader]},
        attrs=attrs)
    return monkey_patch_reader_methods(new_reader)


def shuffle(reader, buffer_size):
    """
    Shuffle the reader.
    """
    return __create_unshared_decorated_reader__(
        'create_shuffle_reader', reader, {'buffer_size': int(buffer_size)})


def batch(reader, batch_size):
    """
    This layer is a reader decorator. It takes a reader and adds
    'batching' decoration on it. When reading with the result
    decorated reader, output data will be automatically organized
    to the form of batches.

    Args:
        reader(Variable): The reader to be decorated with 'batching'.
        batch_size(int): The batch size.

    Returns:
        Variable: The reader which has been decorated with 'batching'.

    Examples:
        .. code-block:: python

            raw_reader = fluid.layers.io.open_files(filenames=['./data1.recordio',
                                                           './data2.recordio'],
                                                    shapes=[(3,224,224), (1)],
                                                    lod_levels=[0, 0],
                                                    dtypes=['float32', 'int64'],
                                                    thread_num=2,
                                                    buffer_size=2)
            batch_reader = fluid.layers.batch(reader=raw_reader, batch_size=5)

            # If we read data with the raw_reader:
            #     data = fluid.layers.read_file(raw_reader)
            # We can only get data instance by instance.
            #
            # However, if we read data with the batch_reader:
            #     data = fluid.layers.read_file(batch_reader)
            # Each 5 adjacent instances will be automatically combined together
            # to become a batch. So what we get('data') is a batch data instead
            # of an instance.
    """
    return __create_unshared_decorated_reader__(
        'create_batch_reader', reader, {'batch_size': int(batch_size)})


def double_buffer(reader, place=None, name=None):
    """
    Wrap a double buffer reader. The data will copy to target place with a
    double buffer queue. If the target place is None, the place that executor
    perform on will be used.

    Args:
        reader(Variable): the reader variable need to be wrapped.
        place(Place): the place of target data. Default is the sample place of
            executor perform.

        name(str): Variable name. None if the user does not care.

    Returns:
        wrapped reader with double buffer.

    Examples:

        >>> reader = fluid.layers.open_files(filenames=['somefile'],
        >>>                                  shapes=[[-1, 784], [-1, 1]],
        >>>                                  dtypes=['float32', 'int64'])
        >>> reader = fluid.layers.double_buffer(reader)
        >>> img, label = fluid.layers.read_file(reader)
    """
    attrs = dict()
    if place is not None:
        attrs['place'] = str(place).upper()
    return __create_unshared_decorated_reader__(
        'create_double_buffer_reader', reader, attrs, name=name)


def multi_pass(reader, pass_num):
    return __create_shared_decorated_reader__(
        'create_multi_pass_reader', reader, {'pass_num': int(pass_num)})


def read_file(reader):
    """
    Execute the given reader and get data via it.

    A reader is also a Variable. It can be a raw reader generated by
    `fluid.layers.open_files()` or a decorated one generated by
    `fluid.layers.double_buffer()` and so on.

    Args:

        reader(Variable): The reader to execute.

    Returns:
        Tuple[Variable]: Data read via the given reader.

    Examples:
        .. code-block:: python

           data_file = fluid.layers.open_files(
                filenames=['mnist.recordio'],
                shapes=[(-1, 748), (-1, 1)],
                lod_levels=[0, 0],
                dtypes=["float32", "int64"])
            data_file = fluid.layers.double_buffer(
                fluid.layers.batch(data_file, batch_size=64))
            input, label = fluid.layers.read_file(data_file)
    """
    helper = LayerHelper('read_file')
    out = [
        helper.create_tmp_variable(
            stop_gradient=True, dtype='float32')
        for _ in range(len(reader.desc.shapes()))
    ]
    helper.append_op(
        type='read', inputs={'Reader': [reader]}, outputs={'Out': out})
    if len(out) == 1:
        return out[0]
    else:
        return out


class Preprocessor(object):
    """
    A block for data pre-processing in reader.

    Args:
        reader (Variable): A reader variable.
        name (str, default None): The name of the reader.

    Examples:
          .. code-block:: python

            preprocessor = fluid.layers.io.Preprocessor(reader=reader)
            with preprocessor.block():
                img, lbl = preprocessor.inputs()
                img_out = img / 2
                lbl_out = lbl + 1
                preprocessor.outputs(img_out, lbl_out)

            data_file = fluid.layers.io.double_buffer(preprocessor())

    """
    BEFORE_SUB_BLOCK = 0
    IN_SUB_BLOCK = 1
    AFTER_SUB_BLOCK = 2

    def __init__(self, reader, name=None):
        self.underlying_reader = reader
        new_reader_name = name if name is not None else unique_name(
            "create_custom_reader")
        self.main_prog = default_main_program()
        self.reader = self.main_prog.current_block().create_var(
            name=new_reader_name)
        self.sub_block = None
        self.source_var_names = None
        self.sink_var_names = None
        self.status = Preprocessor.BEFORE_SUB_BLOCK

    def _is_completed(self):
        return self.sub_block and self.source_var_names and self.sink_var_names

    @contextlib.contextmanager
    def block(self):
        self.status = Preprocessor.IN_SUB_BLOCK
        self.sub_block = self.main_prog.create_block()
        yield
        self.main_prog.rollback()
        self.status = Preprocessor.AFTER_SUB_BLOCK
        if not self._is_completed():
            raise RuntimeError(
                "The definition of preprocessor is incompleted! "
                "Please make sure that you have set input and output "
                "variables by invoking 'inputs' and 'outputs' in "
                "Preprocessor's sub-block.")

    def inputs(self):
        if self.status != Preprocessor.IN_SUB_BLOCK:
            raise RuntimeError(
                "Preprocessor.inputs() can only be invoked inside the sub-block."
            )

        source_shapes = self.underlying_reader.desc.shapes()
        source_dtypes = self.underlying_reader.desc.dtypes()
        source_lod_levels = self.underlying_reader.desc.lod_levels()
        self.source_var_names = [
            unique_name("preprocessor_source")
            for _ in range(len(source_shapes))
        ]
        source_vars = []
        for var_name, shape, dtype, lod_level in zip(
                self.source_var_names, source_shapes, source_dtypes,
                source_lod_levels):
            source_vars.append(self.main_prog.current_block().create_var(
                name=var_name, shape=shape, dtype=dtype, lod_level=lod_level))
        return source_vars

    def outputs(self, *outs):
        if self.status != Preprocessor.IN_SUB_BLOCK:
            raise RuntimeError(
                "Preprocessor.outputs() can only be invoked inside the sub-block."
            )
        self.sink_var_names = [var.name for var in outs]

    def __call__(self, *args, **kwargs):
        if self.status != Preprocessor.AFTER_SUB_BLOCK:
            raise RuntimeError(
                "Preprocessor output can only be retrieved after rnn block.")

        self.main_prog.current_block().append_op(
            type="create_custom_reader",
            inputs={'UnderlyingReader': self.underlying_reader},
            outputs={'Out': [self.reader]},
            attrs={
                "sub_block": self.sub_block,
                "source_var_names": self.source_var_names,
                "sink_var_names": self.sink_var_names
            })
        return monkey_patch_reader_methods(self.reader)


@templatedoc()
def load(out, file_path, load_as_fp16=None):
    """
    ${comment}

    >>> import paddle.fluid as fluid
    >>> tmp_tensor = fluid.layers.create_tensor(dtype='float32')
    >>> fluid.layers.load(tmp_tensor, "./tmp_tensor.bin")

    Args:
        out(${out_type}): ${out_comment}.

        file_path(${file_path_type}): ${file_path_comment}.

        load_as_fp16(${load_as_fp16_type}): ${load_as_fp16_comment}.

    Returns:
        None
    """
    helper = LayerHelper("load", **locals())
    attrs = {"file_path": file_path}
    if load_as_fp16 is not None:
        attrs['load_as_fp16'] = load_as_fp16
    helper.append_op(type="load", inputs={}, output={"Out": out}, args=attrs)
