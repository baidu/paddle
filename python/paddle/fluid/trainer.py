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
import os
import errno
import shutil
import time

from . import core
from . import data_feeder
from . import executor
from . import framework
from . import io
# optimizer is same as the parameter of Trainer.__init__. Rename it to opt_module
from . import optimizer as opt_module
from . import parallel_executor
from .transpiler import distribute_transpiler

__all__ = [
    'Trainer', 'BeginEpochEvent', 'EndEpochEvent', 'BeginStepEvent',
    'EndStepEvent', 'CheckpointConfig'
]


class BeginEpochEvent(object):
    """
    The begin of a training epoch.

    Args:
        epoch_id(int): The current epoch ID.
    """

    def __init__(self, epoch_id):
        self.epoch = epoch_id


class EndEpochEvent(object):
    """
    The end of a training epoch.

    Args:
        epoch_id(int): The current epoch ID.
    """

    def __init__(self, epoch_id):
        self.epoch = epoch_id


class BeginStepEvent(object):
    """
    The begin of a training epoch.

    Args:
        epoch_id(int): The current epoch ID.
        step_id(int): The current step ID.
    """

    def __init__(self, epoch_id, step_id):
        self.epoch = epoch_id
        self.step = step_id
        self.fetch_metrics = True
        """
        If fetch_metrics is true, the metrics will be fetched at the
        EndStepEvent. Default is True.
        """


class EndStepEvent(object):
    """
    The end of a training step.

    Args:
        epoch_id(int): The current epoch ID.
        step_id(int): The current step ID.
        metrics(list): A list of fetched tensor. The order of this list is same
            as the :code:`train_func` returns.
    """

    def __init__(self, epoch_id, step_id, metrics):
        self.epoch = epoch_id
        self.step = step_id
        self.metrics = metrics


class CheckpointConfig(object):
    """
    Parameter object for :code:`save_checkpoint` and
    :code:`fluid.Trainer`. Used to configuration how to save checkpoint.

    Args:
        checkpoint_dir(str): Directory path to save check point. Default is the
            current directory.

        max_num_checkpoints(int): The max number of local check points.
        epoch_interval(int): Every number of epoch to save check point.
        step_interval(int): Every number of step to save check point.

    Examples:
        >>> config = fluid.CheckpointConfig("./checkpoints")
        >>> trainer = fluid.Trainer(train_func=train_program,
        >>>                         place=place,
        >>>                         optimizer_func=optimizer_func,
        >>>                         checkpoint_config=config)
        >>> trainer.train(...)
    """

    def __init__(self,
                 checkpoint_dir=None,
                 max_num_checkpoints=3,
                 epoch_interval=1,
                 step_interval=10):

        assert epoch_interval >= 1
        assert step_interval >= 1

        self.checkpoint_dir = checkpoint_dir \
            if checkpoint_dir is not None else os.getcwd()
        self.max_num_checkpoints = max_num_checkpoints
        self.epoch_interval = epoch_interval
        self.step_interval = step_interval
        self.epoch_id = 0
        self.step_id = 0
        self.load_serial = None
        self.pserver_id = None
        self.lookup_table_name = None


def check_and_get_place(place):
    """
    Check the type of place or get the default place
    Args:
        place(None|core.CUDAPlace|core.CPUPlace): the place that trainer will be executed on.

    Raises:
        TypeError if the type mismatched.

    Returns:
        the original place if it is not None.
        if fluid is compiled with CUDA, returns CUDAPlace(0) by default.
        Otherwise returns CPUPlace by default.
    """
    if place is None:
        if core.is_compiled_with_cuda():
            return core.CUDAPlace(0)
        else:
            return core.CPUPlace()
    else:
        if not isinstance(place, core.CUDAPlace) and not isinstance(
                place, core.CPUPlace):
            raise TypeError("Place should be either CUDAPlace or CPUPlace")
        return place


class Trainer(object):
    """
    A trainer wraps MultiGPU/MultiNode training loops and can be used to train a
    simple neural network easily.

    This API takes a :code:`train_func`. A :code:`train_func` is a function that
    return loss as it first return value. The reset value can be fetched by
    EndStepEvent.metrics

    This API also takes a :code:`optimizer_func` that will return an optimizer
    instance.

    For example, to train a MLP for MNIST dataset, the sample program is

    >>> import paddle.fluid as fluid
    >>>
    >>> def mlp(image, layer_sizes=[200, 100], activation="relu", num_classes=10):
    >>>     hidden = image
    >>>     for layer_size in layer_sizes:
    >>>         hidden = fluid.layers.fc(input=hidden, size=layer_size, act=activation)
    >>>     return fluid.layers.fc(input=hidden, size=num_classes, act="softmax")
    >>>
    >>> def train_mnist_mlp():
    >>>     img = fluid.layers.data(name='image', shape=[784])
    >>>     label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    >>>     prediction = mlp(img)
    >>>     return fluid.layers.mean(fluid.layers.cross_entropy(prediction, label))
    >>>
    >>> def optimizer():
    >>>     return fluid.optimizer.Adam()
    >>>
    >>> trainer = Trainer(train_func=train_mnist_mlp,
    >>>                   optimizer_func=optimizer,
    >>>                   place=fluid.CUDAPlace(0),
    >>>                   parallel=True)
    >>>
    >>> def train_callback(event):
    >>>     if isinstance(event, fluid.EndStepEvent):
    >>>         print "Epoch ID", event.epoch, "Step ID",\
    >>>             event.step, "AvgLoss", event.metrics[0]
    >>>     elif isinstance(event, fluid.EndEpochEvent):
    >>>         trainer.save_params("./model_{0}".format(event.epoch))
    >>>
    >>> trainer.train(num_epochs=100, event_handler=train_callback)

    For more example, please see :ref:`api_guide_high_level_api`.


    Args:
        train_func(callable): A function which will return loss. The loss must be
            a scalar tensor.
        optimizer_func(callable): A function that returns an Optimizer object.
        place(CUDAPlace|CPUPlace): The device place of this trainer. If
            :code:`parallel=True,` all CUDA Places will be used if :code:`place`
            is a :code:`CUDAPlace`.
        parallel(bool): True if use multiple devices.
        checkpoint_config(CheckpointConfig): Configuration about how to save
            checkpoints.
    """

    def __init__(self,
                 train_func,
                 optimizer_func,
                 param_path=None,
                 place=None,
                 parallel=False,
                 checkpoint_config=None):
        self.__stop = False
        self.parallel = parallel

        # config for checkpoint
        # only chief worker will save variables
        self.trainer_id = 0
        self.checkpoint_cfg = checkpoint_config
        if self.checkpoint_cfg:
            assert isinstance(self.checkpoint_cfg, CheckpointConfig)
            serial = _get_latest_checkpoint_serial(
                self.checkpoint_cfg.checkpoint_dir)
            self.checkpoint_cfg.load_serial = serial if serial >= 0 else None

        self.scope = core.Scope()

        # 1. we need to generate a framework.Program by calling
        # program_func. Reference: fluid.program_guard in
        # test_word2vec.py

        self.startup_program = framework.Program()
        self.train_program = framework.Program()

        with framework.program_guard(self.train_program, self.startup_program):
            program_func_outs = train_func()
            self.train_func_outputs = program_func_outs if isinstance(
                program_func_outs, list) else [program_func_outs]
            self.test_program = self.train_program.clone(for_test=True)

            # The first element of program_func_outs is loss.
            loss = self.train_func_outputs[0]

            optimizer = optimizer_func()
            if not isinstance(optimizer, opt_module.Optimizer):
                raise TypeError(
                    "The optimizer should be an instance of Optimizer")
            optimize_ops, params_grads = optimizer.minimize(loss)

        self.place = check_and_get_place(place)

        self._dist_transpile_if_necessary(optimize_ops, params_grads)

        # 2. move the default_main_program to self.program and run the
        # default_startup program on an empty core.Scope()
        # Run startup program
        with self._prog_and_scope_guard():
            exe = executor.Executor(place)
            exe.run(self.startup_program)

        if self.checkpoint_cfg and self.checkpoint_cfg.load_serial is not None:
            self._load_checkpoint()

        if param_path and os.path.isdir(param_path):
            # load params from param_path into scope
            io.load_persistables(
                executor=exe,
                dirname=param_path,
                main_program=self.startup_program)

    def _transpile_nccl2_dist(self):
        # PADDLE_TRAINER_IPS
        if "PADDLE_TRAINER_IPS" not in os.environ:
            self.nccl_id_var = None
        else:
            self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
            port = os.getenv("PADDLE_PSERVER_PORT")
            worker_ips = os.getenv("PADDLE_TRAINER_IPS")
            worker_endpoints = []
            for ip in worker_ips.split(","):
                worker_endpoints.append(':'.join([ip, port]))
            self.num_trainers = len(worker_endpoints)
            current_endpoint = os.getenv("PADDLE_CURRENT_IP") + ":" + port
            worker_endpoints.remove(current_endpoint)
            # TODO(wuyi): use self.nccl_id_var, self.num_trainers and self.trainer_id
            # in ParallelExecutor to start
            # distributed training using NCCL2
            self.nccl_id_var = self.startup_program.global_block().create_var(
                name="NCCLID", persistable=True, type=core.VarDesc.VarType.RAW)
            self.startup_program.global_block().append_op(
                type="gen_nccl_id",
                inputs={},
                outputs={"NCCLID": self.nccl_id_var},
                attrs={
                    "endpoint": current_endpoint,
                    "endpoint_list": worker_endpoints,
                    "trainer_id": self.trainer_id
                })

    def _dist_transpile_if_necessary(self, optimize_ops, params_grads):
        self._transpile_nccl2_dist()
        if self.nccl_id_var != None:
            return

        if "PADDLE_TRAINING_ROLE" not in os.environ:
            return

        # the port of all pservers, needed by both trainer and pserver
        port = os.getenv("PADDLE_PSERVER_PORT", "6174")
        # comma separated ips of all pservers, needed by trainer and
        # pserver
        pserver_ips = os.getenv("PADDLE_PSERVER_IPS", "")
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)
        # total number of workers/trainers in the job, needed by
        # trainer and pserver
        trainers = int(os.getenv("PADDLE_TRAINERS"))
        # the IP of the local machine, needed by pserver only
        current_endpoint = os.getenv("PADDLE_CURRENT_IP", "") + ":" + port
        # the unique trainer id, starting from 0, needed by trainer
        # only
        self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))

        # the role, should be either PSERVER or TRAINER
        training_role = os.getenv("PADDLE_TRAINING_ROLE")
        with self._prog_and_scope_guard():
            t = distribute_transpiler.DistributeTranspiler()
            t.transpile(
                self.trainer_id, pservers=pserver_endpoints, trainers=trainers)
            if training_role == "PSERVER":
                if self.checkpoint_cfg:
                    pserver_id = eplist.index(current_endpoint)
                    self.checkpoint_cfg.pserver_id = pserver_id
                    if t.has_distributed_lookup_table:
                        self.checkpoint_cfg.lookup_table_name = t.table_name

                self.train_program = t.get_pserver_program(current_endpoint)
                self.startup_program = t.get_startup_program(current_endpoint,
                                                             self.train_program)
            elif training_role == "TRAINER":
                self.train_program = t.get_trainer_program()
            else:
                raise ValueError(
                    'TRAINING_ROLE environment variable must be either TRAINER or PSERVER'
                )

    def stop(self):
        """
        stop training
        """
        self.__stop = True

    def train(self, num_epochs, event_handler, reader=None, feed_order=None):
        """
        Start the train loop to train the model.

        Args:
            num_epochs(int): The number of epoch. An epoch will process all data in reader
            event_handler(callable): The event handler. A function with type (ev:Event)->void
            reader(callable): A reader creator object. See also
                :ref:`api_guide_python_reader` .
            feed_order(list): Feeding order of reader. None will following the defining
                order in program

        Returns:
            None
        """
        training_role = os.getenv("PADDLE_TRAINING_ROLE", "")
        if training_role == "PSERVER":
            with self._prog_and_scope_guard():
                exe = executor.Executor(self.place)
                exe.run()
                return
        if self.parallel:
            self._train_by_parallel_executor(num_epochs, event_handler, reader,
                                             feed_order)
        else:
            self._train_by_executor(num_epochs, event_handler, reader,
                                    feed_order)

    def test(self, reader, feed_order):
        """
        Test the model on given test data

        Args:
            reader(callable): The reader that yields test data.
            feed_order(list): Feeding order of reader. None will following the
                defining order in program
        """

        return self._test_by_executor(reader, feed_order,
                                      self.train_func_outputs)

    def save_params(self, param_path):
        """
        Save all parameters into :code:`param_path`.

        Args:
            param_path(str): The path to save parameters.

        Returns:
            None
        """
        with self._prog_and_scope_guard():
            exe = executor.Executor(self.place)
            io.save_persistables(exe, dirname=param_path)

    @contextlib.contextmanager
    def _prog_and_scope_guard(self):
        with framework.program_guard(
                main_program=self.train_program,
                startup_program=self.startup_program):
            with executor.scope_guard(self.scope):
                yield

    def _train_by_executor(self, num_epochs, event_handler, reader, feed_order):
        """
        Train by Executor and single device.

        Args:
            num_epochs:
            event_handler:
            reader:
            feed_order:

        Returns:

        """
        with self._prog_and_scope_guard():
            feed_var_list = build_feed_var_list(self.train_program, feed_order)
            feeder = data_feeder.DataFeeder(
                feed_list=feed_var_list, place=self.place)
            exe = executor.Executor(self.place)
            reader = feeder.decorate_reader(reader, multi_devices=False)
            self._train_by_any_executor(event_handler, exe, num_epochs, reader)

    def _train_by_any_executor(self, event_handler, exe, num_epochs, reader):
        if self.checkpoint_cfg:
            epochs = [
                epoch_id for epoch_id in range(num_epochs)
                if epoch_id >= self.checkpoint_cfg.epoch_id
            ]
        else:
            epochs = [epoch_id for epoch_id in range(num_epochs)]

        for epoch_id in epochs:
            event_handler(BeginEpochEvent(epoch_id))
            for step_id, data in enumerate(reader()):
                if self.__stop:
                    if self.checkpoint_cfg:
                        self._clean_checkpoint()
                    return

                if self.checkpoint_cfg and self.checkpoint_cfg.load_serial \
                    and self.checkpoint_cfg.step_id >= step_id and self.checkpoint_cfg.epoch_id == epoch_id:
                    continue

                begin_event = BeginStepEvent(epoch_id, step_id)
                event_handler(begin_event)
                if begin_event.fetch_metrics:
                    metrics = exe.run(feed=data,
                                      fetch_list=[
                                          var.name
                                          for var in self.train_func_outputs
                                      ])
                else:
                    metrics = exe.run(feed=data, fetch_list=[])

                if self.checkpoint_cfg:
                    self._save_checkpoint(epoch_id, step_id)
                event_handler(EndStepEvent(epoch_id, step_id, metrics))
            event_handler(EndEpochEvent(epoch_id))
        if self.checkpoint_cfg:
            self._clean_checkpoint()

    def _test_by_executor(self, reader, feed_order, fetch_list):
        with executor.scope_guard(self.scope):
            feed_var_list = build_feed_var_list(self.test_program, feed_order)
            feeder = data_feeder.DataFeeder(
                feed_list=feed_var_list, place=self.place)
            exe = executor.Executor(self.place)
            accumulated = len(fetch_list) * [0]
            count = 0
            for data in reader():
                outs = exe.run(program=self.test_program,
                               feed=feeder.feed(data),
                               fetch_list=fetch_list)
                accumulated = [x[0] + x[1][0] for x in zip(accumulated, outs)]
                count += 1

            return [x / count for x in accumulated]

    def _train_by_parallel_executor(self, num_epochs, event_handler, reader,
                                    feed_order):
        with self._prog_and_scope_guard():
            pe = self._get_or_create_parallel_executor()
            feed_var_list = build_feed_var_list(self.train_program, feed_order)
            feeder = data_feeder.DataFeeder(
                feed_list=feed_var_list, place=self.place)
            reader = feeder.decorate_reader(reader, multi_devices=True)
            self._train_by_any_executor(event_handler, pe, num_epochs, reader)

    def _get_parallel_executor(self):
        return getattr(self, 'parallel_executor', None)

    def _get_or_create_parallel_executor(self):
        if self._get_parallel_executor() is None:
            self.parallel_executor = parallel_executor.ParallelExecutor(
                use_cuda=isinstance(self.place, core.CUDAPlace),
                loss_name=self.train_func_outputs[0].name)
        return self._get_parallel_executor()

    def _clean_checkpoint(self):
        assert self.checkpoint_cfg
        clean_checkpoint(checkpoint_dir=self.checkpoint_cfg.checkpoint_dir)

    def _get_checkpoint_load_args(self):
        """
        epoch_id and step_id are runtime arguments, they are not variables, will load them independently.
        """
        return ["epoch_id", "step_id"]

    def _get_checkpoint_save_args(self, epoch_id, step_id):
        """
        epoch_id and step_id are runtime arguments, they are not variables, will save them independently.
        """
        trainer_args = {}
        trainer_args["epoch_id"] = epoch_id
        trainer_args["step_id"] = step_id
        return trainer_args

    def _save_checkpoint(self, epoch_id, step_id):
        assert self.checkpoint_cfg

        if epoch_id % self.checkpoint_cfg.epoch_interval == 0 \
            and step_id % self.checkpoint_cfg.step_interval == 0:
            exe = executor.Executor(self.place)
            save_checkpoint(
                executor=exe,
                checkpoint_dir=self.checkpoint_cfg.checkpoint_dir,
                trainer_id=self.trainer_id,
                trainer_args=self._get_checkpoint_save_args(epoch_id, step_id),
                main_program=self.train_program,
                max_num_checkpoints=self.checkpoint_cfg.max_num_checkpoints)

    def _load_checkpoint(self):
        with self._prog_and_scope_guard():
            exe = executor.Executor(self.place)
            load_checkpoint(
                executor=exe,
                checkpoint_dir=self.checkpoint_cfg.checkpoint_dir,
                main_program=self.startup_program)

            if not self.checkpoint_cfg.pserver_id:
                load_trainer_args = self._get_checkpoint_load_args()
                trainer_args = load_checkpoint(
                    executor=exe,
                    checkpoint_dir=self.checkpoint_cfg.checkpoint_dir,
                    main_program=self.startup_program,
                    role_id=self.trainer_id,
                    is_trainer=True,
                    load_trainer_args=load_trainer_args)

                if len(trainer_args) != 2:
                    raise ValueError(
                        "the return trainer_args length do not equal _get_checkpoint_load_args"
                    )
                self.checkpoint_cfg.epoch_id = int(trainer_args[0])
                self.checkpoint_cfg.step_id = int(trainer_args[1])
            else:
                if self.checkpoint_cfg.lookup_table_name:
                    load_checkpoint(
                        executor=exe,
                        checkpoint_dir=self.checkpoint_cfg.checkpoint_dir,
                        main_program=self.startup_program,
                        role_id=self.checkpoint_cfg.pserver_id,
                        is_trainer=False,
                        load_trainer_args=None,
                        load_lookup_table=self.checkpoint_cfg.lookup_table_name)


def build_feed_var_list(program, feed_order):
    if not isinstance(program, framework.Program):
        raise TypeError("The 'program' should be an object of Program")

    if isinstance(feed_order, list):
        feed_var_list = [
            program.global_block().var(var_name) for var_name in feed_order
        ]
    else:
        if not isinstance(feed_order, dict):
            raise TypeError(
                "The 'feed_order' should be either None, list or dict.")
        if not sorted(feed_order.values()) == list(range(len(feed_order))):
            raise ValueError(
                "The values of 'feed_order' should be a permutation of [0, len(feed_order))"
            )
        sorted_pair_list = sorted(
            list(feed_order.items()), key=lambda item: item[1])
        feed_var_list = [
            program.global_block().var(pair[0]) for pair in sorted_pair_list
        ]
    return feed_var_list


# move Checkpoint APIs from io.py to trainer.py, make all of them are private.
SUCCESS_MARK_FILENAME = "_SUCCESS"
CHECKPOINT_PREFIX = "checkpoint"
MODEL_DIR = "__model__"
LOOKUP_TABLE_DIR = "__lookup_table__"
TRAINER_PREFIX = "trainer"
CHECKPOINT_SEPARATOR = "_"


def save_checkpoint(executor,
                    checkpoint_dir,
                    trainer_id,
                    main_program,
                    trainer_args=None,
                    max_num_checkpoints=3,
                    lookup_table=None,
                    pserver_endpoints=None):
    """
    This function filters out all checkpoint variables from the give
    main_program and then saves these variables to the `checkpoint_dir`
    directory.

    In the training precess, we generally save a checkpoint in each
    iteration. So there might be a lot of checkpoints in the
    `checkpoint_dir`. To avoid them taking too much disk space, the
    `max_num_checkpoints` are introduced to limit the total number of
    checkpoints. If the number of existing checkpints is greater than
    the `max_num_checkpoints`, oldest ones will be scroll deleted.

    A variable is a checkpoint variable and will be saved if it meets
    all following conditions:
        1. It's persistable.
        2. It's type is not FEED_MINIBATCH nor FETCH_LIST nor RAW.
        3. It's name contains no "@GRAD" nor ".trainer_" nor ".block".

    Args:
        executor(Executor): The executor to run for save checkpoint.
        checkpoint_dir(str): The folder where to save checkpoints.
        trainer_id(int): currect trainer id, if id is equal to 0, the trainer
            is chief.
        trainer_args(dict|None): Current training arguments. Such as 'epoch_id'
            and 'step_id'.
            Defaut: None
        main_program(Program): The program whose checkpoint variables will
            be saved.
        max_num_checkpoints(int): The max number of total number of existing
            checkpoints.
            Default: 3
        lookup_table(string|None): the lookup table name, when use distribute
            lookup table, we can get lookup table name by DistributeTranspiler.
            table_name
        pserver_endpoints(list|None): the parameter server ip:port list.
            when use distribute lookup table, we can get pserver_endpoints by
            distribute arguments.

    Returns:
        None

    Raises:
        ValueError: If `checkpoint_dir` is None.
        AssertionError: If `trainer_args` is not a dict.

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            path = "./checkpoints"
            prog = fluid.default_main_program()
            trainer_args = {"epoch_id": 200,
                            "step_id": 20} # just an example
            table_name = "share_w"
            ps_endpoints = ["127.0.0.1:6000","127.0.0.1:6001"]

            save_checkpoint(executor=exe,
                                     checkpoint_dir=path,
                                     trainer_id=0,
                                     trainer_args=trainer_args,
                                     main_program=prog,
                                     max_num_checkpoints=3,
                                     lookup_table=table_name,
                                     pserver_endpoints = ps_endpoints)
    """
    if checkpoint_dir is None:
        raise ValueError("'checkpoint_dir' should not be None")

    if main_program is None:
        raise ValueError('main_program should not be None.')

    if trainer_args:
        assert isinstance(trainer_args, dict)

    is_chief = trainer_id == 0

    _make_chekcpoint_dirs(checkpoint_dir)
    serial = _get_latest_checkpoint_serial(checkpoint_dir) + 1
    cur_dir = _get_serial_dir(checkpoint_dir, serial)

    _save_trainer_args(cur_dir, trainer_id, trainer_args)

    if is_chief:
        _save_persist_vars_without_grad(executor, cur_dir, main_program)

    if is_chief and lookup_table and pserver_endpoints:
        _save_pserver_vars_by_notify(executor, cur_dir, lookup_table,
                                     pserver_endpoints)

    _scroll_delete(checkpoint_dir, max_num_checkpoints)


def load_checkpoint(executor,
                    checkpoint_dir,
                    main_program,
                    role_id=0,
                    is_trainer=True,
                    load_trainer_args=None,
                    load_lookup_table=None):
    """
    This function filters out all checkpoint variables from the give
    main_program and then try to load these variables from the
    `checkpoint_dir` directory.

    In the training precess, we generally save a checkpoint in each
    iteration. So there are more than one checkpoint in the
    `checkpoint_dir` (each checkpoint has its own sub folder), use
    `serial` to specify which serial of checkpoint you would like to
    load.

    A variable is a checkpoint variable and will be loaded if it meets
    all following conditions:
        1. It's persistable.
        2. It's type is not FEED_MINIBATCH nor FETCH_LIST nor RAW.
        3. It's name contains no "@GRAD" nor ".trainer_" nor ".block".

    Args:
        executor(Executor): The executor to run for loading checkpoint.
        checkpoint_dir(str): The folder where all checkpoints are.
        serial(int): The serial of checkpoint you would like to load.
        main_program(Program): The program whose checkpoint variables will
                               be loaded.
        role_id(int):  the trainer id or the parameter server id.
        is_trainer(bool): trainer is True and parameter server is False.
        load_trainer_args(list|None): list about load trainer args.
        load_lookup_table(str|None): the lookup table name

    Returns:
        None

    Raises:
        ValueError: If `checkpoint_dir` is None.
        ValueError: If `main_program` is None.

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            path = "./checkpoints"
            prog = fluid.default_main_program()
            load_checkpoint(executor=exe, checkpoint_dir=path,
                    serial=9, main_program=prog)

            # In this example, `load_checkpoint` function
            # will first filters out all checkpoint variables in the default
            # main program, and then try to load these variables form the
            # folder "./checkpoints/checkpoint_9/__model__".
    """

    if checkpoint_dir is None:
        raise ValueError("'checkpoint_dir' should not be None")

    serial = _get_latest_checkpoint_serial(checkpoint_dir)

    # there are nothing  need to be loaded
    if serial is None or serial < 0:
        return

    if main_program is None:
        raise ValueError('main_program should not be None.')

    if is_trainer and load_trainer_args is None:
        cur_dir = _get_serial_dir(checkpoint_dir, serial)
        _load_persist_vars_without_grad(executor, cur_dir, main_program, True)
        return

    if is_trainer and load_trainer_args:
        return _load_trainer_args(checkpoint_dir, serial, role_id,
                                  load_trainer_args)

    if not is_trainer and load_lookup_table:
        _load_lookup_table_vars(executor, checkpoint_dir, main_program, role_id,
                                load_lookup_table)


def clean_checkpoint(checkpoint_dir, delete_dir=False):
    """
    clean the checkpoint dir, when the train exits normally,
    the trainer will call clean_checkpoint to delete checkpoint directory saved before.
    delete_dir only works when the directory is empty, otherwise, OSError is raised.

    : param checkpoint_dir
    : param delete_dir
    """

    if checkpoint_dir is None:
        raise ValueError("'checkpoint_dir' should not be None")
    _scroll_delete(checkpoint_dir, max_num_checkpoints=0)

    if delete_dir and not os.listdir(checkpoint_dir):
        os.rmdir(checkpoint_dir)


def _load_persist_vars_without_grad(executor,
                                    dirname,
                                    program,
                                    has_model_dir=False):
    """
    This function filters out all checkpoint variables from the give
    program and then trys to load these variables from the given directory.

    A variable is a checkpoint variable if it meets all following
    conditions:
        1. It's persistable.
        2. It's type is not FEED_MINIBATCH nor FETCH_LIST nor RAW.
        3. It's name contains no "@GRAD" nor ".trainer_" nor ".block".

    Args:
        executor(Executor): The executor to run for loading variables.
        dirname(str): The directory path.
        program(Program): The program whose checkpoint variables will
                          be loaded.
        has_model_dir(bool): if True, the function loads variables
                             from a sub directory named '__model__'.
                             Default: False

    Returns:
        None

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"
            prog = fluid.default_main_program()
            _load_persist_vars_without_grad(executor=exe,
                    dirname=param_path, program=prog, has_model_dir=True)

            # In this example, `_load_persist_vars_without_grad` function
            # will first filters out all checkpoint variables in the default
            # main program, and then trys to load these variables form the
            # folder "./my_paddle_model/__model__".
    """

    if has_model_dir:
        dirname = _get_model_dir(dirname)

    io.load_vars(
        executor,
        dirname=dirname,
        main_program=program,
        predicate=_is_checkpoint_var,
        filename=None)


def _load_lookup_table_vars(executor, dirname, program, pserver_id, table_name):
    """
    The parameter server will load lookup table's local file in
    selectedrows variable.

    Args:
        executor(Executor): The executor to run for loading persistable variables
        dirname(str): The directory path
        main_program(Program): Find the variable named table_name in main_program
        pserver_id(int): the serial number in pserver_endpoints list
        table_name(str): lookup table name

    Returns:
        None

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            dirname = "./checkpoints/checkpoint_9/"
            prog = fluid.default_main_program()
            pserver_id = 1
            table_name = "share_w"
            _load_lookup_table_vars(executor=exe,
                    dirname=dirname, program=prog, pserver_id=pserver_id,
                    table_name=table_name)
    """

    for var in program.list_vars():
        if var.name == table_name:
            lookup_table_var = var
            break

    assert lookup_table_var is not None

    lookup_table_dir = os.path.join(dirname, LOOKUP_TABLE_DIR)
    table_file = table_name + CHECKPOINT_SEPARATOR + str(pserver_id)

    load_prog = framework.Program()
    load_block = load_prog.global_block()

    load_block.append_op(
        type='load',
        inputs={},
        outputs={'Out': [lookup_table_var]},
        attrs={'file_path': os.path.join(lookup_table_dir, table_file)})

    executor.run(load_prog)


def _save_persist_vars_without_grad(executor, dirname, program):
    """
    This function filters out all checkpoint variables from the give
    program and then save these variables to a sub-folder '__model__' of
    the given directory.

    A variable is a checkpoint variable if it meets all following
    conditions:
        1. It's persistable.
        2. It's type is not FEED_MINIBATCH nor FETCH_LIST nor RAW.
        3. It's name contains no "@GRAD" nor ".trainer_" nor ".block".

    Args:
        executor(Executor): The executor to run for saving variables.
        dirname(str): The directory path.
        program(Program): The program whose checkpoint variables will
                          be saved.

    Returns:
        None

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"
            prog = fluid.default_main_program()
            _save_persist_vars_without_grad(executor=exe,
                    dirname=param_path, program=prog)

            # In this example, `_save_persist_vars_without_grad` function
            # will first filters out all checkpoint variables in the default
            # main program, and then saves these variables to the folder
            # "./my_paddle_model/__model__".
    """
    cur_dir = _get_model_dir(dirname)
    io.save_vars(
        executor,
        dirname=cur_dir,
        main_program=program,
        vars=None,
        predicate=_is_checkpoint_var,
        filename=None)
    _write_success(cur_dir)


def _save_pserver_vars_by_notify(executor, dirname, lookup_table,
                                 ps_endpoint_list):
    """
    This function will send checkpoint notify message from Trainer 0
    to all the pservers.
    The checkpoint notify message contains lookup table name,
    the absolute path on pserver to save lookup_table.

    Args:
        executor(Executor): The executor to run for send checkpoint notify.
        dirname(str): The folder where to save checkpoints.
        lookup_table(string): the lookup table name, when use distribute
            lookup table, we can get lookup table name by DistributeTranspiler.
            table_name
        ps_endpoint_list(list): the parameter server ip:port list.
            when use distribute lookup table, we can get ps_endpoint_list by
            distribute arguments.
    Return:
        None

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"
            prog = fluid.default_main_program()
            table_name = "share_w"
            ps_endpoints = ["127.0.0.1:6000","127.0.0.1:6001"]

            _save_pserver_vars_by_notify(executor=exe,
                    dirname=param_path, lookup_table=table_name,
                    ps_endpoint_list=ps_endpoints)
    """
    cur_dir = _get_lookuptable_dir(dirname)

    checkpoint_notify_program = framework.Program()
    checkpoint_notify_block = checkpoint_notify_program.global_block()

    attrs = {}
    attrs['epmap'] = ps_endpoint_list
    attrs['dir'] = cur_dir
    attrs['lookup_table'] = lookup_table

    checkpoint_notify_block.append_op(
        type='checkpoint_notify', inputs={}, outputs={}, attrs=attrs)
    executor.run(checkpoint_notify_program)


def _save_trainer_args(dirname, trainer_id, trainer_args):
    assert isinstance(trainer_args, dict)

    cur_dir = _get_trainer_dir(dirname, trainer_id)

    for name, value in list(trainer_args.items()):
        args_file = os.path.join(cur_dir, name)
        with open(args_file, 'w') as f:
            f.write(str(value))
    _write_success(cur_dir)


def _load_trainer_args(checkpoint_dir, serial, trainer_id, trainer_args):
    """
    trainer will load some args from it's independent directory,
    such as epoch_id and step_id.

    Args:
        checkpoint_dir(str): The folder where all checkpoints are.
        serial(int): The serial of checkpoint you would like to load.
        trainer_id(int): current trainer id.
        trainer_args(list): list about load trainer args
    Return:
        None

    Examples:
        .. code-block:: python

            param_path = "./checkpoint/"
            serial = 7
            trainer_id = 2
            trainer_args = ["epoch_id", "step_id"]

            _load_trainer_args(checkpoint_dir=param_path, serial=serial,
            trainer_id=trainer_id, trainer_args=trainer_args)
    """
    assert isinstance(trainer_args, list)

    cur_dir = _get_serial_dir(checkpoint_dir, serial)
    cur_dir = _get_trainer_dir(cur_dir, trainer_id)

    ret_values = []

    for arg in trainer_args:
        cur_file = os.path.join(cur_dir, arg)
        with open(cur_file, 'r') as f:
            contents = f.read()
            ret_values.append(contents.strip())
    return ret_values


def _is_checkpoint_var(var):
    """
    the checkpoint will not save or load all the variables.
    var type is FEED_MINIBATCH/FETCH_LIST/RAW or var name ends with @GRAD are discarded.

    : param var(Variable)
    """
    if var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
            var.desc.type() == core.VarDesc.VarType.FETCH_LIST or \
            var.desc.type() == core.VarDesc.VarType.RAW:
        return False
    # @GRAD are named for gradient variables, checkpoint will not save it.
    if "@GRAD" in var.name:
        return False
    # .trainer_ are named for distribute train variables, checkpoint will not save it.
    if ".trainer_" in var.name:
        return False

    # .block is named for distribute train variables, checkpoint will not save it.
    if ".block" in var.name:
        return False

    return var.persistable


def _make_chekcpoint_dirs(dirs):
    """
    _make_chekcpoint_dirs will makdir local directory directly, when the directory is exist, it will igore it.
    """
    assert dirs is not None

    if os.path.isfile(dirs):
        raise OSError(errno.ENOTDIR, "dirs path shoule be a Directory.", dirs)

    if not os.path.isdir(dirs):
        try:
            os.makedirs(dirs)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise err


def _get_dir_serial(dirname):
    _, serial = dirname.split(CHECKPOINT_SEPARATOR)

    try:
        serial_num = int(serial)
    except ValueError:
        serial_num = -1
    return serial_num


def _get_serial_dir(dirname, serial):
    serial_folder = CHECKPOINT_PREFIX + CHECKPOINT_SEPARATOR + str(serial)
    serial_dir = os.path.join(dirname, serial_folder)
    _make_chekcpoint_dirs(serial_dir)

    return serial_dir


def _get_model_dir(dirname):
    model_dir = os.path.join(dirname, MODEL_DIR)
    _make_chekcpoint_dirs(model_dir)
    return model_dir


def _get_lookuptable_dir(dirname):
    lookuptable_dir = os.path.join(dirname, LOOKUP_TABLE_DIR)
    _make_chekcpoint_dirs(lookuptable_dir)
    return lookuptable_dir


def _get_trainer_dir(dirname, trainer_id):
    trainer_folder = TRAINER_PREFIX + CHECKPOINT_SEPARATOR + str(trainer_id)
    trainer_dir = os.path.join(dirname, trainer_folder)
    _make_chekcpoint_dirs(trainer_dir)
    return trainer_dir


def _scroll_delete(dirname, max_num_checkpoints=3):
    dirs = os.listdir(dirname)
    serial_map = {}
    for serial in dirs:
        serial_num = _get_dir_serial(serial)
        serial_map[serial_num] = serial

    if len(list(serial_map.keys())) <= max_num_checkpoints:
        return

    serials = list(serial_map.keys())
    serials.sort(reverse=True)
    serials = serials[max_num_checkpoints:]
    for serial in serials:
        cur_dir = _get_serial_dir(dirname, serial)
        try:
            shutil.rmtree(cur_dir)
        except OSError as err:
            if err.errno != errno.ENOENT:
                raise err


def _write_success(dirname):
    """
    write an empty file named "_SUCCESS" in checkpoint dir, indicate this checkpoint is correct.

    : param dirname
    """
    success_file = os.path.join(dirname, SUCCESS_MARK_FILENAME)
    with open(success_file, 'a') as f:
        now = time.ctime()
        f.write(now)


def _get_latest_checkpoint_serial(checkpoint_dir):
    """
    get the latest file in checkpoint directory, the _SUCCESS file must exist in the directory

    : param checkpoint_dir
    """
    if not checkpoint_dir:
        return -1

    def has_success(checkpoint_dir, cur_dir):
        """
        is _SUCCESS in this dir
        """

        serial = _get_dir_serial(cur_dir)
        if serial == -1 or not os.path.isdir(
                os.path.join(checkpoint_dir, cur_dir)):
            return -1

        success_path = os.path.join(
            _get_serial_dir(checkpoint_dir, serial), MODEL_DIR,
            SUCCESS_MARK_FILENAME)
        if os.path.isfile(success_path):
            return serial

    if not os.path.isdir(checkpoint_dir):
        return -1

    current_dir = -1
    dirs = os.listdir(checkpoint_dir)
    for cur_dir in dirs:
        success_num = has_success(checkpoint_dir, cur_dir)
        if success_num > current_dir:
            current_dir = success_num
    return current_dir
