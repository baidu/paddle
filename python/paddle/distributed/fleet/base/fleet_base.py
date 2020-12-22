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

from __future__ import print_function
import copy
import warnings
import paddle
import os
from paddle.fluid.framework import dygraph_only
from paddle.fluid import compiler
from .role_maker import UserDefinedRoleMaker, PaddleCloudRoleMaker, RoleMakerBase
from .strategy_compiler import StrategyCompiler
from .distributed_strategy import DistributedStrategy
from .meta_optimizer_factory import MetaOptimizerFactory
from .runtime_factory import RuntimeFactory
from paddle.fluid.wrapped_decorator import wrap_decorator
from paddle.fluid.dygraph import parallel_helper


def _inited_runtime_handler_(func):
    def __impl__(*args, **kwargs):
        cls = args[0]

        if cls._runtime_handle is None:
            raise ValueError("Fleet can not find suitable runtime handler")

        return func(*args, **kwargs)

    return __impl__


def _is_non_distributed_check_(func):
    def __impl__(*args, **kwargs):
        cls = args[0]

        if cls._role_maker is not None and cls._role_maker._is_non_distributed(
        ) is True:
            warnings.warn(
                "%s() function doesn't work when use non_distributed fleet." %
                (func.__name__))
            return

        return func(*args, **kwargs)

    return __impl__


inited_runtime_handler = wrap_decorator(_inited_runtime_handler_)
is_non_distributed_check = wrap_decorator(_is_non_distributed_check_)


class Fleet(object):
    """
    Unified API for distributed training of PaddlePaddle
    Please reference the https://github.com/PaddlePaddle/FleetX for details


    Returns:
        Fleet: A Fleet instance

    Example for collective training:

        .. code-block:: python

            import paddle
            paddle.enable_static()
            import paddle.distributed.fleet as fleet

            fleet.init(is_collective=True)

            strategy = fleet.DistributedStrategy()
            optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)

            # do distributed training


    Example for parameter server training:

        .. code-block:: python

            import paddle
            paddle.enable_static()
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            fleet.init(strategy=strategy)

            optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            optimizer = fleet.distributed_optimizer(optimizer)

            if fleet.is_first_worker():
                print("this is first worker")

            print("current node index: {}".format(fleet.worker_index()))
            print("total number of worker num: {}".format(fleet.worker_num()))

            if fleet.is_worker():
                print("this is worker")
            print("worker endpoints: {}".format(fleet.worker_endpoints(to_string=True)))

            print("server num: {}".format(fleet.server_num()))
            print("server endpoints: {}".format(fleet.server_endpoints(to_string=True)))

            if fleet.is_server():
                print("this is server")
            fleet.stop_worker()


    """

    def __init__(self):
        self._role_maker = None
        self.strategy_compiler = None
        self._is_collective = False
        self._runtime_handle = None
        self._util = None
        self._context = {}

    def init(self, role_maker=None, is_collective=False, strategy=None):
        """
        Initialize role_maker in Fleet.

        This function is responsible for the distributed architecture
        what you want to run your code behind.

        Args:
            role_maker (RoleMakerBase, optional): A ``RoleMakerBase`` containing the configuration
                of environment variables related to distributed training.If you did not initialize 
                the rolemaker by yourself, it will be automatically initialized to PaddleRoleMaker.
                The default value is None.
            is_collective (Boolean, optional): A ``Boolean`` variable determines whether the program 
                runs on the CPU or GPU. False means set distributed training using CPU, and True means
                GPU.The default value is False.The default value is False.
            strategy (DistributedStrategy): Extra properties for distributed training. 
                For details, please refer to paddle.distributed.fleet.DistributedStrategy. Default: None.


        Returns:
            None

        Examples1:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

        Examples2:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init(is_collective=True)

        Examples3:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                role = fleet.PaddleCloudRoleMaker()
                fleet.init(role)

        Examples4:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                strategy = fleet.DistributedStrategy()
                fleet.init(strategy=strategy)

        """
        if strategy is None:
            strategy = DistributedStrategy()
        self._user_defined_strategy = copy.deepcopy(strategy)

        if role_maker is None:
            if isinstance(is_collective, bool):
                self._is_collective = is_collective
                self._role_maker = PaddleCloudRoleMaker(
                    is_collective=self._is_collective)
            else:
                raise ValueError(
                    "`is_collective` should be instance of `bool`, but got {}".
                    format(type(is_collective)))
        else:
            if isinstance(role_maker, RoleMakerBase):
                self._role_maker = role_maker
            else:
                raise ValueError(
                    "`role_maker` should be subclass of `RoleMakerBase`, but got {}".
                    format(type(role_maker)))
        self._role_maker._generate_role()

        import paddle.distributed.fleet as fleet
        fleet.util._set_role_maker(self._role_maker)

        self.strategy_compiler = StrategyCompiler()

        if self._role_maker._is_non_distributed() and self._is_collective:
            if paddle.fluid.core.is_compiled_with_cuda():
                gpus_num = paddle.fluid.core.get_cuda_device_count()
                if gpus_num != 1:
                    raise ValueError(
                        "CUDA_VISIBLE_DEVICES shoule be set only 1 card if you use `python` to launch fleet program."
                    )

        if paddle.fluid.framework.in_dygraph_mode():
            if self.worker_num() == 1:
                return
            if parallel_helper._is_parallel_ctx_initialized():
                warnings.warn(
                    "The dygraph parallel environment has been initialized.")
            else:
                # FLAGS_nccl_nrings is used for dynamic graph multi-stream communication
                if "FLAGS_nccl_nrings" in os.environ:
                    warnings.warn(
                        "You have set the environment variable FLAGS_nccl_nrings "
                        "outside the program, so the nccl_comm_num in "
                        "DistributedStrategy will not take effect here.")
                else:
                    os.environ["FLAGS_nccl_nrings"] = str(
                        self._user_defined_strategy.nccl_comm_num)
                paddle.distributed.init_parallel_env()

    def is_first_worker(self):
        """
        Check whether the node is the first instance of worker.

        Returns:
            bool: True if this is the first node of worker,
                  False if not.

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.is_first_worker()

        """
        return self._role_maker._is_first_worker()

    def worker_index(self):
        """
        Get current worker index.

        Returns:
            int: node id

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.worker_index()

        """
        return self._role_maker._worker_index()

    def worker_num(self):
        """
        Get current total worker number.

        Returns:
            int: worker numbers

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.worker_num()

        """
        return self._role_maker._worker_num()

    def is_worker(self):
        """
        Check whether the node is an instance of worker.

        Returns:
            bool: True if this is a node of worker,
                  False if not.

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.is_worker()

        """
        return self._role_maker._is_worker()

    def worker_endpoints(self, to_string=False):
        """
        Get current worker endpoints, such as ["127.0.0.1:1001", "127.0.0.1:1002"].

        Returns:
            list/string: server endpoints

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.worker_endpoints()

        """
        if to_string:
            return ",".join(self._role_maker._get_trainer_endpoints())
        else:
            return self._role_maker._get_trainer_endpoints()

    def server_num(self):
        """
        Get current total worker number.

        Returns:
            int: server number

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.server_num()
        """
        return len(self._role_maker._get_pserver_endpoints())

    def server_index(self):
        """
        Get current server index.

        Returns:
            int: node id

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.server_index()

        """
        return self._role_maker._server_index()

    def server_endpoints(self, to_string=False):
        """
        Get current server endpoints, such as ["127.0.0.1:1001", "127.0.0.1:1002"].

        Returns:
            list/string: server endpoints

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.server_endpoints()

        """

        if to_string:
            return ",".join(self._role_maker._get_pserver_endpoints())
        else:
            return self._role_maker._get_pserver_endpoints()

    def is_server(self):
        """
        Check whether the node is an instance of server.

        Returns:
            bool: True if this is a node of server,
                  False if not.

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()
                fleet.is_server()

        """
        return self._role_maker._is_server(
        ) or self._role_maker._is_heter_worker()

    def barrier_worker(self):
        """
        barrier all workers

        Returns:
            None
        """
        self._role_maker._barrier("worker")

    @is_non_distributed_check
    @inited_runtime_handler
    def init_worker(self):
        """
        initialize `Communicator` for parameter server training.


        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                fleet.init_worker()

        """
        self._runtime_handle._init_worker()

    @is_non_distributed_check
    @inited_runtime_handler
    def init_server(self, *args, **kwargs):
        """
        init_server executor to initialize startup program,
        if the `args` is not empty, it will run load_persistables for increment training.


        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                fleet.init_server()

        """
        self._runtime_handle._init_server(*args, **kwargs)

    @is_non_distributed_check
    @inited_runtime_handler
    def run_server(self):
        """
        run server will run pserver main program with executor.

        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                if fleet.is_server():
                    fleet.init_server()

        """
        self._runtime_handle._run_server()

    @is_non_distributed_check
    @inited_runtime_handler
    def stop_worker(self):
        """
        stop `Communicator` and give training complete notice to parameter server.

        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                fleet.init_server()

        """
        self._runtime_handle._stop_worker()

    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names,
                             target_vars,
                             main_program=None,
                             export_for_deployment=True):
        """
        save inference model for inference.

        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                fleet.init_server()

        """

        self._runtime_handle._save_inference_model(
            executor, dirname, feeded_var_names, target_vars, main_program,
            export_for_deployment)

    def save_persistables(self, executor, dirname, main_program=None, mode=1):
        """

        saves all persistable tensors from :code:`main_program` to
        the folder :code:`dirname`. You can refer to

        The :code:`dirname` is used to specify the folder where persistable tensors
        are going to be saved. If you would like to save tensors in separate
        files, set :code:`filename` None.

        Args:
            executor(Executor): The executor to run for saving persistable tensors.
                                You can refer to :ref:`api_guide_executor_en` for
                                more details.

            dirname(str, optional): The saving directory path.
                                When you need to save the parameter to the memory, set it to None.
            main_program(Program, optional): The program whose persistbale tensors will
                                             be saved. Default: None.


        Returns:
            None

        Examples:

            .. code-block:: text

                import paddle
                paddle.enable_static()
                import paddle.distributed.fleet as fleet

                fleet.init()

                # build net
                # fleet.distributed_optimizer(...)

                exe = paddle.static.Executor(paddle.CPUPlace())
                fleet.save_persistables(exe, "dirname", paddle.static.default_main_program())

        """

        self._runtime_handle._save_persistables(executor, dirname, main_program,
                                                mode)

    def distributed_optimizer(self, optimizer, strategy=None):
        """
        Optimizer for distributed training.

        For the distributed training, this method would rebuild a new instance of DistributedOptimizer.
        Which has basic Optimizer function and special features for distributed training.

        Args:
            optimizer(Optimizer): The executor to run for init server.
            strategy(DistributedStrategy): Extra properties for distributed optimizer. 
                It is recommended to use DistributedStrategy in fleet.init(). The strategy
                here is for compatibility. If the strategy in fleet.distributed_optimizer() 
                is not None, then it will overwrite the DistributedStrategy in fleet.init(), 
                which will take effect in distributed training.

        Returns:
            Fleet: instance of fleet.

        Examples:

            .. code-block:: python

                import paddle
                import paddle.distributed.fleet as fleet
                fleet.init(is_collective=True)
                strategy = fleet.DistributedStrategy()
                optimizer = paddle.optimizer.SGD(learning_rate=0.001)
                optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)

        """
        self.user_defined_optimizer = optimizer

        if strategy is not None:
            warnings.warn(
                "It is recommended to use DistributedStrategy "
                "in fleet.init(). The strategy here is only for compatibility. "
                "If the strategy in fleet.distributed_optimizer() is "
                "not None, then it will overwrite the DistributedStrategy in fleet.init(), "
                "which will take effect in distributed training.")
            self._user_defined_strategy = copy.deepcopy(strategy)

        self._context = {}
        return self

    @dygraph_only
    def distributed_model(self, model):
        """
        Return distributed data parallel model (Only work in dygraph mode)

        Args:
            model (Layer): the user-defind model which inherits Layer.

        Returns:
            distributed data parallel model which inherits Layer.

        Examples:

            .. code-block:: python

                import paddle
                import paddle.nn as nn
                from paddle.distributed import fleet

                class LinearNet(nn.Layer):
                    def __init__(self):
                        super(LinearNet, self).__init__()
                        self._linear1 = nn.Linear(10, 10)
                        self._linear2 = nn.Linear(10, 1)

                    def forward(self, x):
                        return self._linear2(self._linear1(x))

                # 1. initialize fleet environment
                fleet.init(is_collective=True)

                # 2. create layer & optimizer
                layer = LinearNet()
                loss_fn = nn.MSELoss()
                adam = paddle.optimizer.Adam(
                    learning_rate=0.001, parameters=layer.parameters())

                # 3. get data_parallel model using fleet
                adam = fleet.distributed_optimizer(adam)
                dp_layer = fleet.distributed_model(layer)

                # 4. run layer
                inputs = paddle.randn([10, 10], 'float32')
                outputs = dp_layer(inputs)
                labels = paddle.randn([10, 1], 'float32')
                loss = loss_fn(outputs, labels)

                print("loss:", loss.numpy())

                loss.backward()

                adam.step()
                adam.clear_grad()


        """
        assert model is not None
        self.model = paddle.DataParallel(
            model,
            comm_buffer_size=self._user_defined_strategy.fuse_grad_size_in_MB,
            last_comm_buffer_size=self._user_defined_strategy.
            last_comm_group_size_MB)
        return self.model

    @dygraph_only
    def state_dict(self):
        """
        Get state dict information from optimizer.
        (Only work in dygraph mode)

        Returns: 
            state_dict(dict) : dict contains all the Tensor used by optimizer

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle
                from paddle.distributed import fleet

                fleet.init(is_collective=True)

                value = np.arange(26).reshape(2, 13).astype("float32")
                a = paddle.to_tensor(value)

                layer = paddle.nn.Linear(13, 5)
                adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

                adam = fleet.distributed_optimizer(adam)
                dp_layer = fleet.distributed_model(layer)
                state_dict = adam.state_dict()
        """
        # imitate target optimizer retrieval
        return self.user_defined_optimizer.state_dict()

    @dygraph_only
    def set_state_dict(self, state_dict):
        """
        Load optimizer state dict.
        (Only work in dygraph mode)

        Args: 
            state_dict(dict) : Dict contains all the Tensor needed by optimizer

        Returns:
            None

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle
                from paddle.distributed import fleet

                fleet.init(is_collective=True)

                value = np.arange(26).reshape(2, 13).astype("float32")
                a = paddle.to_tensor(value)

                layer = paddle.nn.Linear(13, 5)
                adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

                adam = fleet.distributed_optimizer(adam)
                dp_layer = fleet.distributed_model(layer)
                state_dict = adam.state_dict()
                paddle.save(state_dict, "paddle_dy")
                para_state_dict = paddle.load("paddle_dy")
                adam.set_state_dict(para_state_dict)
        """
        # imitate target optimizer retrieval
        return self.user_defined_optimizer.set_state_dict(state_dict)

    @dygraph_only
    def set_lr(self, value):
        """
        Set the value of the learning rate manually in the optimizer. 
        (Only work in dygraph mode)

        Args:
            value (float|Tensor): the value of learning rate

        Returns: 
            None 

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle
                from paddle.distributed import fleet

                fleet.init(is_collective=True)

                value = np.arange(26).reshape(2, 13).astype("float32")
                a = paddle.to_tensor(value)

                layer = paddle.nn.Linear(13, 5)
                adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

                adam = fleet.distributed_optimizer(adam)
                dp_layer = fleet.distributed_model(layer)

                lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
                for i in range(5):
                    adam.set_lr(lr_list[i])
                    lr = adam.get_lr()
                    print("current lr is {}".format(lr))
                # Print:
                #    current lr is 0.2
                #    current lr is 0.3
                #    current lr is 0.4
                #    current lr is 0.5
                #    current lr is 0.6
        """
        # imitate target optimizer retrieval
        return self.user_defined_optimizer.set_lr(value)

    @dygraph_only
    def get_lr(self):
        """
        Get current step learning rate.
        (Only work in dygraph mode)

        Returns:
            float: The learning rate of the current step.

        Examples:

            .. code-block:: python

                import numpy as np
                import paddle
                from paddle.distributed import fleet

                fleet.init(is_collective=True)

                value = np.arange(26).reshape(2, 13).astype("float32")
                a = paddle.to_tensor(value)

                layer = paddle.nn.Linear(13, 5)
                adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

                adam = fleet.distributed_optimizer(adam)
                dp_layer = fleet.distributed_model(layer)

                lr = adam.get_lr()
                print(lr) # 0.01
        """
        # imitate target optimizer retrieval
        return self.user_defined_optimizer.get_lr()

    @dygraph_only
    def step(self):
        """
        Execute the optimizer once.
        (Only work in dygraph mode)

        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle
                import paddle.nn as nn
                from paddle.distributed import fleet

                class LinearNet(nn.Layer):
                    def __init__(self):
                        super(LinearNet, self).__init__()
                        self._linear1 = nn.Linear(10, 10)
                        self._linear2 = nn.Linear(10, 1)

                    def forward(self, x):
                        return self._linear2(self._linear1(x))

                # 1. initialize fleet environment
                fleet.init(is_collective=True)

                # 2. create layer & optimizer
                layer = LinearNet()
                loss_fn = nn.MSELoss()
                adam = paddle.optimizer.Adam(
                    learning_rate=0.001, parameters=layer.parameters())

                # 3. get data_parallel model using fleet
                adam = fleet.distributed_optimizer(adam)
                dp_layer = fleet.distributed_model(layer)

                # 4. run layer
                inputs = paddle.randn([10, 10], 'float32')
                outputs = dp_layer(inputs)
                labels = paddle.randn([10, 1], 'float32')
                loss = loss_fn(outputs, labels)

                print("loss:", loss.numpy())

                loss.backward()

                adam.step()
                adam.clear_grad()


        """
        # imitate target optimizer retrieval
        return self.user_defined_optimizer.step()

    @dygraph_only
    def clear_grad(self):
        """
        Clear the gradients of all optimized parameters for model.
        (Only work in dygraph mode)

        Returns: 
            None

        Examples:

            .. code-block:: python

                import paddle
                import paddle.nn as nn
                from paddle.distributed import fleet

                class LinearNet(nn.Layer):
                    def __init__(self):
                        super(LinearNet, self).__init__()
                        self._linear1 = nn.Linear(10, 10)
                        self._linear2 = nn.Linear(10, 1)

                    def forward(self, x):
                        return self._linear2(self._linear1(x))

                # 1. initialize fleet environment
                fleet.init(is_collective=True)

                # 2. create layer & optimizer
                layer = LinearNet()
                loss_fn = nn.MSELoss()
                adam = paddle.optimizer.Adam(
                    learning_rate=0.001, parameters=layer.parameters())

                # 3. get data_parallel model using fleet
                adam = fleet.distributed_optimizer(adam)
                dp_layer = fleet.distributed_model(layer)

                # 4. run layer
                inputs = paddle.randn([10, 10], 'float32')
                outputs = dp_layer(inputs)
                labels = paddle.randn([10, 1], 'float32')
                loss = loss_fn(outputs, labels)

                print("loss:", loss.numpy())

                loss.backward()

                adam.step()
                adam.clear_grad()

        """
        # imitate target optimizer retrieval
        return self.user_defined_optimizer.clear_grad()

    def _final_strategy(self):
        if "valid_strategy" not in self._context:
            print(
                "WARNING: You may need to call minimize function before this function is called"
            )
            return {}
        else:
            return self._context["valid_strategy"]

    def _get_applied_meta_list(self):
        if "applied_meta_list" not in self._context:
            print(
                "WARNING: You may need to call minimize function before _get_applied_meta_list called"
            )
            return []
        else:
            return self._context["applied_meta_list"]

    def _get_applied_graph_list(self):
        if "applied_graph_list" not in self._context:
            print(
                "WARNING: You may need to call minimize function before _get_applied_graph_list called"
            )
            return []
        else:
            return self._context["applied_graph_list"]

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        Add distributed operations to minimize ``loss`` by updating ``parameter_list``.

        Args:
            loss (Tensor): A ``Tensor`` containing the value to minimize.
            startup_program (Program, optional): :ref:`api_fluid_Program` for
                initializing parameters in ``parameter_list``. The default value
                is None, at this time :ref:`api_fluid_default_startup_program` will be used.
            parameter_list (Iterable, optional): Iterable of ``Tensor`` or ``Tensor.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Tensor``  or ``Tensor.name`` that don't need
                to be updated. The default value is None.

        Returns:
            tuple: tuple (optimize_ops, params_grads), A list of operators appended
            by minimize and a list of (param, grad) tensor pairs, param is
            ``Parameter``, grad is the gradient value corresponding to the parameter.
            The returned tuple can be passed to ``fetch_list`` in ``Executor.run()`` to
            indicate program pruning. If so, the program will be pruned by ``feed`` and
            ``fetch_list`` before run, see details in ``Executor``.

        Examples:

            .. code-block:: python

                import paddle
                paddle.enable_static()
                import paddle.distributed.fleet as fleet
                import paddle.nn.functional as F

                hid_dim = 10
                label_dim = 2
                input_x = paddle.static.data(name='x', shape=[None, 13], dtype='float32')
                input_y = paddle.static.data(name='y', shape=[None, 1], dtype='int64')
                fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim, activation='tanh')
                fc_2 = paddle.static.nn.fc(x=fc_1, size=hid_dim, activation='tanh')
                prediction = paddle.static.nn.fc(x=[fc_2], size=label_dim, activation='softmax')
                cost = F.cross_entropy(input=prediction, label=input_y)
                avg_cost = paddle.mean(x=cost)

                fleet.init(is_collective=True)
                strategy = fleet.DistributedStrategy()
                optimizer = paddle.optimizer.SGD(learning_rate=0.001)
                optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
                optimizer.minimize(avg_cost)

                # for more examples, please reference https://github.com/PaddlePaddle/FleetX

        """
        context = {}
        context["user_defined_strategy"] = copy.deepcopy(
            self._user_defined_strategy)
        if paddle.fluid.framework.in_dygraph_mode():
            # imitate target optimizer retrieval
            target_opt = self.user_defined_optimizer
            self._context = context
            return target_opt.minimize(loss)

        # cache original feed forward program
        self.origin_main_program = loss.block.program
        context["origin_main_program"] = self.origin_main_program
        context["loss"] = loss
        if startup_program == None:
            self.origin_startup_program = \
                paddle.static.default_startup_program().clone(for_test=False)
            startup_program = paddle.static.default_startup_program()
        else:
            self.origin_startup_program = \
                startup_program.clone(for_test=False)

        context["origin_startup_program"] = startup_program
        context["role_maker"] = self._role_maker

        # compile time
        distributed_optimizer_list = \
            MetaOptimizerFactory()._get_valid_meta_optimizers(
                self.user_defined_optimizer)

        context["user_defined_strategy"] = copy.deepcopy(
            self._user_defined_strategy)
        copy_user_defined_strategy = copy.deepcopy(self._user_defined_strategy)

        # trigger the auto-parallel in very strict condition
        # strategy = DistributedStrategy()
        # strategy.auto = True
        # optimizer = paddle.optimizer.SGD(learning_rate=0.1)
        # optimizer = fleet.distributed_optimizer(optimizer, strategy)
        if copy_user_defined_strategy._is_strict_auto():
            # turn on all the strategy for each optimizer
            for opt in distributed_optimizer_list:
                opt._enable_strategy(copy_user_defined_strategy, context)

        valid_optimizer_list = []
        valid_graph_optimizer_list = []
        can_not_apply_optimizer_list = []
        # recall meta optimizers for ranking
        for opt in distributed_optimizer_list:
            opt._set_basic_info(loss, self._role_maker,
                                self.user_defined_optimizer,
                                copy_user_defined_strategy)
            if opt._can_apply() and not opt._is_graph_out():
                valid_optimizer_list.append(opt)
            elif opt._can_apply() and opt._is_graph_out():
                valid_graph_optimizer_list.append(opt)
            else:
                can_not_apply_optimizer_list.append(opt)
        # combine recalled meta optimizers to be a valid meta optimizer
        meta_optimizer, graph_optimizer = \
            self.strategy_compiler.generate_optimizer(
                loss, self._role_maker, self.user_defined_optimizer,
                copy_user_defined_strategy, valid_optimizer_list,
                valid_graph_optimizer_list)

        valid_strategy = self.strategy_compiler._get_valid_strategy(
            copy_user_defined_strategy, can_not_apply_optimizer_list)

        context["valid_strategy"] = copy.deepcopy(valid_strategy)

        applied_meta_list = self.strategy_compiler._get_applied_meta_list()
        applied_graph_list = self.strategy_compiler._get_applied_graph_list()

        context['applied_meta_list'] = applied_meta_list
        context['applied_graph_list'] = applied_graph_list

        self._context = context

        self.valid_strategy = valid_strategy
        self.valid_strategy._enable_env()

        optimize_ops = []
        params_grads = []

        if self._role_maker._is_non_distributed() and not self._is_collective:
            if self._runtime_handle is None:
                self._runtime_handle = RuntimeFactory()._create_runtime(context)

            compiled_program = compiler.CompiledProgram(
                self.origin_main_program).with_data_parallel(
                    loss_name=loss.name, share_vars_from=None)
            loss.block.program._graph = compiled_program
            return self.user_defined_optimizer.minimize(
                loss, startup_program, parameter_list, no_grad_set=no_grad_set)

        if meta_optimizer:
            optimize_ops, params_grads = meta_optimizer.minimize(
                loss, startup_program, parameter_list, no_grad_set=no_grad_set)

            default_program = paddle.static.default_main_program()

            if id(default_program) != id(loss.block.program):
                paddle.fluid.framework.switch_main_program(loss.block.program)

        else:
            optimize_ops, params_grads = self.user_defined_optimizer.minimize(
                loss, startup_program, parameter_list, no_grad_set=no_grad_set)

        context["program_optimize_ops"] = optimize_ops
        context["program_params_grads"] = params_grads

        if graph_optimizer:
            optimize_ops, params_grads = graph_optimizer.minimize(
                loss, startup_program, parameter_list, no_grad_set=no_grad_set)
            # since we do not encourage users to use graph operations
            # if a graph optimizer takes effect, mostly
            # optimizers_ops and params_grads are None
            # i.e. users can not modify current computation graph anymore
            context["graph_optimize_ops"] = optimize_ops
            context["graph_optimize_grads"] = params_grads

        if self._runtime_handle is None:
            self._runtime_handle = RuntimeFactory()._create_runtime(context)

        import paddle.distributed.fleet as fleet
        fleet.util._set_strategy(context["valid_strategy"])

        return optimize_ops, params_grads
