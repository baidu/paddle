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

import paddle
from paddle.distributed.fleet.proto import distributed_strategy_pb2
from paddle.fluid.framework import Variable, set_flags, core
from paddle.fluid.wrapped_decorator import wrap_decorator
import google.protobuf.text_format
import google.protobuf

__all__ = ["DistributedStrategy"]

non_auto_func_called = True


def __non_auto_func_called__(func):
    def __impl__(*args, **kwargs):
        global non_auto_func_called
        non_auto_func_called = False
        return func(*args, **kwargs)

    return __impl__


is_strict_auto = wrap_decorator(__non_auto_func_called__)


def get_msg_dict(msg):
    res_dict = {}
    fields = msg.DESCRIPTOR.fields
    for f in fields:
        res_dict[f.name] = getattr(msg, f.name)
    return res_dict


def assign_configs_value(msg, config):
    fields = msg.DESCRIPTOR.fields
    for key in config:
        for f in fields:
            if key == f.name:
                if f.label == 3:
                    getattr(msg, f.name).extend(config[f.name])
                elif f.label == 1 or f.label == 2:
                    setattr(msg, f.name, config[f.name])


def check_configs_key(msg, config, field_name):
    key_list = msg.DESCRIPTOR.fields_by_name.keys()
    for key in config:
        assert key in key_list, "key:{} not in {}".format(key, field_name)


class DistributedJobInfo(object):
    """
    DistributedJobInfo will serialize all distributed training information
    Just for inner use: 1) debug 2) replicate experiments
    """

    def __init__(self):
        self.job_info = distributed_strategy_pb2.DistributedJobInfo()

    def _set_worker_num(self, worker_num):
        self.job_info.worker_num = worker_num

    def _set_server_num(self, server_num):
        self.job_info.server_num = server_num

    def _set_worker_ips(self, worker_ips):
        self.job_info.worker_ips.extend(worker_ips)

    def _set_server_endpoints(self, server_endpoints):
        self.job_info.server_endpoints.extend(server_endpoints)

    def _set_origin_startup(self, origin_startup_prog):
        self.job_info.origin_startup = str(origin_startup_prog)

    def _set_origin_main(self, origin_main_prog):
        self.job_info.origin_main = str(origin_main_prog)

    def _distributed_main(self, distributed_main_prog):
        self.job_info.distributed_main = str(distributed_main_prog)

    def _optimizer_name(self, optimizer_name):
        self.job_info.optimizer_name = optimizer_name

    def _set_distributed_strategy(self, dist_strategy):
        self.job_info.strategy = dist_strategy


class DistributedStrategy(object):
    __lock_attr = False

    def __init__(self):
        """
        DistributedStrategy is the main configuration entry for distributed training of Paddle.
        All of the distributed training configurations can be configured in DistributedStrategy,
        such as automatic mixed precision (AMP), Layer-wise Adaptive Rate Scaling (LARS), 
        asynchronous update parameter server(ASGD), etc.
        
        DistributedStrategy can be serialized into protobuf file or deserialized from protobuf file

        Users who run local training usually configure BuildStrategy and ExecutionStrategy, and 
        DistributedStrategy supports configurations from BuildStrategy and ExecutionStrategy

        """
        self.strategy = distributed_strategy_pb2.DistributedStrategy()
        self.__lock_attr = True

    def __setattr__(self, key, value):
        if self.__lock_attr and not hasattr(self, key):
            raise TypeError("%s is not a attribute of %s" %
                            (key, self.__class__.__name__))
        object.__setattr__(self, key, value)

    def save_to_prototxt(self, output):
        """
        Serialize current DistributedStrategy to string and save to output file

        Examples:
          .. code-block:: python
        
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.dgc = True
            strategy.recompute = True
            strategy.recompute_configs = {"checkpoints": ["x"]}
            strategy.save_to_prototxt("dist_strategy.prototxt")
        """
        with open(output, "w") as fout:
            fout.write(str(self.strategy))

    def load_from_prototxt(self, pb_file):
        """
        Load from prototxt file for DistributedStrategy initialization

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.load_from_prototxt("dist_strategy.prototxt")
        """
        with open(pb_file, 'r') as f:
            self.strategy = google.protobuf.text_format.Merge(
                str(f.read()), self.strategy)

    @property
    def execution_strategy(self):
        """
        Configure ExecutionStrategy for DistributedStrategy

        Examples:
          .. code-block:: python

            import paddle
            exe_strategy = paddle.fluid.ExecutionStrategy()
            exe_strategy.num_threads = 10
            exe_strategy.num_iteration_per_drop_scope = 10
            exe_strategy.num_iteration_per_run = 10

            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.execution_strategy = exe_strategy
        """
        execution_strategy = paddle.fluid.ExecutionStrategy()
        fields = self.strategy.execution_strategy.DESCRIPTOR.fields
        for f in fields:
            setattr(execution_strategy, f.name,
                    getattr(self.strategy.execution_strategy, f.name))
        return execution_strategy

    @execution_strategy.setter
    @is_strict_auto
    def execution_strategy(self, strategy):
        fields = self.strategy.execution_strategy.DESCRIPTOR.fields
        for f in fields:
            setattr(self.strategy.execution_strategy, f.name,
                    getattr(strategy, f.name))

    @property
    def build_strategy(self):
        """
        Configure BuildStrategy for DistributedStrategy
        Note that the properties of BuildStrategy are valid in DistributedStrategy
        only if the property is non-distributed strategy.

        Examples:
          .. code-block:: python

            import paddle
            build_strategy = paddle.fluid.BuildStrategy()
            build_strategy.enable_sequential_execution = True
            build_strategy.fuse_elewise_add_act_ops = True
            build_strategy.fuse_bn_act_ops = True
            build_strategy.enable_auto_fusion = True
            build_strategy.fuse_relu_depthwise_conv = True
            build_strategy.fuse_broadcast_ops = True
            build_strategy.fuse_all_optimizer_ops = True
            build_strategy.enable_inplace = True
            
            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.build_strategy = build_strategy
        """

        build_strategy = paddle.fluid.BuildStrategy()
        fields = self.strategy.build_strategy.DESCRIPTOR.fields
        for f in fields:
            setattr(build_strategy, f.name,
                    getattr(self.strategy.build_strategy, f.name))
        return build_strategy

    @build_strategy.setter
    @is_strict_auto
    def build_strategy(self, strategy):
        fields = self.strategy.build_strategy.DESCRIPTOR.fields
        for f in fields:
            if f.label == 1 or f.label == 2:  # optional and required field
                setattr(self.strategy.build_strategy, f.name,
                        getattr(strategy, f.name))
            elif f.label == 3:  # repeated field
                getattr(self.strategy.build_strategy,
                        f.name).extend(getattr(strategy, f.name))

    @property
    def a_sync(self):
        """
        Indicating whether we are using asynchronous stocastic gradient descent updates
        for training. This property is valid when we are using parameter server training, 
        which is implied by setting approperate RoleMaker
        Default value: True

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            role_maker = fleet.PaddleCloudRoleMaker()
            fleet.init(role_maker)

            strategy = fleet.DistributedStrategy()
            strategy.a_sync = True  # by default this is True
            
            # code block for defining loss and local optimizer
            # sgd = fleet.distributed_optimizer(optimizer, strategy)
        """
        return self.strategy.a_sync

    @a_sync.setter
    @is_strict_auto
    def a_sync(self, flag):
        if isinstance(flag, bool):
            self.strategy.a_sync = flag
            self.a_sync_configs = {"k_steps": 0}
        else:
            raise ValueError(
                "The type of `flag` is invalid, expected type is bool, but received %s".
                format(type(flag)))

    @property
    def a_sync_configs(self):
        """
        Set a_sync update configurations. In general, asynchronous parameter server
        training has serveral configurable settings that can be configured through
        a dict.

        **Notes**:
            k_step(int): number of local optimization updates before communication

            max_merge_var_num(int): maximum number of merged gradients before communication

            send_queue_size(int): a buffer size of worker communication

            independent_recv_thread(bool): if we are using independent recv thread for communication

            thread_pool_size(int): number of thread pool

            send_wait_times(int): waiting time for sending gradients

            runtime_split_send_recv(bool): if we are using Tensor split for send and recv during runtime

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            role_maker = fleet.PaddleCloudRoleMaker()
            fleet.init(role_maker)

            strategy = fleet.DistributedStrategy()
            strategy.a_sync = True  # by default this is True
            configs = {"k_steps": 1024, "send_queue_size": 32}
            strategy.a_sync_configs = configs

            # code block for defining loss and local optimizer
            # sgd = fleet.distributed_optimizer(optimizer, strategy)

        """
        return get_msg_dict(self.strategy.a_sync_configs)

    @a_sync_configs.setter
    @is_strict_auto
    def a_sync_configs(self, configs):
        check_configs_key(self.strategy.a_sync_configs, configs,
                          "a_sync_configs")
        assign_configs_value(self.strategy.a_sync_configs, configs)

    @property
    def amp(self):
        """
        Indicating whether we are using automatic mixed precision training
        Default Value: False

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.amp = True # by default this is false

        """
        return self.strategy.amp

    @amp.setter
    @is_strict_auto
    def amp(self, flag):
        if isinstance(flag, bool):
            self.strategy.amp = flag
        else:
            print("WARNING: amp should have value of bool type")

    @property
    def amp_configs(self):
        """
        Set automatic mixed precision training configurations. In general, amp has serveral configurable
        settings that can be configured through a dict.

        **Notes**:
            init_loss_scaling(float): The initial loss scaling factor. Default 32768.

            use_dynamic_loss_scaling(bool): Whether to use dynamic loss scaling. Default True.

            incr_every_n_steps(int): Increases loss scaling every n consecutive steps with finite gradients. Default 1000.

            decr_every_n_nan_or_inf(int): Decreases loss scaling every n accumulated steps with nan or inf gradients. Default 2.

            incr_ratio(float): The multiplier to use when increasing the loss scaling. Default 2.0.

            decr_ratio(float): The less-than-one-multiplier to use when decreasing the loss scaling. Default 0.5.

            custom_white_list(list[str]): Users' custom white list which always execution fp16.

            custom_black_list(list[str]): Users' custom black list which forbidden execution fp16.

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.amp = True
            strategy.amp_configs = {
                "init_loss_scaling": 32768,
                "custom_white_list": ['conv2d']}
        """
        return get_msg_dict(self.strategy.amp_configs)

    @amp_configs.setter
    @is_strict_auto
    def amp_configs(self, configs):
        check_configs_key(self.strategy.amp_configs, configs, "amp_configs")
        assign_configs_value(self.strategy.amp_configs, configs)

    @property
    def recompute(self):
        """
        Indicating whether we are using forward recomputation for memory optimization
        Default value: False

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.recompute = True
            # suppose x and y are names of checkpoint tensors for recomputation
            strategy.recompute_configs = {"checkpoints": ["x", "y"]}
        """
        return self.strategy.recompute

    @property
    def sync_nccl_allreduce(self):
        """
        Indicating whether we are using synchronized all reduce in each communication thread
        We note that system overhead is usually lower when sync_nccl_allreduce = True

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.sync_nccl_allreduce = True
        """
        return self.strategy.sync_nccl_allreduce

    @sync_nccl_allreduce.setter
    @is_strict_auto
    def sync_nccl_allreduce(self, flag):
        if isinstance(flag, bool):
            self.strategy.sync_nccl_allreduce = flag
        else:
            print("WARNING: sync_nccl_allreduce should have value of bool type")

    @property
    def use_hierarchical_allreduce(self):
        """
        Indicating whether we are using hierarchical allreduce in collective communication
        Hierarchical allreduce often does allreduce within a certain node group and then do
        allreduce among the leaders of each group

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.use_hierarchical_allreduce = True
        """
        return self.strategy.use_hierarchical_allreduce

    @use_hierarchical_allreduce.setter
    @is_strict_auto
    def use_hierarchical_allreduce(self, flag):
        if isinstance(flag, bool):
            self.strategy.use_hierarchical_allreduce = flag
        else:
            print(
                "WARNING: use_hierarchical_allreduce should have value of bool type"
            )

    @property
    def hierarchical_allreduce_inter_nranks(self):
        """
        Number of ranks for low level node groups in hierarchical allreduce
        Default value: number of GPU cards on each single GPU machine

        Example:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.hierarchical_allreduce_inter_nranks = 8
        """
        return self.strategy.hierarchical_allreduce_inter_nranks

    @hierarchical_allreduce_inter_nranks.setter
    @is_strict_auto
    def hierarchical_allreduce_inter_nranks(self, value):
        if isinstance(value, int):
            self.strategy.hierarchical_allreduce_inter_nranks = value
        else:
            print(
                "WARNING: hierarchical_allreduce_inter_nranks should have value of int type"
            )

    @property
    def sync_batch_norm(self):
        """
        Indicating whether we are using sync_batch_norm to do synchronous batch normalization among all training nodes.
        
        Default value: False

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.sync_batch_norm = True
        """

        return self.strategy.sync_batch_norm

    @sync_batch_norm.setter
    @is_strict_auto
    def sync_batch_norm(self, flag):
        if isinstance(flag, bool):
            self.strategy.sync_batch_norm = flag
        else:
            print("WARNING: sync_batch_norm should have value of bool type")

    @property
    def fuse_all_reduce_ops(self):
        """
        Indicating whether we are using fuse_all_reduce_ops for gradient fusion during backward phase of training
        Default value: True

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.fuse_all_reduce_ops = False
        """
        return self.strategy.fuse_all_reduce_ops

    @fuse_all_reduce_ops.setter
    @is_strict_auto
    def fuse_all_reduce_ops(self, flag):
        if isinstance(flag, bool):
            self.strategy.fuse_all_reduce_ops = flag
        else:
            print("WARNING: fuse_all_reduce_ops should have value of bool type")

    @property
    def fuse_grad_size_in_MB(self):
        """
        Specifying the size of gradient to fuse in Mega-Bytes

        Default value: 32

        Examples:
          .. code-block:: python
        
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.fuse_grad_size_in_MB = 50
        """
        return self.strategy.fuse_grad_size_in_MB

    @fuse_grad_size_in_MB.setter
    @is_strict_auto
    def fuse_grad_size_in_MB(self, value):
        if isinstance(value, int):
            self.strategy.fuse_grad_size_in_MB = value
        else:
            print("WARNING: fuse_grad_size_in_MB should have value of int type")

    @property
    def _fuse_grad_size_in_TFLOPS(self):
        return self.strategy.fuse_grad_size_in_TFLOPS

    @_fuse_grad_size_in_TFLOPS.setter
    @is_strict_auto
    def _fuse_grad_size_in_TFLOPS(self, value):
        if isinstance(value, float):
            self.strategy.fuse_grad_size_in_TFLOPS = value
        else:
            print(
                "WARNING: fuse_grad_size_in_TFLOPS should have value of float type"
            )

    @property
    def nccl_comm_num(self):
        """
        Specifying the number of NCCL communicator

        Default value: 1

        Examples:
          .. code-block:: python
        
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.nccl_comm_num = 2
        """

        return self.strategy.nccl_comm_num

    @nccl_comm_num.setter
    @is_strict_auto
    def nccl_comm_num(self, value):
        if isinstance(value, int):
            self.strategy.nccl_comm_num = value
        else:
            print("WARNING: nccl_comm_num should have value of int type")

    @recompute.setter
    @is_strict_auto
    def recompute(self, flag):
        if isinstance(flag, bool):
            self.strategy.recompute = flag
        else:
            print("WARNING: recompute should have value of bool type")

    @property
    def recompute_configs(self):
        """
        Set recompute configurations. In general, the recompute strategy of current
        implementation should have some manually assign checkpoints

        Examples:
          .. code-block:: python
        
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.recompute = True
            strategy.recompute_configs = {"checkpoints": ["x", "y"]}

        """
        return get_msg_dict(self.strategy.recompute_configs)

    @recompute_configs.setter
    @is_strict_auto
    def recompute_configs(self, configs):
        check_configs_key(self.strategy.recompute_configs, configs,
                          "checkpoint_configs")
        assign_configs_value(self.strategy.recompute_configs, configs)

    @property
    def pipeline(self):
        """
        Indicating whether we are using pipeline parallelism for distributed training.
        Current implementation mainly focus on single GPU machine pipeline parallelism and
        data parallelism across GPU machine. The pipeline information is indicated through
        device_guard information in user-defined program.

        Examples:
          .. code-block:: python
        
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.pipeline = True

        """
        return self.strategy.pipeline

    @pipeline.setter
    @is_strict_auto
    def pipeline(self, flag):
        if isinstance(flag, bool):
            self.strategy.pipeline = flag
        else:
            print("WARNING: pipeline should have value of bool type")

    @property
    def pipeline_configs(self):
        """
        Set pipeline parallelism configurations. In pipeline parallelism,
        different parts of neural networks are running on different GPUS.
        There are Tensor queue buffer between each pair of neighborhood GPUS 
        that are responsible for synchronizing hidden Tensor results between
        GPUs. Pipeline parallelism consists of serveral producer-consumer style
        hardware pairs, such as GPU-GPU, CPU-GPU, GPU-XPU. The best way to speedup
        pipeline parallelism is to make the size of Tensor in Tensor queue smaller, 
        so that we will have a faster producer for downstream consumers.

        **Notes**:
            **Detailed arguments for pipeline_configs**

            **micro_batch**: the number of small batches in each user defined batch

        Examples:
          .. code-block:: python
        
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.pipeline = True
            strategy.pipeline_configs = {"micro_batch": 12}

        """

        return get_msg_dict(self.strategy.pipeline_configs)

    @pipeline_configs.setter
    @is_strict_auto
    def pipeline_configs(self, configs):
        check_configs_key(self.strategy.pipeline_configs, configs,
                          "pipeline_configs")
        assign_configs_value(self.strategy.pipeline_configs, configs)

    @property
    def localsgd(self):
        """
        Indicating whether we are using Local SGD training. Default Value: False
        For more details, please refer to
        `Don't Use Large Mini-Batches, Use Local SGD <https://arxiv.org/pdf/1808.07217.pdf>`_.


        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.localsgd = True # by default this is false

        """
        return self.strategy.localsgd

    @localsgd.setter
    @is_strict_auto
    def localsgd(self, flag):
        if isinstance(flag, bool):
            self.strategy.localsgd = flag
        else:
            print("WARNING: localsgd should have value of bool type")

    @property
    def localsgd_configs(self):
        """
        Set LocalSGD training configurations. LocalSGD has a configurable
        setting that can be configured through a dict.

        **Notes**:
            k_steps(int) The local steps for training before parameter synchronization. Default 1.
            begin_step(int) The step of begining training by localsgd. Default 1.

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.localsgd = True
            strategy.localsgd_configs = {"k_steps": 4,
                                         "begin_step": 30}
        """

        return get_msg_dict(self.strategy.localsgd_configs)

    @localsgd_configs.setter
    @is_strict_auto
    def localsgd_configs(self, configs):
        check_configs_key(self.strategy.localsgd_configs, configs,
                          "localsgd_configs")
        assign_configs_value(self.strategy.localsgd_configs, configs)

    @property
    def adaptive_localsgd(self):
        """
        Indicating whether we are using Adaptive Local SGD training. Default Value: False
        For more details, please refer to `Adaptive Communication Strategies to Achieve 
        the Best Error-Runtime Trade-off in Local-Update SGD <https://arxiv.org/pdf/1810.08313.pdf>`_.


        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.adaptive_localsgd = True # by default this is false

        """
        return self.strategy.adaptive_localsgd

    @adaptive_localsgd.setter
    @is_strict_auto
    def adaptive_localsgd(self, flag):
        if isinstance(flag, bool):
            self.strategy.adaptive_localsgd = flag
        else:
            print("WARNING: adaptive_localsgd should have value of bool type")

    @property
    def adaptive_localsgd_configs(self):
        """
        Set AdaptiveLocalSGD training configurations. AdaptiveLocalSGD has a configurable
        setting that can be configured through a dict.

        **Notes**:
            init_k_steps(int) The initial steps for training before adaptive localsgd.
                              Then, the adaptive localsgd method will modify init_k_steps automatically.
                              Default 1.
            begin_step(int) The step of begining training by adaptive localsgd. Default 1.

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.adaptive_localsgd = True
            strategy.adaptive_localsgd_configs = {"init_k_steps": 1,
                                                  "begin_step": 30}
        """

        return get_msg_dict(self.strategy.adaptive_localsgd_configs)

    @adaptive_localsgd_configs.setter
    @is_strict_auto
    def adaptive_localsgd_configs(self, configs):
        check_configs_key(self.strategy.adaptive_localsgd_configs, configs,
                          "adaptive_localsgd_configs")
        assign_configs_value(self.strategy.adaptive_localsgd_configs, configs)

    @property
    def dgc(self):
        """
        Indicating whether we are using Deep Gradient Compression training. For more details, please refer to
        [Deep Gradient Compression](https://arxiv.org/abs/1712.01887).

        Default Value: False

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.dgc = True # by default this is false

        """
        return self.strategy.dgc

    @dgc.setter
    @is_strict_auto
    def dgc(self, flag):
        if isinstance(flag, bool):
            self.strategy.dgc = flag
        else:
            print("WARNING: dgc should have value of bool type")

    @property
    def dgc_configs(self):
        """
        Set Deep Gradient Compression training configurations. In general, dgc has serveral configurable
        settings that can be configured through a dict.

        **Notes**:
            rampup_begin_step(int): The beginning step from which gradient compression is implemented. Default 0.

            rampup_step(int): Time steps used in sparsity warm-up periods. Default is 1. \
                    For example, if the sparsity is [0.75, 0.9375, 0.984375, 0.996, 0.999], and the rampup_step is 100, \
                    it will use 0.75 at 0~19 steps, and 0.9375 at 20~39 steps, and so on. And when reach sparsity array \
                    ends, it will use 0.999 then and after.

            sparsity(list[float]): Get top important element from gradient tensor, the ratio is (1 - sparsity). \
                    Default is [0.999]. For example, if the sparsity is [0.99, 0.999], the top [1%, 0.1%] important \
                    element will be transmitted.

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.dgc = True
            strategy.dgc_configs = {"rampup_begin_step": 1252}
        """
        return get_msg_dict(self.strategy.dgc_configs)

    @dgc_configs.setter
    @is_strict_auto
    def dgc_configs(self, configs):
        check_configs_key(self.strategy.dgc_configs, configs, "dgc_configs")
        assign_configs_value(self.strategy.dgc_configs, configs)

    @property
    def fp16_allreduce(self):
        """
        Indicating whether we are using fp16 gradient allreduce training
        Default Value: False

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.fp16_allreduce = True # by default this is false

        """
        return self.strategy.fp16_allreduce

    @fp16_allreduce.setter
    @is_strict_auto
    def fp16_allreduce(self, flag):
        if not isinstance(flag, bool):
            raise TypeError('fp16_allreduce must be value of bool type')
        self.strategy.fp16_allreduce = flag

    @property
    def gradient_merge(self):
        """
        Gradient Merge, also called as Gradient Accumulation,
        is a strategy for large batch training. With this strategy,
        model parameter will not be updated until user-defined steps.
        For each step, the forward network and the backward network
        will run to calculate the gradient of model parameters.
        For every k step, the optimization network will run,
        applying a specific optimization method (such as SGD, Adam)
        to model parameters.

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.gradient_merge = True
            strategy.gradient_merge_configs = {"k_steps": 4, "avg": True}
        """
        return self.strategy.gradient_merge

    @gradient_merge.setter
    @is_strict_auto
    def gradient_merge(self, flag):
        if isinstance(flag, bool):
            self.strategy.gradient_merge = flag
        else:
            print("WARNING: gradient_merge should have value of bool type")

    @property
    def gradient_merge_configs(self):
        """
        the key-value configs of distribute_strategy

        **Note**:
            k_steps(int): the update period of the parameters.

            avg(bool): whether to average the gradients of each mini-batch, the default value is `True`

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.gradient_merge = True
            strategy.gradient_merge_configs = {"k_steps": 4, "avg": True}
        """
        return get_msg_dict(self.strategy.gradient_merge_configs)

    @gradient_merge_configs.setter
    @is_strict_auto
    def gradient_merge_configs(self, configs):
        check_configs_key(self.strategy.gradient_merge_configs, configs,
                          "gradient_configs")
        assign_configs_value(self.strategy.gradient_merge_configs, configs)

    @property
    def lars(self):
        """
        Set lars configurations. lars is used to deal with the convergence problems when the global 
        batch size is larger than 8k.  For more details, please refer to 
        [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888).

        Default Value: False

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.lars = True # by default this is false
        """
        return self.strategy.lars

    @lars.setter
    @is_strict_auto
    def lars(self, flag):
        if isinstance(flag, bool):
            self.strategy.lars = flag
        else:
            print("WARNING: lars should have value of bool type")

    @property
    def lars_configs(self):
        """
        Set Lars training configurations.

        **Notes**:
        **lars_coeff (float)**: trust ratio in lars formula.
        **lars_weight_decay** (float): weight decay coefficient in lars formula.
        **epsilon (float)**: argument is used to avoid potential devision-by-zero 
        when compute the local lr; 
        **exclude_from_weight_decay ([string])**: is a list of name strings of layers which
        will be exclude from weight decay in lars formula.

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.lars = True
            strategy.lars_configs = {
                        "lars_coeff": 0.01,
                        "lars_weight_decay": 0.0005,
                        "epsilon": 0,
                        "exclude_from_weight_decay": ['batch_norm', '.b_0']
                    }
        """
        return get_msg_dict(self.strategy.lars_configs)

    @lars_configs.setter
    @is_strict_auto
    def lars_configs(self, configs):
        check_configs_key(self.strategy.lars_configs, configs, "lars_configs")
        assign_configs_value(self.strategy.lars_configs, configs)

    @property
    def lamb(self):
        """
        Set lamb configurations. lamb is used to deal with the convergence problems for large 
        batch size training, specially for attention-related model like BERT. For more details, 
        please refer to 
        [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962).

        Default Value: False
        
        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.lamb = True # by default this is false
        """

        return self.strategy.lamb

    @lamb.setter
    @is_strict_auto
    def lamb(self, flag):
        if isinstance(flag, bool):
            self.strategy.lamb = flag
        else:
            print("WARNING: lamb should have value of bool type")

    @property
    def lamb_configs(self):
        """
        Set Lars training configurations.

        **Notes**:
        **lamb_weight_decay** (float): weight decay coefficient in lamb formula.
        **exclude_from_weight_decay ([string])**: is a list of name strings of layers which
        will be exclude from weight decay in lamb formula.

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.lamb = True
            strategy.lamb_configs = {
                    'lamb_weight_decay': 0.01,
                    'exclude_from_weight_decay': [],
                }
        """
        return get_msg_dict(self.strategy.lamb_configs)

    @lamb_configs.setter
    @is_strict_auto
    def lamb_configs(self, configs):
        check_configs_key(self.strategy.lamb_configs, configs, "lamb_configs")
        assign_configs_value(self.strategy.lamb_configs, configs)

    @property
    def elastic(self):
        """
        Indicating whether we want to do current distributed training on clusters with elastic resources.
        Currently, this is configuration is not valid.
        """
        return self.strategy.elastic

    @elastic.setter
    @is_strict_auto
    def elastic(self, flag):
        if isinstance(flag, bool):
            self.strategy.elastic = flag
        else:
            print("WARNING: elastic should have value of bool type")

    @property
    def auto(self):
        """
        Indicating whether we are using auto-parallel configuration
        This feature is currently an experimental feature. Currently, 
        auto-parallelism can be used only when a user does not set any other
        strategy configs except auto. For details, please reference the following
        code example
        Default Value: False

        Examples:
          .. code-block:: python

            import paddle
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.auto = True

            optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        """
        return self.strategy.auto

    @auto.setter
    def auto(self, flag):
        if isinstance(flag, bool):
            self.strategy.auto = flag
        else:
            print("WARNING: auto should have value of bool type")

    @property
    def cudnn_exhaustive_search(self):
        """
        Indicating whether to use exhaustive search method to choose convolution algorithms.
        Exhaustive search attempts all cuDNN algorithms to choose the fastest algorithm.
        This method is time-consuming, the choosed algorithm will be cached for the given layer specifications.
        Once the layer specifications (like batch size, feature map size) are changed, it will search again.
        Default Value: True

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.cudnn_exhaustive_search = False

            optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        """
        return self.strategy.cudnn_exhaustive_search

    @cudnn_exhaustive_search.setter
    @is_strict_auto
    def cudnn_exhaustive_search(self, flag):
        if isinstance(flag, bool):
            self.strategy.cudnn_exhaustive_search = flag
        else:
            print(
                "WARNING: cudnn_exhaustive_search should have value of bool type"
            )

    @property
    def conv_workspace_size_limit(self):
        """
        The workspace limit size in MB unit for choosing cuDNN convolution algorithms.
        The inner funciton of cuDNN obtain the fastest suited algorithm that fits within this memory limit.
        Usually, large workspace size may lead to choose faster algorithms,
        but significant increasing memory workspace. Users need to trade-off between memory and speed.
        Default Value: 4000

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.conv_workspace_size_limit = 1024

            optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        
        """
        return self.strategy.conv_workspace_size_limit

    @conv_workspace_size_limit.setter
    @is_strict_auto
    def conv_workspace_size_limit(self, value):
        if isinstance(value, int):
            self.strategy.conv_workspace_size_limit = value
        else:
            print(
                "WARNING: conv_workspace_size_limit should have value of int type"
            )

    @property
    def cudnn_batchnorm_spatial_persistent(self):
        """
        Indicates whether to use the mode CUDNN_BATCHNORM_SPATIAL_PERSISTENT function in batchnorm.
        This is only useful in cudnn.
        Default Value: True

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.cudnn_batchnorm_spatial_persistent = True

            optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            optimizer = fleet.distributed_optimizer(optimizer, strategy)

        """
        return self.strategy.cudnn_batchnorm_spatial_persistent

    @cudnn_batchnorm_spatial_persistent.setter
    @is_strict_auto
    def cudnn_batchnorm_spatial_persistent(self, flag):
        if isinstance(flag, bool):
            self.strategy.cudnn_batchnorm_spatial_persistent = flag
        else:
            print(
                "WARNING: cudnn_batchnorm_spatial_persistent should have value of bool type"
            )

    def _enable_env(self):
        strategy = self.strategy
        keys = [
            "FLAGS_cudnn_batchnorm_spatial_persistent",
            "FLAGS_conv_workspace_size_limit",
            "FLAGS_cudnn_exhaustive_search",
            "FLAGS_sync_nccl_allreduce",
            "FLAGS_fuse_parameter_memory_size",
            "FLAGS_fuse_parameter_groups_size",
        ]
        values = [
            bool(strategy.cudnn_batchnorm_spatial_persistent),
            int(strategy.conv_workspace_size_limit),
            bool(strategy.cudnn_exhaustive_search),
            bool(strategy.sync_nccl_allreduce),
            int(strategy.fuse_grad_size_in_MB),
            int(strategy.fuse_grad_size_in_TFLOPS),
        ]

        for i, key in enumerate(keys):
            if core.globals().is_public(key):
                core.globals()[key] = values[i]

    def _is_strict_auto(self):
        global non_auto_func_called
        if self.strategy.auto and non_auto_func_called:
            return True
        return False

    def __repr__(self):
        spacing = 2
        max_k = 38
        max_v = 38

        length = max_k + max_v + spacing

        h1_format = "    " + "|{{:^{}s}}|\n".format(length)
        h2_format = "    " + "|{{:>{}s}}{}{{:^{}s}}|\n".format(max_k, " " *
                                                               spacing, max_v)

        border = "    +" + "".join(["="] * length) + "+"
        line = "    +" + "".join(["-"] * length) + "+"

        draws = border + "\n"
        draws += h1_format.format("")
        draws += h1_format.format("DistributedStrategy Overview")
        draws += h1_format.format("")

        fields = self.strategy.DESCRIPTOR.fields
        str_res = ""

        env_draws = line + "\n"
        for f in fields:
            if "build_strategy" in f.name or "execution_strategy" in f.name:
                continue
            if "_configs" in f.name:
                continue
            else:
                if isinstance(getattr(self.strategy, f.name), bool):
                    if hasattr(self.strategy, f.name + "_configs"):
                        if getattr(self.strategy, f.name):
                            draws += border + "\n"
                            draws += h1_format.format(
                                "{}=True <-> {}_configs".format(f.name, f.name))
                            draws += line + "\n"
                            my_configs = getattr(self.strategy,
                                                 f.name + "_configs")
                            config_fields = my_configs.DESCRIPTOR.fields
                            for ff in config_fields:
                                if isinstance(
                                        getattr(my_configs, ff.name),
                                        google.protobuf.pyext._message.
                                        RepeatedScalarContainer):
                                    values = getattr(my_configs, ff.name)
                                    for i, v in enumerate(values):
                                        if i == 0:
                                            draws += h2_format.format(ff.name,
                                                                      str(v))
                                        else:
                                            draws += h2_format.format("",
                                                                      str(v))
                                else:
                                    draws += h2_format.format(
                                        ff.name,
                                        str(getattr(my_configs, ff.name)))
                    else:
                        env_draws += h2_format.format(
                            f.name, str(getattr(self.strategy, f.name)))
                else:
                    env_draws += h2_format.format(
                        f.name, str(getattr(self.strategy, f.name)))

        result_res = draws + border + "\n" + h1_format.format(
            "Environment Flags, Communication Flags")
        result_res += env_draws

        build_strategy_str = border + "\n"
        build_strategy_str += h1_format.format("Build Strategy")
        build_strategy_str += line + "\n"

        fields = self.strategy.build_strategy.DESCRIPTOR.fields
        for f in fields:
            build_strategy_str += h2_format.format(
                f.name, str(getattr(self.strategy.build_strategy, f.name)))
        build_strategy_str += border + "\n"

        execution_strategy_str = h1_format.format("Execution Strategy")
        execution_strategy_str += line + "\n"

        fields = self.strategy.execution_strategy.DESCRIPTOR.fields
        for f in fields:
            execution_strategy_str += h2_format.format(
                f.name, str(getattr(self.strategy.execution_strategy, f.name)))
        execution_strategy_str += border + "\n"

        result_res += build_strategy_str + execution_strategy_str
        return result_res
