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

from __future__ import print_function
import os
import sys

# The legacy core need to be removed before "import core",
# in case of users installing paddlepadde without -U option
core_suffix = 'so'
if os.name == 'nt':
    core_suffix = 'pyd'

legacy_core = os.path.abspath(os.path.dirname(
    __file__)) + os.sep + 'core.' + core_suffix
if os.path.exists(legacy_core):
    sys.stderr.write('Deleting legacy file ' + legacy_core + '\n')
    try:
        os.remove(legacy_core)
    except Exception as e:
        raise e

# import all class inside framework into fluid module
from . import framework
from .framework import *
# import all class inside executor into fluid module
from . import executor
from .executor import *

from . import data_feed_desc
from .data_feed_desc import *

from . import dataset
from .dataset import *

from .data import *

from . import trainer_desc

from . import io
from . import evaluator
from . import initializer
from .initializer import set_global_initializer
from . import layers
from . import dygraph
from . import contrib
from . import nets
from . import optimizer
from . import backward
from .backward import gradients
from . import regularizer
from . import average
from . import metrics
from . import transpiler
from . import incubate
from .input import embedding, one_hot
from . import distribute_lookup_table
from .param_attr import ParamAttr, WeightNormParamAttr
from .data_feeder import DataFeeder
from .core import LoDTensor, LoDTensorArray, CPUPlace, XPUPlace, CUDAPlace, CUDAPinnedPlace, Scope, _Scope
from .incubate import fleet
from .incubate import data_generator
from .transpiler import DistributeTranspiler, \
    memory_optimize, release_memory, DistributeTranspilerConfig
from .lod_tensor import create_lod_tensor, create_random_int_lodtensor
from . import clip
from . import profiler
from . import unique_name
from . import parallel_executor
from .parallel_executor import *
from . import compiler
from .compiler import *
from paddle.fluid.layers.math_op_patch import monkey_patch_variable
from . import install_check
from .dygraph.nn import *
from .dygraph.layers import *
from .dygraph.base import enable_dygraph, disable_dygraph
from .io import save, load, load_program_state, set_program_state
from .dygraph.checkpoint import save_dygraph, load_dygraph
from .dygraph.varbase_patch_methods import monkey_patch_varbase
from . import generator
from .core import _cuda_synchronize

Tensor = LoDTensor
enable_imperative = enable_dygraph
disable_imperative = disable_dygraph

__all__ = framework.__all__ + executor.__all__ + \
    trainer_desc.__all__ + transpiler.__all__ + \
    parallel_executor.__all__ + lod_tensor.__all__ + \
    data_feed_desc.__all__ + compiler.__all__ + backward.__all__  + generator.__all__ + [
        'io',
        'initializer',
        'embedding',
        'one_hot',
        'layers',
        'contrib',
        'data',
        'dygraph',
        'enable_dygraph',
        'disable_dygraph',
        'enable_imperative',
        'disable_imperative',
        'transpiler',
        'nets',
        'optimizer',
        'learning_rate_decay',
        'backward',
        'regularizer',
        'LoDTensor',
        'LoDTensorArray',
        'CPUPlace',
        'XPUPlace',
        'CUDAPlace',
        'CUDAPinnedPlace',
        'Tensor',
        'ParamAttr',
        'WeightNormParamAttr',
        'DataFeeder',
        'clip',
        'profiler',
        'unique_name',
        'Scope',
        'install_check',
        'save',
        'load',
        'VarBase',
        '_cuda_synchronize'
    ]


def __bootstrap__():
    """
    Enable reading gflags from environment variables.

    Returns:
        None
    """
    import sys
    import os
    import platform
    from . import core

    in_test = 'unittest' in sys.modules

    try:
        num_threads = int(os.getenv('OMP_NUM_THREADS', '1'))
    except ValueError:
        num_threads = 1

    if num_threads > 1:
        print(
            'WARNING: OMP_NUM_THREADS set to {0}, not 1. The computation '
            'speed will not be optimized if you use data parallel. It will '
            'fail if this PaddlePaddle binary is compiled with OpenBlas since'
            ' OpenBlas does not support multi-threads.'.format(num_threads),
            file=sys.stderr)
        print('PLEASE USE OMP_NUM_THREADS WISELY.', file=sys.stderr)

    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    sysstr = platform.system()
    read_env_flags = [
        'check_nan_inf',
        'fast_check_nan_inf',
        'benchmark',
        'eager_delete_scope',
        'fraction_of_cpu_memory_to_use',
        'initial_cpu_memory_in_mb',
        'init_allocated_mem',
        'paddle_num_threads',
        'dist_threadpool_size',
        'eager_delete_tensor_gb',
        'fast_eager_deletion_mode',
        'memory_fraction_of_eager_deletion',
        'allocator_strategy',
        'reader_queue_speed_test_mode',
        'print_sub_graph_dir',
        'pe_profile_fname',
        'inner_op_parallelism',
        'enable_parallel_graph',
        'fuse_parameter_groups_size',
        'multiple_of_cupti_buffer_size',
        'fuse_parameter_memory_size',
        'tracer_profile_fname',
        'dygraph_debug',
        'use_system_allocator',
        'enable_unused_var_check',
        'free_idle_chunk',
        'free_when_no_cache_hit',
        'call_stack_level',
        'sort_sum_gradient',
        'max_inplace_grad_add',
    ]
    if 'Darwin' not in sysstr:
        read_env_flags.append('use_pinned_memory')

    if os.name != 'nt':
        read_env_flags.append('cpu_deterministic')

    if core.is_compiled_with_mkldnn():
        read_env_flags.append('use_mkldnn')
        read_env_flags.append('tracer_mkldnn_ops_on')
        read_env_flags.append('tracer_mkldnn_ops_off')

    if core.is_compiled_with_dist():
        #env for rpc
        read_env_flags.append('rpc_deadline')
        read_env_flags.append('rpc_retry_times')
        read_env_flags.append('rpc_server_profile_path')
        read_env_flags.append('enable_rpc_profiler')
        read_env_flags.append('rpc_send_thread_num')
        read_env_flags.append('rpc_get_thread_num')
        read_env_flags.append('rpc_prefetch_thread_num')
        read_env_flags.append('rpc_disable_reuse_port')
        read_env_flags.append('rpc_retry_bind_port')

        read_env_flags.append('worker_update_interval_secs')

        if core.is_compiled_with_brpc():
            read_env_flags.append('max_body_size')
            #set brpc max body size
            os.environ['FLAGS_max_body_size'] = "2147483647"

    if core.is_compiled_with_cuda():
        read_env_flags += [
            'fraction_of_gpu_memory_to_use',
            'initial_gpu_memory_in_mb',
            'reallocate_gpu_memory_in_mb',
            'cudnn_deterministic',
            'enable_cublas_tensor_op_math',
            'conv_workspace_size_limit',
            'cudnn_exhaustive_search',
            'selected_gpus',
            'sync_nccl_allreduce',
            'cudnn_batchnorm_spatial_persistent',
            'gpu_allocator_retry_time',
            'local_exe_sub_scope_limit',
            'gpu_memory_limit_mb',
        ]
    core.init_gflags(["--tryfromenv=" + ",".join(read_env_flags)])
    core.init_glog(sys.argv[0])
    # don't init_p2p when in unittest to save time.
    core.init_devices(not in_test)


# TODO(panyx0718): Avoid doing complex initialization logic in __init__.py.
# Consider paddle.init(args) or paddle.main(args)
monkey_patch_variable()
__bootstrap__()
monkey_patch_varbase()
