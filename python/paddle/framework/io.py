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

from __future__ import print_function

import os
import collections
import pickle
import six
import warnings
import sys

import paddle

# deprecated module import
from paddle import fluid
from paddle.fluid import core
from paddle.fluid.io import _unpack_saved_dict, _pack_loaded_dict
from paddle.fluid.framework import Variable, _varbase_creator, _dygraph_tracer
from paddle.fluid.dygraph.jit import _SaveLoadConfig
from paddle.fluid.dygraph.io import _construct_program_holders, _construct_params_and_buffers
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX, INFER_PARAMS_INFO_SUFFIX

__all__ = [
    'save',
    'load',
]


def _build_saved_state_dict(state_dict):
    save_dict = {}
    name_table = {}
    for key, value in state_dict.items():
        if isinstance(value, (Variable, core.VarBase)):
            save_dict[key] = value.numpy()
            name_table[key] = value.name
        else:
            save_dict[key] = value
    save_dict["StructuredToParameterName@@"] = name_table

    return save_dict


def _load_state_dict_from_save_inference_model(model_path, config):
    # 1. load program desc & construct _ProgramHolder
    programs = _construct_program_holders(model_path, config.model_filename)

    # 2. load layer parameters & buffers
    with fluid.dygraph.guard():
        persistable_var_dict = _construct_params_and_buffers(
            model_path, programs, config.params_filename, append_suffix=False)

        # 3. construct state_dict
        load_param_dict = dict()
        for var_name in persistable_var_dict:
            load_param_dict[var_name] = persistable_var_dict[var_name].numpy()

        # if *.info exists, we can recover structured_name
        var_info_filename = str(config.params_filename) + ".info"
        var_info_path = os.path.join(model_path, var_info_filename)
        if os.path.exists(var_info_path):
            with open(var_info_path, 'rb') as f:
                extra_var_info = pickle.load(f)
            structured_para_dict = dict()
            for var_name in load_param_dict:
                structured_name = extra_var_info[var_name].get(
                    'structured_name', None)
                assert structured_name is not None, "Cannot find saved variable (%s)'s structured name in saved model." % var_name
                structured_para_dict[structured_name] = load_param_dict[
                    var_name]
            load_param_dict = structured_para_dict

    return load_param_dict


def _load_state_dict_from_save_params(model_path):
    # Try to load all the files in the directory in VarBase format, 
    # the file name is used as the name of VarBase
    load_var_list = []

    # 1. load file names
    var_name_list = []
    for root, _, files in os.walk(model_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            tmp_var_name = os.path.relpath(file_path, model_path)
            var_name = tmp_var_name.replace("\\", "/")
            var_name_list.append(var_name)

    # 2. create and load VarBase
    with fluid.dygraph.guard():
        for name in var_name_list:
            new_var = _varbase_creator(name=name, persistable=True)
            _dygraph_tracer().trace_op(
                type='load',
                inputs={},
                outputs={'Out': new_var},
                attrs={'file_path': os.path.join(model_path, name)})
            load_var_list.append(new_var)

    # 3. construct state_dict
    load_param_dict = dict()
    for var in load_var_list:
        load_param_dict[var.name] = var.numpy()

    return load_param_dict


# NOTE(chenweihang): [ Handling of use cases of API paddle.load ]
# `paddle.load` may be used to load saved results of:
# 1. Expected cases:
#   - need [full filename] when loading
#       - paddle.save
#       - paddle.static.save
#       - paddle.fluid.save_dygraph
#   - need [prefix] when loading [compatible for paddle 2.x]
#       - paddle.jit.save
#       - paddle.static.save_inference_model
#   - need [directory] when loading [compatible for paddle 1.x]
#       - paddle.fluid.io.save_inference_model
#       - paddle.fluid.io.save_params/save_persistable
# 2. Error cases:
#   - no error case
def _build_load_path_and_config(path, config):
    # NOTE(chenweihang): If both [prefix save format] and [directory save format] exist,
    # raise error, avoid confusing behavior
    prefix_format_path = path + INFER_MODEL_SUFFIX
    prefix_format_exist = os.path.exists(prefix_format_path)
    directory_format_exist = os.path.isdir(path)
    if prefix_format_exist and directory_format_exist:
        raise ValueError(
            "The %s.pdmodel and %s directory exist at the same time, "
            "don't know which one to load, please make sure that the specified target "
            "of ``path`` is unique." % (path, path))
    elif not prefix_format_exist and not directory_format_exist:
        error_msg = "The ``path`` (%s) to load model not exists."
        # if current path is a prefix, and the path.pdparams or path.pdopt
        # is exist, users may want use `paddle.load` load the result of 
        # `fluid.save_dygraph`, we raise error here for users
        params_file_path = path + ".pdparams"
        opti_file_path = path + ".pdopt"
        if os.path.exists(params_file_path) or os.path.exists(opti_file_path):
            error_msg += " If you want to load the results saved by `fluid.save_dygraph`, " \
                "please specify the full file name, not just the file name prefix. For " \
                "example, it should be written as `paddle.load('model.pdparams')` instead of " \
                "`paddle.load('model')`."
        raise ValueError(error_msg % path)
    else:
        if prefix_format_exist:
            file_prefix = os.path.basename(path)
            model_path = os.path.dirname(path)
            if config.model_filename is not None:
                warnings.warn(
                    "When loading the result saved with the "
                    "specified file prefix, the ``model_filename`` config does "
                    "not take effect.")
            config.model_filename = file_prefix + INFER_MODEL_SUFFIX
            if config.params_filename is not None:
                warnings.warn(
                    "When loading the result saved with the "
                    "specified file prefix, the ``params_filename`` config does "
                    "not take effect.")
            config.params_filename = file_prefix + INFER_PARAMS_SUFFIX
        else:
            # Compatible with the old save_inference_model format
            model_path = path

    return model_path, config


def _parse_load_config(configs):
    supported_configs = ['model_filename', 'params_filename', 'keep_name_table']

    # input check
    for key in configs:
        if key not in supported_configs:
            raise ValueError(
                "The additional config (%s) of `paddle.load` is not supported."
                % key)

    # construct inner config
    inner_config = _SaveLoadConfig()
    inner_config.model_filename = configs.get('model_filename', None)
    inner_config.params_filename = configs.get('params_filename', None)
    inner_config.keep_name_table = configs.get('keep_name_table', None)

    return inner_config


def save(obj, path):
    '''
    Save an object to the specified path.
    
    .. note::
        Now only supports save ``state_dict`` of Layer or Optimizer.

    .. note::
        Different from ``paddle.jit.save``, since the save result of ``paddle.save`` is a single file, 
        there is no need to distinguish multiple saved files by adding a suffix. The argument ``path`` 
        of ``paddle.save`` will be directly used as the saved file name instead of a prefix. 
        In order to unify the saved file name format, we recommend using the paddle standard suffix:
        1. for ``Layer.state_dict`` , recommend to use ``.pdparams`` ; 
        2. for ``Optimizer.state_dict`` , recommend to use ``.pdopt`` . 
        For specific examples, please refer to API code examples.
    
    Args:
        obj(Object) : The object to be saved.
        path(str) : The path of the object to be saved. 
          If saved in the current directory, the input path string will be used as the file name. 

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle

            emb = paddle.nn.Embedding(10, 10)
            layer_state_dict = emb.state_dict()
            paddle.save(layer_state_dict, "emb.pdparams")
            scheduler = paddle.optimizer.lr.NoamDecay(	
                d_model=0.01, warmup_steps=100, verbose=True)
            adam = paddle.optimizer.Adam(
                learning_rate=scheduler,
                parameters=emb.parameters())
            opt_state_dict = adam.state_dict()
            paddle.save(opt_state_dict, "adam.pdopt")
    '''

    # 1. input check
    if not isinstance(obj, dict):
        raise NotImplementedError(
            "Now only supports save state_dict of Layer or Optimizer, "
            "expect dict, but received %s." % type(obj))

    if len(obj) == 0:
        warnings.warn("The input state dict is empty, no need to save.")

    filename = os.path.basename(path)
    if filename == "":
        raise ValueError("The input path MUST be format of dirname/filename "
                         "[dirname\\filename in Windows system], but received "
                         "filename is empty string.")

    # 2. save object
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    # TODO(chenweihang): supports save other object
    saved_obj = _build_saved_state_dict(obj)
    saved_obj = _unpack_saved_dict(saved_obj)

    # When value of dict is lager than 4GB ,there is a Bug on 'MAC python3.5/6'
    if sys.platform == 'darwin' and sys.version_info.major == 3 and (
            sys.version_info.minor == 5 or sys.version_info.minor == 6):
        pickle_bytes = pickle.dumps(saved_obj, protocol=2)
        with open(path, 'wb') as f:
            max_bytes = 2**30
            for i in range(0, len(pickle_bytes), max_bytes):
                f.write(pickle_bytes[i:i + max_bytes])
    else:
        with open(path, 'wb') as f:
            pickle.dump(saved_obj, f, protocol=2)


def load(path, **configs):
    '''
    Load an object can be used in paddle from specified path.

    .. note::
        Now only supports load ``state_dict`` of Layer or Optimizer.

    .. note::
        In order to use the model parameters saved by paddle more efficiently, 
        ``paddle.load`` supports loading ``state_dict`` of Layer from the result of 
        other save APIs except ``paddle.save`` , but the argument ``path`` format is 
        different:
        1. loading from ``paddle.static.save`` or ``paddle.Model().save(training=True)`` ,  
        ``path`` needs to be a complete file name, such as ``model.pdparams`` or 
        ``model.pdopt`` ; 
        2. loading from ``paddle.jit.save`` or ``paddle.static.save_inference_model`` 
        or ``paddle.Model().save(training=False)`` , ``path`` need to be a file prefix, 
        such as ``model/mnist``, and ``paddle.load`` will get information from 
        ``mnist.pdmodel`` and ``mnist.pdiparams`` ;
        3. loading from paddle 1.x APIs ``paddle.fluid.io.save_inference_model`` or 
        ``paddle.fluid.io.save_params/save_persistables`` , ``path`` need to be a 
        directory, such as ``model`` and model is a directory.

    .. note::
        If you load ``state_dict`` from the saved result of static mode API such as 
        ``paddle.static.save`` or ``paddle.static.save_inference_model`` , 
        the structured variable name in dynamic mode will cannot be restored. 
        You need to set the argument ``use_structured_name=False`` when using 
        ``Layer.set_state_dict`` later.

    Args:
        path(str) : The path to load the target object. Generally, the path is the target 
            file path. When loading state_dict from the saved result of the API used to save 
            the inference model, the path may be a file prefix or directory.
        **configs (dict, optional): other load configuration options for compatibility. We do not 
            recommend using these configurations, they may be removed in the future. If not necessary, 
            DO NOT use them. Default None.
            The following options are currently supported:
            (1) model_filename (str): The inference model file name of the paddle 1.x 
            ``save_inference_model`` save format. Default file name is :code:`__model__` . 
            (2) params_filename (str): The persistable variables file name of the paddle 1.x 
            ``save_inference_model`` save format. No default file name, save variables separately 
            by default.

    Returns:
        Object(Object): a target object can be used in paddle

    Examples:
        .. code-block:: python

            import paddle

            emb = paddle.nn.Embedding(10, 10)
            layer_state_dict = emb.state_dict()
            paddle.save(layer_state_dict, "emb.pdparams")
            scheduler = paddle.optimizer.lr.NoamDecay(	
                d_model=0.01, warmup_steps=100, verbose=True)
            adam = paddle.optimizer.Adam(
                learning_rate=scheduler,
                parameters=emb.parameters())
            opt_state_dict = adam.state_dict()
            paddle.save(opt_state_dict, "adam.pdopt")

            load_layer_state_dict = paddle.load("emb.pdparams")
            load_opt_state_dict = paddle.load("adam.pdopt")
    '''
    load_result = None
    config = _parse_load_config(configs)

    if os.path.isfile(path):
        # we think path is file means this file is created by paddle.save
        with open(path, 'rb') as f:
            load_result = pickle.load(f) if six.PY2 else pickle.load(
                f, encoding='latin1')
        load_result = _pack_loaded_dict(load_result)
        if not config.keep_name_table and "StructuredToParameterName@@" in load_result:
            del load_result["StructuredToParameterName@@"]
    else:
        # file prefix and directory are compatible cases
        model_path, config = _build_load_path_and_config(path, config)
        # check whether model file exists
        if config.model_filename is None:
            model_filename = '__model__'
        else:
            model_filename = config.model_filename
        model_file_path = os.path.join(model_path, model_filename)

        if os.path.exists(model_file_path):
            # Load state dict by `jit.save/io.save_inference_model` save format
            # NOTE(chenweihang): [ Compatibility of save_inference_model save format ]
            # The model saved by `save_inference_model` does not completely correspond to 
            # the information required by the `state_dict` under the dygraph. 
            # `save_inference_model` not save structured name, we need to remind 
            # the user to configure the `use_structured_name` argument when `set_state_dict`
            # NOTE(chenweihang): `jit.save` doesn't save optimizer state 
            load_result = _load_state_dict_from_save_inference_model(model_path,
                                                                     config)
        else:
            # load state dict by `io.save_params/persistables` save format
            # TODO(chenweihang): [ Now only supports loading parameters seperately ]
            # If users save all parameters as one file, the [ variable.name -> variable ]
            # mapping info will lost, so users need to give variable list, but users build 
            # variable list in dygraph mode is difficult, we recommend users to use
            # paddle.static.load_program_state in this case
            load_result = _load_state_dict_from_save_params(model_path)

    return load_result
