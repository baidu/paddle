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
import six
import pickle
import numpy as np

from paddle import compat as cpt
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid import backward
from paddle.fluid.dygraph import layers
from paddle.fluid.layers import nn
from paddle.fluid.dygraph.base import switch_to_static_graph
from paddle.fluid.data_feeder import check_type

__all__ = ['TranslatedLayer']

VARIABLE_FILENAME = "__variables__"
EXTRA_VAR_INFO_FILENAME = "__variables.info__"


def _load_program_desc(model_file_path):
    # 1. parse program desc
    with open(model_file_path, "rb") as f:
        program_desc_str = f.read()

    program_desc = core.ProgramDesc(program_desc_str)
    if not core._is_program_version_supported(program_desc._version()):
        raise ValueError("Unsupported program version: %d\n" %
                         program_desc._version())
    return program_desc


def _is_persistable(var_desc):
    if var_desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
            var_desc.type() == core.VarDesc.VarType.FETCH_LIST or \
            var_desc.type() == core.VarDesc.VarType.READER or \
            var_desc.type() == core.VarDesc.VarType.RAW:
        return False
    return var_desc.persistable()


def _is_parameter(persistable_var_desc, program_desc):
    # 1. firstly, param should be input of op
    input_ops = []  # op can be repeated
    for block_idx in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(block_idx)
        for op_idx in six.moves.range(block.op_size()):
            op = block.op(op_idx)
            # NOTE: parameter is the input of a certain op
            if persistable_var_desc.name() in op.input_arg_names():
                input_ops.append(op)
    # 2. secondly, param should not be output of op or be same op's output
    for block_idx in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(block_idx)
        for op_idx in six.moves.range(block.op_size()):
            op = block.op(op_idx)
            if persistable_var_desc.name() in op.output_arg_names():
                # such as batch_norm_op
                if op in input_ops:
                    continue
                else:
                    return False
    return True


def _get_persistable_vars(program_desc):
    persistable_vars = []
    for i in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(i)
        persistable_vars.extend(list(filter(_is_persistable, block.all_vars())))
    return persistable_vars


def _get_persistable_var_names(program_desc):
    """
    Get all persistable variable names in ProgramDesc.
    """
    var_names = []
    persistable_vars = _get_persistable_vars(program_desc)
    for var in persistable_vars:
        var_names.append(var.name())
    return var_names


def _get_all_var_names(program_desc):
    all_var_names = set()
    for i in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(i)
        for var in block.all_vars():
            all_var_names.add(var.name())
    return all_var_names


def _append_loaded_suffix(name):
    """
    Append loaded suffix to the given variable name
    e.g. x ==> x@LOADED
    """
    suffix = core.loaded_var_suffix()
    name = cpt.to_text(name)
    if suffix not in name:
        name = name + suffix
    return name


def _remove_loaded_suffix(name):
    """
    Remove loaded suffix to the given variable name
    e.g. x@LOADED ==> x
    """
    suffix = core.loaded_var_suffix()
    name = cpt.to_text(name)
    return name.replace(suffix, '')


def _append_loaded_suffix_to_var(program_desc):
    persistable_vars = _get_persistable_vars(program_desc)
    for var_desc in persistable_vars:
        old_name = var_desc.name()
        new_name = _append_loaded_suffix(var_desc.name())
        var_desc.set_name(new_name)
        for block_idx in six.moves.range(program_desc.num_blocks()):
            block = program_desc.block(block_idx)
            for op_idx in six.moves.range(block.op_size()):
                op = block.op(op_idx)
                op._rename_input(old_name, new_name)
                op._rename_output(old_name, new_name)


@switch_to_static_graph
def _build_program_by_desc(program_desc):
    prog = framework.Program()
    with framework.program_guard(prog, framework.Program()):
        prog.desc = program_desc
        prog.blocks = [
            framework.Block(prog, i)
            for i in six.moves.range(prog.desc.num_blocks())
        ]
        prog._sync_with_cpp()
    return prog


def _change_is_test_status(program_desc, is_test):
    # change all `is_test` attributes
    for i in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(i)
        for j in six.moves.range(block.op_size()):
            op = block.op(j)
            if op.has_attr('is_test'):
                op._set_attr('is_test', is_test)


class _ProgramHolder(object):
    """
    Holds the execution information of a Program.

    _ProgramHolder is the execution unit of TranslatedLayer, 
    if TranslatedLayer contains multiple _ProgramHolder, 
    it can execute multiple methods

    _ProgramHolder is an internal concept.
    """

    def __init__(self, program_desc):
        super(_ProgramHolder, self).__init__()

        # input, output, persistable var info
        self._input_names = []
        self._persistable_names = []
        self._output_descs = []

        # execution scope
        self._inner_scope = core.Scope()

        # forward program
        self._infer_program_desc = self._preprocess(program_desc)
        # forward + backward program
        self._train_program_desc = self._append_backward_desc(
            self._infer_program_desc)

    @property
    def infer_program(self):
        return self._infer_program_desc

    @property
    def train_program(self):
        return self._train_program_desc

    @property
    def input_names(self):
        return self._input_names

    @property
    def output_decs(self):
        return self._output_descs

    @property
    def persistable_names(self):
        return self._persistable_names

    @property
    def scope(self):
        return self._inner_scope

    def _preprocess(self, program_desc):
        # 1. Prune original program
        # remove feed, fetch and scale-1 op, remove op_callstack attr
        ops_to_remove = []
        root_block = program_desc.block(0)
        for i in six.moves.range(root_block.op_size()):
            op = root_block.op(i)
            if op.type() == 'feed':
                ops_to_remove.append(i)
                feed_var_name = cpt.to_bytes(op.input('X')[0])
                root_block._remove_var(feed_var_name)
                self._input_names.append(cpt.to_bytes(op.output('Out')[0]))
            elif op.type() == 'scale' and op.output('Out')[0].startswith(
                    'save_infer_model/scale_'):
                ops_to_remove.append(i)
                out_var_name = cpt.to_bytes(op.output('Out')[0])
                root_block._remove_var(out_var_name)
                self._output_descs.append(
                    root_block.find_var(cpt.to_bytes(op.input('X')[0])))
            elif op.type() == 'fetch':
                ops_to_remove.append(i)
                fetch_var_name = cpt.to_bytes(op.output('Out')[0])
                root_block._remove_var(fetch_var_name)
                # NOTE: some old pre-train models have no extra scale_op
                if not op.input('X')[0].startswith('save_infer_model/scale_'):
                    self._output_descs.append(
                        root_block.find_var(cpt.to_bytes(op.input('X')[0])))
            else:
                if op.has_attr("op_callstack"):
                    op.remove_attr("op_callstack")

        for op_idx in reversed(ops_to_remove):
            root_block._remove_op(op_idx, op_idx + 1)

        # 2. Input processing, reverse feed vars
        self._input_names.reverse()

        # 3. Output processing, add scale for outputs
        tmp_program = _build_program_by_desc(program_desc)
        # NOTE: [why need append scale for outputs]
        # When dealing with some more complex pre-training models, there 
        # will be situations where the pre-training model has multiple 
        # fetch outputs. In the scenario of multiple fetch outputs, 
        # there is a special case where multiple outputs of the model 
        # may be on the same branch. According to the user's subsequent 
        # use, multiple outputs may be associated with multiple branches.
        # These subsequent operations are added in TranslatedLayer is 
        # agnostic during initialization, which results in subsequent 
        # gradient accumulation operations that are required on the 
        # output node in the middle of the branch will not be performed, 
        # resulting in error, details see pull request:
        # [https://github.com/PaddlePaddle/Paddle/pull/24627]
        self._append_scale_to_output(tmp_program)

        # 4. Persistable vars processing
        # - append @LOADED suffix to persistable vars
        # NOTE: [why need to append suffix to persistable vars]
        # Dygraph and static graph mode use the same naming mechanism. 
        # If users want to load the model fine-tune, it is possible 
        # to add the existing Layer in the loaded model to enhance 
        # the network. For example, the original saved model has linear, 
        # and later after loading, a new linear is added. At this time, 
        # there will be a problem of duplicate names, so here is unified 
        # to add the LOADED suffix to the parameters of the model loaded
        # during training. And in order to avoid multiple @LOADED suffix
        # are appended to variable name, we only append @LOADED suffix to
        # the variable that not contains @LOADED suffix.
        _append_loaded_suffix_to_var(program_desc)
        # - get persistable var
        self._persistable_names = _get_persistable_var_names(program_desc)

        return program_desc

    @switch_to_static_graph
    def _append_scale_to_output(self, program):
        # 1. append scale & save var
        scale_output_vars = []
        with framework.program_guard(program):
            for i, out in enumerate(self._output_descs):
                var = program.global_block().var(out.name())
                var = nn.scale(
                    var, 1., name="translated_layer/scale_{}".format(i))
                scale_output_vars.append(var)
        # 2. update output names & descs
        for i, var in enumerate(scale_output_vars):
            self._output_descs[i] = var.desc

    @switch_to_static_graph
    def _append_backward_desc(self, infer_program_desc):
        program_desc_copy = core.ProgramDesc(infer_program_desc)

        # 1. set all `is_test` attributes to False
        _change_is_test_status(program_desc_copy, False)

        # 2. prepare program and related var
        # NOTE: To reuse backward interfaces, build Program firstly.
        # Originally, there is no need to build a program, but need to almost
        # rewrite a series of methods for append_backward for program_desc. 
        # Therefore, in order to reuse the method of backward.py, build the program here.
        program = _build_program_by_desc(program_desc_copy)

        targets = []
        for out in self._output_descs:
            targets.append(program.global_block().var(out.name()))

        # 3. append backward
        backward.gradients(targets=targets, inputs=[])
        return program.desc


# [ TranslatedLayer : Run program in imperative mode ]
# 
# DESIGN IDEA: using an special operator `RunProgram`, execute program inside operator.
#
# Op's Inputs:
#   - the input variable of the user feed
#   - the necessary parameters of the network
# Op's Outputs:
#   - the output variable of fetch
# 
# This op receives a complete program desc, internally creates scope
# and executor, executes this program. Key points:
#
# 1. Data Sharing: 
#   The varBase of the dynamic graph is not in the scope, so before the op
#   executes the program internally, create persistent variables with the
#   same name as feed, parameters, and fetch in the scope, and share the
#   LoDTensor of the op input.
# 
# 2. Forward and Backward Separation:
#   Because the dynamic graph op performs the forward and backward separately,
#   in the forward op RunProgram, we only execute the forward part of whole program,
#   and in the backward op RunProgramGrad, we execute the backward part of program.
#   We can not separate the program into forward and backward part, which will 
#   make some control flow execution logic wrong.


# NOTE: [compatible] deal with model saved by save_inference_model,
# which need get var info from program desc
def _load_persistable_vars_by_program(model_path,
                                      program_holder,
                                      params_filename=None):
    # make sure the path has been checked
    persistable_vars = _get_persistable_vars(program_holder.infer_program)
    load_var_dict = {}
    for each_var in persistable_vars:
        orig_each_name = _remove_loaded_suffix(each_var.name())
        if _is_parameter(each_var, program_holder.infer_program):
            # create output varbase
            new_var = framework.ParamBase(
                shape=each_var.shape(),
                dtype=each_var.dtype(),
                name=each_var.name(),
                type=each_var.type(),
                persistable=True)
        else:
            new_var = framework._varbase_creator(
                type=each_var.type(),
                name=each_var.name(),
                shpae=each_var.shape(),
                dtype=each_var.dtype(),
                persistable=True)
        if params_filename is None:
            framework._dygraph_tracer().trace_op(
                type='load',
                inputs={},
                outputs={'Out': new_var},
                attrs={'file_path': os.path.join(model_path, orig_each_name)})
        new_var.stop_gradient = False
        load_var_dict[each_var.name()] = new_var

    if params_filename is not None:
        load_var_list = []
        for name in sorted(load_var_dict.keys()):
            load_var_list.append(load_var_dict[name])

        framework._dygraph_tracer().trace_op(
            type='load_combine',
            inputs={},
            outputs={'Out': load_var_list},
            attrs={'file_path': os.path.join(model_path, params_filename)})

        for each_var in persistable_vars:
            if not _is_parameter(each_var, program_holder.infer_program):
                continue
            param = load_var_dict[each_var.name()]
            param.stop_gradient = False

    # NOTE: [Recovery stop gradient information based on the program]
    # After loading the model, the stop_gradient information 
    # of the original variable is lost, but if a parameter does not
    # have a corresponding @GRAD variable in the backward program,
    # it can be said that it is also stop_gradient
    all_var_names = _get_all_var_names(program_holder.train_program)
    for var_name in load_var_dict:
        grad_var_name = var_name + core.grad_var_suffix()
        if grad_var_name not in all_var_names:
            load_var_dict[var_name].stop_gradient = True

    return load_var_dict


def _load_persistable_vars(model_path,
                           var_info_path,
                           separate_params=False,
                           params_filename=None):
    # 1. load extra var info
    with open(var_info_path, 'rb') as f:
        extra_var_info = pickle.load(f)

    # 2. construct var dict
    load_var_dict = dict()
    load_var_list = []
    # NOTE: some var may not be Parameter
    for name in sorted(extra_var_info):
        # append suffix, see [why need to append suffix to persistable vars]
        new_name = _append_loaded_suffix(name)
        # create output varbase
        if extra_var_info[name].get('trainable', None) is not None:
            # use default shape and dtype
            new_var = framework.ParamBase(
                shape=[1],  # only to pass check, this shape is not meaningful
                dtype=core.VarDesc.VarType.FP32,
                name=new_name,
                persistable=True)
        else:
            new_var = framework._varbase_creator(
                name=new_name, persistable=True)

        # load separate vars
        if separate_params is True:
            framework._dygraph_tracer().trace_op(
                type='load',
                inputs={},
                outputs={'Out': new_var},
                attrs={'file_path': os.path.join(model_path, name)})

        new_var.stop_gradient = extra_var_info[name]['stop_gradient']
        load_var_dict[new_name] = new_var
        load_var_list.append(new_var)

    # 3. load all vars
    if separate_params is False:
        if params_filename is not None:
            var_file_path = os.path.join(model_path, params_filename)
        else:
            var_file_path = os.path.join(model_path, VARIABLE_FILENAME)
        framework._dygraph_tracer().trace_op(
            type='load_combine',
            inputs={},
            outputs={'Out': load_var_list},
            attrs={'file_path': var_file_path})

    return load_var_dict


def _construct_program_holders(model_path, model_filename=None):
    # make sure the path has been checked
    program_holder_dict = dict()

    if model_filename is not None:
        # [compatible] if assign model_filename, only can load one program as Layer.forward
        model_filename = os.path.basename(model_filename)
        model_file_path = os.path.join(model_path, model_filename)
        program_holder_dict['forward'] = _ProgramHolder(
            _load_program_desc(model_file_path))
    else:
        for _, _, file_names in os.walk(model_path):
            for name in file_names:
                if 'model' in name:
                    model_file_path = os.path.join(model_path, name)
                    method_name = name.strip('_')
                    if method_name == 'model':
                        method_name = 'forward'
                    else:
                        method_name.replace('model', '')
                    program_holder_dict[method_name] = _ProgramHolder(
                        _load_program_desc(model_file_path))

    return program_holder_dict


def _construct_params_and_buffers(model_path,
                                  programs,
                                  separate_params=False,
                                  params_filename=None):
    var_info_path = os.path.join(model_path, EXTRA_VAR_INFO_FILENAME)
    if os.path.exists(var_info_path):
        var_dict = _load_persistable_vars(model_path, var_info_path,
                                          separate_params, params_filename)
    else:
        var_dict = _load_persistable_vars_by_program(
            model_path, programs['forward'], params_filename)
    return var_dict


class TranslatedLayer(layers.Layer):
    """
    TranslatedLayer is a imperative Layer for holding the model loaded by 
    :ref:`api_imperative_jit_load` . It can be used like a general Layer 
    object in eval or train mode.
    
    .. note:
        The TranslatedLayer objects should not be created by constructor, it only can be loaded and constructed by :ref:`api_imperative_jit_load` .

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid
            from paddle.fluid.dygraph import Linear
            from paddle.fluid.dygraph import declarative

            BATCH_SIZE = 32
            BATCH_NUM = 20

            def random_batch_reader():
                def _get_random_images_and_labels(image_shape, label_shape):
                    image = np.random.random(size=image_shape).astype('float32')
                    label = np.random.random(size=label_shape).astype('int64')
                    return image, label

                def __reader__():
                    for _ in range(BATCH_NUM):
                        batch_image, batch_label = _get_random_images_and_labels(
                            [BATCH_SIZE, 784], [BATCH_SIZE, 1])
                        yield batch_image, batch_label

                return __reader__

            class LinearNet(fluid.dygraph.Layer):
                def __init__(self, in_size, out_size):
                    super(LinearNet, self).__init__()
                    self._linear = Linear(in_size, out_size)

                @declarative
                def forward(self, x):
                    return self._linear(x)

            # enable dygraph mode
            fluid.enable_dygraph() 

            # 1. train & save model.
            # create network
            net = LinearNet(784, 1)
            adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
            # create data loader
            train_loader = fluid.io.DataLoader.from_generator(capacity=5)
            train_loader.set_batch_generator(random_batch_reader())
            # train
            for data in train_loader():
                img, label = data
                label.stop_gradient = True

                cost = net(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()
                adam.minimize(avg_loss)
                net.clear_gradients()

            model_path = "linear.example.model"
            fluid.dygraph.jit.save(
                layer=net,
                model_path=model_path,
                input_spec=[img])

            # 2. load model as TranslatedLayer
            translated_layer = fluid.dygraph.jit.load(model_path)
            # inference
            translated_layer.eval()
            x = fluid.dygraph.to_variable(np.random.random((1, 784)).astype('float32'))
            pred = translated_layer(x)
            # fine-tune
            translated_layer.train()
            adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=translated_layer.parameters())
            train_loader = fluid.io.DataLoader.from_generator(capacity=5)
            train_loader.set_batch_generator(random_batch_reader())
            for data in train_loader():
                img, label = data
                label.stop_gradient = True

                cost = translated_layer(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()
                adam.minimize(avg_loss)
                translated_layer.clear_gradients()
    """

    def __init__(self, programs, persistable_vars):
        super(TranslatedLayer, self).__init__()

        if not isinstance(programs, dict):
            raise TypeError(
                "TranslatedLayer need to use _ProgramHolder's dict for initialization."
            )
        if not isinstance(persistable_vars, dict):
            raise TypeError(
                "TranslatedLayer need to use persisatbale variable dict for initialization."
            )

        self._program_holder_dict = programs

        for name, var in persistable_vars.items():
            if isinstance(var, framework.ParamBase):
                self.add_parameter(name, var)
            elif isinstance(var, core.VarBase):
                self.register_buffer(name, var)
            else:
                raise TypeError(
                    "Adding persistent variable which  to layer is not supported now"
                )

        self._is_test = True

    @staticmethod
    @framework.dygraph_only
    def _construct(model_path, configs=None):
        # 0. dir and filename check
        model_path = os.path.normpath(model_path)
        if not os.path.isdir(model_path):
            raise ValueError("There is no directory named '%s'" % model_path)
        model_filename = None
        params_filename = None
        separate_params = False
        if configs is not None:
            model_filename = configs.model_filename
            params_filename = configs.params_filename
            separate_params = configs.separate_params

        # 1. load program desc & construct _ProgramHolder
        programs = _construct_program_holders(model_path, model_filename)

        # 2. load layer parameters & parameter attirbutes
        persistable_vars = _construct_params_and_buffers(
            model_path, programs, separate_params, params_filename)

        # 3. construct TranslatedLayer object
        translated_layer = TranslatedLayer(programs, persistable_vars)

        # 4. create TranslatedLayer's execution method
        for method_name, program_holder in programs.items():
            setattr(TranslatedLayer, method_name,
                    TranslatedLayer._execution_method_creator(method_name,
                                                              program_holder))

        # 5. set TranslatedLayer's default mode to eval
        translated_layer.eval()

        return translated_layer

    @staticmethod
    def _execution_method_creator(method_name, program_holder):
        def __impl__(self, *input):
            if framework.in_dygraph_mode():
                return self.__dynamic_mode_impl__(program_holder, *input)
            return self.__static_mode_impl__(program_holder, *input)

        __impl__.__name__ = method_name
        return __impl__

    def train(self):
        self._is_test = False

    def eval(self):
        self._is_test = True

    def program(self, method_name='forward'):
        if not isinstance(method_name, six.string_types):
            raise ValueError(
                "Input `method_name` type error, expected str, but received %s."
                % type(method_name))
        program_holder = self._program_holder_dict.get(method_name, None)
        if program_holder is None:
            raise RuntimeError(
                "The method `%s` is not in current TranslatedLayer." %
                method_name)
        trace_program = program_holder.infer_program if self._is_test else program_holder.train_program
        return _build_program_by_desc(trace_program)

    def __dynamic_mode_impl__(self, program_holder, *input):
        # 1. prepare inputs, outputs, attrs
        input_vars = []
        for i, value in enumerate(input):
            if not isinstance(value, (np.ndarray, core.VarBase)):
                raise TypeError(
                    "The type of input in TranslatedLayer must be numpy array or Variable(VarBase), but received %s."
                    % type(value))
            # NOTE: In order to unify the API, firstly convert the input to VarBase
            if isinstance(value, np.ndarray):
                var = core.VarBase(
                    value=value,
                    name=program_holder.input_names[i],
                    persistable=False,
                    place=framework._current_expected_place(),
                    zero_copy=True)
            else:
                var = value
                # NOTE: we changed var name here, 
                # but it may be an important name set by user
                var.name = program_holder.input_names[i]
            input_vars.append(var)

        persistable_vars = []
        for var_name in program_holder.persistable_names:
            if var_name in self._parameters:
                persistable_vars.append(self._parameters[var_name])
            elif var_name in self._buffers:
                persistable_vars.append(self._buffers[var_name])
            else:
                raise ValueError(
                    "The persistable variable %s is not exists in current TranslatedLayer."
                    % var_name)

        output_vars = []
        for var_desc in program_holder.output_decs:
            var = core.VarBase(var_desc.dtype(),
                               var_desc.shape(),
                               var_desc.name(), var_desc.type(), False)
            output_vars.append(var)

        # hold forward variables
        tmp_scope_vec = core.VarBase(core.VarDesc.VarType.FP32, [],
                                     "program_out_scope",
                                     core.VarDesc.VarType.STEP_SCOPES, True)
        tmp_scope_vec.value().set_scope(program_holder.scope)

        # 2. run prorgam by op
        trace_program = program_holder.infer_program if self._is_test else program_holder.train_program
        end_op_index = program_holder.infer_program.block(0).op_size()
        framework._dygraph_tracer().trace_op(
            type='run_program',
            inputs={'X': input_vars,
                    'Params': persistable_vars},
            outputs={'Out': output_vars,
                     'OutScope': tmp_scope_vec},
            attrs={
                'global_block': trace_program.block(0),
                'start_op_index': 0,
                'end_op_index': end_op_index,
                'is_test': self._is_test
            })

        # NOTE(chenweihang): [ why need set param's gradient type here ]
        # if user set sparse gradient mode, the param's gradient
        # will be SelectedRows, not LoDTensor. But tracer will just
        # set param grad VarBase by forward VarBase(LoDTensor)
        # If we don't change grad_var type here, RunProgramOp need
        # transform SelectedRows to LoDTensor forcely, it may not
        # be user wanted result.
        for persistable_var in persistable_vars:
            grad_var_name = var.name + core.grad_var_suffix()
            grad_var = trace_program.block(0).find_var(
                cpt.to_bytes(grad_var_name))
            # NOTE: cannot find var desc maybe not problem, 
            # such as in batch_norm
            if grad_var is None:
                continue
            persistable_var._set_grad_type(grad_var.type())

        # 3. prepare output, keep same form with inputs
        outs = output_vars
        if len(output_vars) == 1:
            outs = output_vars[0]
        return outs

    def __static_mode_impl__(self, program_holder, *input):
        # NOTE(chenweihang): [TranslatedLayer execution in dy2static]
        # When the translatedLayer is executed in static graph mode, 
        # we cannot directly execute run_program_op, which will cause
        # the program information of the original model to be lost, 
        # so here we need to restore the block, variable, operator 
        # and other members of the program according to the 
        # program desc information
        # TODO(chenweihang): Now only supports TranslatedLayer executed
        # in dynamic to static mode, not supports executed TranslatedLayer
        # in static mode, which need to rebuild startup_program of 
        # current layer

        # 1. input check
        input_vars = []
        for i, var in enumerate(input):
            check_type(var, program_holder.input_names[i], framework.Variable,
                       'TranslatedLayer')
            # NOTE: we changed var name here, 
            # but it may be an important name set by user
            var.name = program_holder.input_names[i]
            input_vars.append(var)

        # 2. recover forward program
        # NOTE(chenweihang): Here we only need to recover forward
        # part of program, because in static mode, the backward part
        # will be appended later by minimize or backward
        # NOTE(chenweihang): Here we use default program directly,
        # this is consisstent with Layer.LayerObjectHelper
        tmp_program = _build_program_by_desc


######### Debug Functions ##########


# NOTE: The StaticModelRunner is still unstable at this stage. 
# I hope to keep this debugging tool, and it can be deleted 
# after the function is stable.
class DescParser():
    @classmethod
    def print_program_desc(cls, prog, skip_op_callstack=True):
        block_idx = 0
        for i in six.moves.range(prog.num_blocks()):
            block = prog.block(i)
            cls.print_block_desc(block, block_idx, skip_op_callstack)
            block_idx += 1

    @classmethod
    def print_block_desc(cls, block, block_idx, skip_op_callstack=True):
        indent = 0

        print("{0}{1} // block {2}".format(
            cls._get_indent_space(indent), '{', block_idx))

        indent += 1
        # sort all vars
        all_vars = block.all_vars()
        for var in all_vars:
            print("{}{}".format(
                cls._get_indent_space(indent), cls.var_desc_to_code(var)))

        if len(all_vars) > 0:
            print("")

        for i in six.moves.range(block.op_size()):
            op = block.op(i)
            print("{}{}".format(
                cls._get_indent_space(indent),
                cls.op_desc_to_code(op, skip_op_callstack)))
        indent -= 1

        print("{0}{1}".format(cls._get_indent_space(indent), '}'))

    @classmethod
    def var_desc_to_code(cls, var):
        if var.type() == core.VarDesc.VarType.SELECTED_ROWS or var.type(
        ) == core.VarDesc.VarType.LOD_TENSOR:
            var_str = "{name} : fluid.{type}.shape{shape}.astype({dtype})".\
                format(i="{", e="}", name=var.name(), type=var.type(), shape=var.shape(), dtype=var.dtype())
        else:
            var_str = "{name} : fluid.{type})".\
                format(i="{", e="}", name=var.name(), type=var.type())

        var_str = "var " + var_str

        if var.persistable():
            var_str = "persist " + var_str

        return var_str

    @classmethod
    def op_desc_to_code(cls, op, skip_op_callstack=True):
        outputs_str = "{"
        for i in range(0, len(op.output_names())):
            outputs_str += "{name}=".format(name=op.output_names()[i])
            o = op.output(op.output_names()[i])
            outputs_str += "{value}".format(value=o)
            if i != len(op.output_names()) - 1:
                outputs_str += ", "
        outputs_str += "}"

        inputs_str = "{"
        for i in range(0, len(op.input_names())):
            inputs_str += "{name}=".format(name=op.input_names()[i])
            o = op.input(op.input_names()[i])
            inputs_str += "{value}".format(value=o)

            if i != len(op.input_names()) - 1:
                inputs_str += ", "
        inputs_str += "}"

        attr_names = sorted(op.attr_names())
        attrs_str = ""
        for i in range(0, len(attr_names)):
            name = attr_names[i]
            if skip_op_callstack and name == "op_callstack":
                continue

            attr_type = op.attr_type(name)
            if attr_type == core.AttrType.BLOCK:
                a = "{name} = block[{value}]".format(
                    name=name, type=attr_type, value=op._block_attr_id(name))
                attrs_str += a
                if i != len(attr_names) - 1:
                    attrs_str += ", "
                continue

            if attr_type == core.AttrType.BLOCKS:
                a = "{name} = blocks{value}".format(
                    name=name, type=attr_type, value=op._blocks_attr_ids(name))
                attrs_str += a
                if i != len(attr_names) - 1:
                    attrs_str += ", "
                continue

            a = "{name} = {value}".format(
                name=name, type=attr_type, value=op.attr(name))
            attrs_str += a
            if i != len(attr_names) - 1:
                attrs_str += ", "

        if outputs_str != "{}":
            op_str = "{outputs} = {op_type}(inputs={inputs}, {attrs})".\
                format(outputs = outputs_str, op_type=op.type(), inputs=inputs_str, attrs=attrs_str)
        else:
            op_str = "{op_type}(inputs={inputs}, {attrs})".\
                format(op_type=op.type(), inputs=inputs_str, attrs=attrs_str)
        return op_str

    @classmethod
    def _get_indent_space(cls, indent, space_num=4):
        ret = ""
        for i in range(0, indent * space_num):
            ret += " "

        return ret
