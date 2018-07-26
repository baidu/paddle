#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unlessf required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..layer_helper import LayerHelper
from ..param_attr import ParamAttr
from ..framework import convert_np_dtype_to_dtype_
from ..framework import Variable
from ..initializer import Constant, force_init_on_cpu
from ..core import VarDesc
from .layer_function_generator import templatedoc
import numpy

__all__ = [
    'create_tensor',
    'create_parameter',
    'create_global_var',
    'cast',
    'concat',
    'sums',
    'assign',
    'fill_constant_batch_size_like',
    'fill_constant',
    'argmin',
    'argmax',
    'argsort',
    'ones',
    'zeros',
    'reverse',
]


def create_tensor(dtype, name=None, persistable=False):
    """
    Create an variable, which will hold a LoDTensor with data type dtype.

    Args:
        dtype(string): 'float32'|'int32'|..., the data type of the
            created tensor.
        name(string): The name of the created tensor, if not set,
            the name will be a random unique one.
        persistable(bool): Set the persistable flag of the create tensor.

    Returns:
        Variable: The tensor variable storing the created tensor.

    Examples:
        .. code-block:: python

          tensor = fluid.layers.create_tensor(dtype='float32')
    """
    helper = LayerHelper("create_tensor", **locals())
    return helper.create_variable(
        name=helper.name, dtype=dtype, persistable=persistable)


def create_parameter(shape,
                     dtype,
                     name=None,
                     attr=None,
                     is_bias=False,
                     default_initializer=None):
    """
    Create a parameter. The parameter is a learnable variable, which can have
    gradient, and can be optimized.

    NOTE: this is a very low-level API. This API is useful when you create
    operator by your self. instead of using layers.

    Args:
        shape(list[int]): shape of the parameter
        dtype(string): element type of the parameter
        attr(ParamAttr): attributes of the parameter
        is_bias(bool): This can affect which default initializer is chosen
                       when default_initializer is None. If is_bias,
                       initializer.Constant(0.0) will be used. Otherwise,
                       Xavier() will be used.
        default_initializer(Initializer): initializer for the parameter

    Returns:
        the created parameter.

    Examples:
        >>> W = fluid.layers.create_parameter(shape=[784, 200], dtype='float32')
        >>> data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
        >>> hidden = fluid.layers.matmul(x=data, y=W)
    """
    helper = LayerHelper("create_parameter", **locals())
    if attr is None:
        attr = ParamAttr(name=name)
    return helper.create_parameter(attr, shape, dtype, is_bias,
                                   default_initializer)


def create_global_var(shape,
                      value,
                      dtype,
                      persistable=False,
                      force_cpu=False,
                      name=None):
    """
    Create a new variable in the global block(block 0).

    Args:
        shape(list[int]): shape of the variable
        value(float): the value of the variable. The new created 
                      variable will be filled with it.
        dtype(string): data type of the variable
        persistable(bool): if this variable is persistable. 
                           Default: False
        force_cpu(bool): force this variable to be on CPU. 
                         Default: False
        name(str|None): The name of the variable. If set to None the variable 
                        name will be generated automatically. 
                        Default: None

    Returns:
        Variable: the created Variable

    Examples:
        .. code-block:: python

            var = fluid.create_global_var(shape=[2,3], value=1.0, dtype='float32', 
                                 persistable=True, force_cpu=True, name='new_var')
    """
    helper = LayerHelper("global_var", **locals())
    var = helper.create_global_variable(
        dtype=dtype, shape=shape, persistable=persistable, name=name)
    helper.set_variable_initializer(
        var, initializer=Constant(
            value=float(value), force_cpu=force_cpu))
    return var


def cast(x, dtype):
    """
    This layer takes in the Variable :attr:`x` with :attr:`x.dtype` and casts 
    it to the output with :attr:`dtype`.

    Args:
        x (Variable): The input Variable for casting.
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of the output Variable.

    Returns:
        Variable: The output Variable after casting.

    Examples:
        .. code-block:: python

            data = fluid.layers.data(name='x', shape=[13], dtype='float32')
            result = fluid.layers.cast(x=data, dtype='float64')
    """
    helper = LayerHelper('cast', **locals())
    out = helper.create_tmp_variable(dtype=dtype)
    helper.append_op(
        type='cast',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'in_dtype': x.dtype,
               'out_dtype': out.dtype})
    return out


def concat(input, axis=0, name=None):
    """
    **Concat**

    This function concatenates the input along the axis mentioned
    and returns that as the output.

    Args:
        input(list): List of tensors to be concatenated
        axis(int): Integer axis along which the tensors will be concatenated
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        Variable: Output variable of the concatenation

    Examples:
        .. code-block:: python

           out = fluid.layers.concat(input=[Efirst, Esecond, Ethird, Efourth])
    """
    helper = LayerHelper('concat', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='concat',
        inputs={'X': input},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    return out


def sums(input, out=None):
    """
    This function performs the sum operation on the input and returns the
    result as the output.

    Args:
        input (Variable|list): The input tensor that has the elements
                               that need to be summed up.
        out (Variable|None): Output parameter. The sum result.
                             Default: None

    Returns:
        Variable: the sum of input. The same as the argument 'out'

    Examples:
        .. code-block:: python

          tmp = fluid.layers.zeros(shape=[10], dtype='int32')
          i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
          a0 = layers.array_read(array=tmp, i=i)
          i = layers.increment(x=i)
          a1 = layers.array_read(array=tmp, i=i)
          mean_a0 = layers.mean(a0)
          mean_a1 = layers.mean(a1)
          a_sum = layers.sums(input=[mean_a0, mean_a1])
    """
    helper = LayerHelper('sum', **locals())
    if out is None:
        out = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='sum',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={'use_mkldnn': False})
    return out


def assign(input, output=None):
    """
    **Assign**

    This function copies the *input* Variable to the *output* Variable.

    Args:
        input(Variable|numpy.ndarray): The source variable
        output(Variable|None): The destination variable

    Returns:
        Variable: The destination variable that was supplied as the *output*.

    Examples:
        .. code-block:: python

          out = fluid.layers.create_tensor(dtype='float32')
          hidden = fluid.layers.fc(input=data, size=10)
          fluid.layers.assign(hidden, out)
    """
    helper = LayerHelper('assign', **locals())
    if output is None:
        output = helper.create_tmp_variable(dtype=input.dtype)
    if isinstance(input, Variable):
        helper.append_op(
            type='assign', inputs={'X': [input]}, outputs={'Out': [output]})
    elif isinstance(input, numpy.ndarray):
        dtype = convert_np_dtype_to_dtype_(input.dtype)
        if dtype == VarDesc.VarType.FP32:
            value_name = "fp32_values"
            values = [float(v) for v in input.flat]
        elif dtype == VarDesc.VarType.INT32:
            value_name = "int32_values"
            values = [int(v) for v in input.flat]
        else:
            raise ValueError("Unsupported dtype %s", input.dtype)
        if input.size > 1024 * 1024:
            raise ValueError("The size of input is too big. Please consider "
                             "saving it to file and 'load_op' to load it")

        helper.append_op(
            type='assign_value',
            outputs={'Out': [output]},
            attrs={
                'dtype': dtype,
                'shape': list(input.shape),
                value_name: values
            })
    else:
        raise ValueError("Wrong type for assign input: %s" % type(input))

    return output


def fill_constant(shape, dtype, value, force_cpu=False, out=None):
    """
    **fill_constant**

    This function creates a tensor with specified `shape` and `dtype`, and
    initializes it with a constant specifed by `value`.

    The attribute `stop_gradient` of the created tensor is set to True.

    Args:
        shape(tuple|list|None): Shape of the output tensor.
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of the output tensor.
        value(float): The constant value used to initialize the output tensor.
        out(Variable): The output tensor.
        force_cpu(True|False): data should be on CPU if set true.

    Returns:
        Variable: The tensor variable storing the output.

    Examples:
        .. code-block:: python

          data = fluid.layers.fill_constant(shape=[1], value=0, dtype='int64')
    """

    helper = LayerHelper("fill_constant", **locals())
    if out is None:
        out = helper.create_tmp_variable(dtype=dtype)
    helper.append_op(
        type='fill_constant',
        inputs={},
        outputs={'Out': [out]},
        attrs={
            'shape': shape,
            'dtype': out.dtype,
            'value': float(value),
            'force_cpu': force_cpu or force_init_on_cpu()
        })
    out.stop_gradient = True
    return out


@templatedoc()
def fill_constant_batch_size_like(input,
                                  shape,
                                  dtype,
                                  value,
                                  input_dim_idx=0,
                                  output_dim_idx=0):
    """
    ${comment}

    It also sets *stop_gradient* to True.

    >>> data = fluid.layers.fill_constant_batch_size_like(
    >>>             input=like, shape=[1], value=0, dtype='int64')

    Args:
        input(${input_type}): ${input_comment}.

        shape(${shape_type}): ${shape_comment}.

        dtype(${dtype_type}): ${dtype_comment}.

        value(${value_type}): ${value_comment}.

        input_dim_idx(${input_dim_idx_type}): ${input_dim_idx_comment}.

        output_dim_idx(${output_dim_idx_type}): ${output_dim_idx_comment}.

    Returns:
        ${out_comment}.
    """
    helper = LayerHelper("fill_constant_batch_size_like", **locals())
    out = helper.create_tmp_variable(dtype=dtype)
    helper.append_op(
        type='fill_constant_batch_size_like',
        inputs={'Input': input},
        outputs={'Out': [out]},
        attrs={
            'shape': shape,
            'dtype': out.dtype,
            'value': float(value),
            'input_dim_idx': input_dim_idx,
            'output_dim_idx': output_dim_idx
        })
    out.stop_gradient = True
    return out


def argmin(x, axis=0):
    """
    **argmin**

    This function computes the indices of the min elements
    of the input tensor's element along the provided axis.

    Args:
        x(Variable): The input to compute the indices of
                     the min elements.
        axis(int): Axis to compute indices along.

    Returns:
        Variable: The tensor variable storing the output

    Examples:
        .. code-block:: python

          out = fluid.layers.argmin(x=in, axis=0)
          out = fluid.layers.argmin(x=in, axis=-1)
    """
    helper = LayerHelper("arg_min", **locals())
    out = helper.create_tmp_variable(VarDesc.VarType.INT64)
    helper.append_op(
        type='arg_min',
        inputs={'X': x},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    return out


def argmax(x, axis=0):
    """
    **argmax**

    This function computes the indices of the max elements
    of the input tensor's element along the provided axis.

    Args:
        x(Variable): The input to compute the indices of
                     the max elements.
        axis(int): Axis to compute indices along.

    Returns:
        Variable: The tensor variable storing the output

    Examples:
        .. code-block:: python

          out = fluid.layers.argmax(x=in, axis=0)
          out = fluid.layers.argmax(x=in, axis=-1)
    """
    helper = LayerHelper("arg_max", **locals())
    out = helper.create_tmp_variable(VarDesc.VarType.INT64)
    helper.append_op(
        type='arg_max',
        inputs={'X': x},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    return out


def argsort(input, axis=-1, name=None):
    """
    Performs sorting on the input Variable along the given axis, and outputs 
    sorted data Varibale and its corresponding index Variable with the same 
    shape as :attr:`input`.

    .. code-block:: text
    
        For example, the given axis is -1 and the input Variable

            input = [[0.15849551, 0.45865775, 0.8563702 ],
                     [0.12070083, 0.28766365, 0.18776911]],

        after argsort, the sorted Vairable becomes

            out = [[0.15849551, 0.45865775, 0.8563702 ],
                   [0.12070083, 0.18776911, 0.28766365]],

        and the sorted indices along the given axis turn outs to be

            indices = [[0, 1, 2], 
                       [0, 2, 1]]

    Args:
        input(Variable): The input Variable for sorting.
        axis(int): The axis along which to sort the input Variable. When 
                   :attr:`axis` < 0, the actual axis will be :attr:`axis` + 
                   rank(:attr:`input`). Default -1, the last dimension.
        name(str|None): (optional) A name for this layer. If set None, the 
                   layer will be named automatically.

    Returns:
        tuple: A tuple of sorted data Variable and the sorted indices.

    Examples:
        .. code-block:: python

            input = fluid.layers.data(data=[2, 3])
            out, indices = fluid.layers.argsort(input, axis=0)
    """
    helper = LayerHelper("argsort", **locals())
    out = helper.create_tmp_variable(dtype=input.dtype, stop_gradient=True)
    ids = helper.create_tmp_variable(VarDesc.VarType.INT64, stop_gradient=True)
    helper.append_op(
        type='argsort',
        inputs={'X': input},
        outputs={'Out': out,
                 'Indices': ids},
        attrs={'axis': axis})
    return out, ids


def ones(shape, dtype, force_cpu=False):
    """
    **ones**

    This function creates a tensor of specified *shape* and
    *dtype*, and initializes this with 1.

    It also sets *stop_gradient* to True.

    Args:
        shape(tuple|list|None): Shape of output tensor
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of output tensor

    Returns:
        Variable: The tensor variable storing the output

    Examples:
        .. code-block:: python

          data = fluid.layers.ones(shape=[1], dtype='int64')
    """
    return fill_constant(value=1.0, **locals())


def zeros(shape, dtype, force_cpu=False):
    """
    **zeros**

    This function creates a tensor of specified *shape* and
    *dtype*, and initializes this with 0.

    It also sets *stop_gradient* to True.

    Args:
        shape(tuple|list|None): Shape of output tensor.
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of output tensor.
        force_cpu(bool, default False): Whether to make output stay on CPU.

    Returns:
        Variable: The tensor variable storing the output.

    Examples:
        .. code-block:: python

          data = fluid.layers.zeros(shape=[1], dtype='int64')
    """
    return fill_constant(value=0.0, **locals())


def reverse(x, axis):
    """
    **reverse**

    This function reverse the input 'x' along given axises.

    Args:
        x(Vairbale): the input to be reversed.
        axis(int|tuple|list): Axis that along which order of elements
                    is reversed. If it is a tuple or a list, reversing
                    will be apply on each axis in the tuple or list.

    Returns:
        Variable: The reversed tensor.

    Examples:
        .. code-block:: python

          out = fluid.layers.reverse(x=in, axis=0)
          # or:
          out = fluid.layers.reverse(x=in, axis=[0,1])
    """
    if isinstance(axis, int):
        axis = [axis]
    helper = LayerHelper("reverse", **locals())
    out = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type='reverse',
        inputs={'Input': x},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    return out


def save(x, file_path, overwrite=True):
    """
    Saves a variable as a file.

    Args:
        x(variable): The Tensor/LoDTensor to be saved.
        file_path(str): The file path where the variable will be saved.
        overwrite(bool): Whether or not cover the given file when it has already
            existed. If it's set 'False' and the file is existed, a runtime
            error will be thrown.
    """
    helper = LayerHelper("save", **locals())
    helper.append_op(
        type="save",
        inputs={"input": x},
        outputs={},
        args={"file_path": file_path,
              "overwrite": overwrite})


def save_combine(x, file_path, overwrite=True):
    """
    Saves a list of variables into a single file.

    Args:
        x(list): A list of Tensor/LoDTensor variables to be saved together in
                 a single file.
        file_path(str): The file path where variables will be saved.
        overwrite(bool): Whether or not cover the given file when it has already
            existed. If it's set 'False' and the file is existed, a runtime
            error will be thrown.

    Returns:
        There is no return value.

    Examples:

        .. code-block:: python

            v1 = fluid.layers.data(name="data",
                                   shape=(4, 6),
                                   dtype="float32")
            v2 = fluid.layers.data(name="data",
                                   shape=(6, 8, 4),
                                   dtype="float32")
            normed = fluid.layers.save_combine([v1, v2], file_path="output")
    """
    helper = LayerHelper("save_combine", **locals())
    helper.append_op(
        type="save_combine",
        inputs={"input": x},
        outputs={},
        args={"file_path": file_path,
              "overwrite": overwrite})


def load_combine(out, file_path):
    """
    Loads a list of vairables from a single file.

    Args:
        out(list): The list of variables to be read from the disk file.
        file_path(str): The path of the disk file.
    """
    helper = LayerHelper("load_combine", **locals())
    helper.append_op(
        type="load_combine",
        inputs={},
        output={"Out": out},
        args={"file_path": file_path})
