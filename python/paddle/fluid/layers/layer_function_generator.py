#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import functools
import warnings
import string

from six.moves import cStringIO
from ..proto import framework_pb2
from ..framework import OpProtoHolder, Variable
from ..layer_helper import LayerHelper

__all__ = ['deprecated', 'generate_layer_fn', 'autodoc', 'templatedoc']


def _convert_(name):
    """
    Formatting.

    Args:
       name: The name/alias

    This function takes in a name and converts it to a standard format of
    group1_group2. Where as per the regular expression, group1 can have
    alphabets and numbers and group2 has capital alphabets.

    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _type_to_str_(tp):
    return framework_pb2.AttrType.Name(tp)


_two_dollar_pattern_ = re.compile(r"\$\$([^\$]+)\$\$")
_single_dollar_pattern_ = re.compile(r"\$([^\$]+)\$")
_two_bang_pattern_ = re.compile(r"!!([^!]+)!!")


def escape_math(text):
    return _two_bang_pattern_.sub(
        r'$$\1$$',
        _single_dollar_pattern_.sub(r':math:`\1`',
                                    _two_dollar_pattern_.sub(r"!!\1!!", text)))


def _generate_doc_string_(op_proto):
    """
    Generate docstring by OpProto

    Args:
        op_proto (framework_pb2.OpProto): a protobuf message typed OpProto

    Returns:
        str: the document string
    """

    if not isinstance(op_proto, framework_pb2.OpProto):
        raise TypeError("OpProto should be `framework_pb2.OpProto`")

    buf = cStringIO()
    buf.write(escape_math(op_proto.comment))
    buf.write('\nArgs:\n')
    for each_input in op_proto.inputs:
        line_begin = '    {0}: '.format(_convert_(each_input.name))
        buf.write(line_begin)
        buf.write(escape_math(each_input.comment))
        if each_input.duplicable:
            buf.write("  Duplicatable.")
        if each_input.dispensable:
            buf.write("  Optional.")
        buf.write('\n')

    skip_attrs = OpProtoHolder.generated_op_attr_names()

    for each_attr in op_proto.attrs:
        if each_attr.name in skip_attrs:
            continue
        buf.write('    ')
        buf.write(each_attr.name)
        buf.write(' (')
        buf.write(_type_to_str_(each_attr.type))
        buf.write('): ')
        buf.write(escape_math(each_attr.comment))
        buf.write('\n')

    if len(op_proto.outputs) != 0:
        buf.write('\nReturns:\n')
        buf.write('    ')
        for each_opt in op_proto.outputs:
            if not each_opt.intermediate:
                break
        buf.write(escape_math(each_opt.comment))

    return buf.getvalue()


def generate_layer_fn(op_type):
    """Register the Python layer for an Operator.

    Args:
       op_type: The name of the operator to be created.

    This function takes in the operator type (sigmoid, mean , average etc) and
    creates the operator functionality.

    """
    op_proto = OpProtoHolder.instance().get_op_proto(op_type)
    not_intermediate_outputs = \
        [output for output in op_proto.outputs if not output.intermediate]
    intermediate_outputs = \
        [output for output in op_proto.outputs if output.intermediate]

    if len(not_intermediate_outputs) != 1:
        raise ValueError("Only one non intermediate output operator can be",
                         "automatically generated. {0}".format(op_type))

    if not_intermediate_outputs[0].duplicable:
        raise ValueError(
            "Only non duplicable op can be automatically generated.")

    for output in intermediate_outputs:
        if output.duplicable:
            raise ValueError("The op can be automatically generated only when ",
                             "all intermediate ops are not duplicable.")

    o_name = not_intermediate_outputs[0].name
    intermediate_output_names = [output.name for output in intermediate_outputs]

    def infer_and_check_dtype(op_proto, *args, **kwargs):
        """
        This function performs the sanity check for dtype and
        instance type.
        """
        dtype = None
        for ipt in op_proto.inputs:
            name = _convert_(ipt.name)
            val = kwargs.pop(name, [])
            if not isinstance(val, list) and not isinstance(val, tuple):
                val = [val]
            if len(val) == 0:
                val = [args[0]]
                args = args[1:]

            for each in val:
                if not isinstance(each, Variable):
                    raise ValueError("input of {0} must be variable".format(
                        op_type))

                if dtype is None:
                    dtype = each.dtype
                elif dtype != each.dtype:
                    raise ValueError(
                        "operator {0} must input same dtype. {1} vs {2}".format(
                            op_type, dtype, each.dtype))

        return dtype

    def func(*args, **kwargs):
        helper = LayerHelper(op_type, **kwargs)

        dtype = infer_and_check_dtype(op_proto, *args, **kwargs)

        inputs = dict()
        for ipt in op_proto.inputs:
            name = _convert_(ipt.name)
            val = kwargs.pop(name, [])
            if not isinstance(val, list) and not isinstance(val, tuple):
                val = [val]
            if len(val) == 0 and len(args) != 0:
                val = args[0]
                args = args[1:]
            inputs[ipt.name] = val

        outputs = dict()
        out = kwargs.pop(_convert_(o_name), [])
        if out:
            out_var = out[0] if (isinstance(out, list) or
                                 isinstance(out, tuple)) else out
        else:
            out_var = helper.create_tmp_variable(dtype=dtype)
        outputs[o_name] = [out_var]
        for name in intermediate_output_names:
            outputs[name] = [helper.create_tmp_variable(dtype=dtype)]
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=kwargs)
        return helper.append_activation(out_var)

    func.__name__ = op_type
    func.__doc__ = _generate_doc_string_(op_proto)
    return func


def deprecated(func_or_class):
    """
    Deprecated warning decorator. It will result a warning message.
    Should be used before class or function, member function
    """

    @functools.wraps(func)
    def func_wrapper(*args, **kwargs):
        """
        Wrap func with deprecated warning
        """
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return func_wrapper


def autodoc(comment=""):
    def __impl__(func):
        func.__doc__ = _generate_doc_string_(OpProtoHolder.instance(
        ).get_op_proto(func.__name__)) + comment
        return func

    return __impl__


def templatedoc(op_type=None):
    """
    Decorator of layer function. It will use the docstring from the layer
    function as the template. The template arguments are:

    * ${comment}: The operator comment written in CPP.
    * ${{name}_comment}: The comment of ${name} written with AddAttr, AddOutput,
        and AddInput. The ${name} is Python snake style. i.e., xxx_xxx.
    * ${{name}_type}: The type of ${name}.

    Returns:
        Decorated function.
    """

    def trim_ending_dot(msg):
        return msg.rstrip('.')

    def __impl__(func):
        if op_type is None:
            op_type_name = func.__name__
        else:
            op_type_name = op_type
        op_proto = OpProtoHolder.instance().get_op_proto(op_type_name)
        tmpl = string.Template(func.__doc__)

        comment_lines = op_proto.comment.split("\n")
        comment = ""
        for line in comment_lines:
            line = line.strip()
            if len(line) != 0:
                comment += escape_math(line)
                comment += " "
            elif len(comment) != 0:
                comment += "\n    \n    "

        args = {"comment": trim_ending_dot(comment)}
        for each_input in op_proto.inputs:
            input_name = _convert_(each_input.name)
            args["{0}_comment".format(input_name)] = trim_ending_dot(
                each_input.comment)
            args["{0}_type".format(input_name)] = "Variable"
        for each_attr in op_proto.attrs:
            input_name = _convert_(each_attr.name)
            args["{0}_comment".format(input_name)] = trim_ending_dot(
                each_attr.comment)
            args["{0}_type".format(input_name)] = _type_to_str_(each_attr.type)

        for each_opt in op_proto.outputs:
            output_name = _convert_(each_opt.name)
            args["{0}_comment".format(output_name)] = trim_ending_dot(
                each_opt.comment)
            args["{0}_type".format(output_name)] = "Variable"
        func.__doc__ = tmpl.substitute(args)
        return func

    return __impl__
