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

from ..framework import Variable, unique_name
from .layer_function_generator import OpProtoHolder
from ..initializer import force_init_on_cpu


def monkey_patch_variable():
    def unique_tmp_name():
        return unique_name.generate("tmp")

    def safe_get_dtype(var):
        try:
            dtype = var.dtype
        except:
            raise ValueError("Cannot get data type from %s", var.name)
        return dtype

    def create_tensor(block, value, dtype, shape):
        value = float(value)
        tmp_name = unique_tmp_name()
        var = block.create_var(name=tmp_name, shape=shape, dtype=dtype)
        block.append_op(
            type="fill_constant",
            outputs={'Out': [var]},
            attrs={
                'dtype': var.dtype,
                'shape': shape,
                'value': value,
                'force_cpu': force_init_on_cpu()
            })
        return var

    def create_scalar(block, value, dtype):
        return create_tensor(block, value, dtype, shape=[1])

    def create_tensor_with_batchsize(ref_var, value, dtype):
        assert isinstance(ref_var, Variable)
        value = float(value)
        tmp_name = unique_tmp_name()
        var = ref_var.block.create_var(name=tmp_name, dtype=dtype)
        batch_dim = -1
        for i, d in enumerate(ref_var.shape):
            if d < 0:
                batch_dim = i
                break
        assert batch_dim != -1
        ref_var.block.append_op(
            type='fill_constant_batch_size_like',
            outputs={'Out': [var]},
            inputs={'Input': [ref_var]},
            attrs={
                'shape': ref_var.shape,
                'value': value,
                'input_dim_idx': batch_dim,
                'output_dim_idx': batch_dim
            })
        return var

    def astype(self, dtype):
        """
        Cast a variable to a specified data type.
        NOTE: The variable must be a Tensor
        Args:
            self(Variable): The source variable
            dtype: The target dtype

        Returns:
            Variable with new dtype
        """
        tmp_name = unique_tmp_name()
        out = self.block.create_var(name=tmp_name, dtype=dtype)
        self.block.append_op(
            type="cast",
            inputs={"X": [self]},
            outputs={"Out": [out]},
            attrs={"in_dtype": self.dtype,
                   "out_dtype": out.dtype})
        return out

    def _elemwise_method_creator_(method_name, op_type, reverse=False):
        def __impl__(self, other_var):
            lhs_dtype = safe_get_dtype(self)

            if not isinstance(other_var, Variable):
                if reverse:
                    has_batch_size = False
                    for elem in self.shape:
                        if elem < 0:
                            has_batch_size = True
                            break
                    if not has_batch_size:
                        other_var = create_tensor(
                            self.block,
                            other_var,
                            dtype=lhs_dtype,
                            shape=self.shape)
                    else:
                        other_var = create_tensor_with_batchsize(
                            self, other_var, lhs_dtype)
                else:
                    # add fill_op to self.block
                    other_var = create_scalar(
                        self.block, value=other_var, dtype=lhs_dtype)

            rhs_dtype = safe_get_dtype(other_var)
            if lhs_dtype != rhs_dtype:
                other_var = astype(other_var, lhs_dtype)
            if reverse:
                tmp = self
                self = other_var
                other_var = tmp

            tmp_name = unique_tmp_name()
            out = self.block.create_var(name=tmp_name, dtype=lhs_dtype)

            axis = -1
            if other_var.shape[0] == -1:
                axis = 0
            assert len(self.shape) >= len(other_var.shape), (
                "The rank of the first argument of an binary operator cannot "
                "be smaller than the rank of its second argument: %s vs %s" %
                (len(self.shape), len(other_var.shape)))

            self.block.append_op(
                type=op_type,
                inputs={'X': [self],
                        'Y': [other_var]},
                outputs={'Out': out},
                attrs={'axis': axis})
            return out

        comment = OpProtoHolder.instance().get_op_proto(op_type).comment

        __impl__.__doc__ = """
        {0}
        Args:
            self(Variable): left hand variable
            other_var(Variable|float|int): right hand variable

        Returns:
            Variable
        """.format(comment)
        __impl__.__name__ = method_name
        return __impl__

    # inject methods
    for method_name, op_type, reverse in (
        ("__add__", "elementwise_add", False),
            # a+b == b+a. Do not need to reverse explicitly
        ("__radd__", "elementwise_add", False),
        ("__sub__", "elementwise_sub", False),
        ("__rsub__", "elementwise_sub", True),
        ("__mul__", "elementwise_mul", False),
            # a*b == b*a. Do not need to reverse explicitly
        ("__rmul__", "elementwise_mul", False),
        ("__div__", "elementwise_div", False),
        ("__truediv__", "elementwise_div", False),
        ("__rdiv__", "elementwise_div", True),
        ("__rtruediv__", "elementwise_div", True),
        ("__pow__", "elementwise_pow", False),
        ("__rpow__", "elementwise_pow", True),
            # for logical compare
        ("__eq__", "equal", False),
        ("__ne__", "not_equal", False),
        ("__lt__", "less_than", False),
        ("__le__", "less_equal", False),
        ("__gt__", "greater_than", False),
        ("__ge__", "greater_equal", False)):
        setattr(Variable, method_name,
                _elemwise_method_creator_(method_name, op_type, reverse))

    Variable.astype = astype
