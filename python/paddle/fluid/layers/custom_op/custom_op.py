#-*- coding:utf-8 -*-

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

from .. import nn
from ...framework import in_dygraph_mode, dygraph_only, _dygraph_tracer, default_main_program
from paddle.fluid import core


def with_mateclass(meta, *bases):
    class impl(meta):
        def __new__(cls, name, temp_bases, attrs):
            return meta(name, bases, attrs)

    return type.__new__(impl, "impl", (), {})


class _HookMixin(object):
    pass


class _ContextMethodMixin(object):
    def save_for_backward(self, *tensor):
        self.forward_tensor = tensor

    def mark_dirty(self, *args):
        self.dirty_tensor = args

    def mark_non_differentiable(self, *args):
        self.non_differentiable = args


# TODO(weixin):implement with C++
# core.CusPyFunc
class CFunction(object):
    @classmethod
    def apply(cls, *args, **kwargs):
        # pass        
        return core.CusPyFunc_apply(cls, *args, **kwargs)


class BackwardFunction(core.CusPyFunc_Context, _ContextMethodMixin):
    pass


class FunctionMeta(type):
    def __init__(cls, name, bases, attrs):

        cls._backward_function = type(name + '_backward', (BackwardFunction, ),
                                      {"_forward_function": cls})

        if hasattr(cls, "backward"):
            cls._backward_id = core.CusPyFunc_AppendPythonContext2Op(
                cls.backward)
        else:
            cls._backward_id = -1

        return super(FunctionMeta, cls).__init__(name, bases, attrs)


class PyFunction(with_mateclass(FunctionMeta, CFunction)):
    # class PyFunction(object): 
    @staticmethod
    def forward(ctx, *args, **kwargs):
        assert False, "Not implemented"

    @staticmethod
    def backward(ctx, *args, **kwargs):
        assert False, "Not implemented"
