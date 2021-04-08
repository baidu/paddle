# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.framework import dygraph_only
from paddle.fluid import core


def with_mateclass(meta, *bases):
    class impl(meta):
        def __new__(cls, name, temp_bases, attrs):
            return meta(name, bases, attrs)

    return type.__new__(impl, "impl", (), {})


class CFunction(object):
    @classmethod
    @dygraph_only
    def apply(cls, *args, **kwargs):
        place = paddle.fluid.framework._current_expected_place()
        with paddle.fluid.dygraph.no_grad():
            return core.PyLayer_apply(place, cls, *args, **kwargs)


class BackwardFunction(object):
    def backward(self, *args, **kwargs):
        with paddle.fluid.dygraph.no_grad():
            return self._forward_cls.backward(*args, **kwargs)


class FunctionMeta(type):
    def __init__(cls, name, bases, attrs):
        cls._backward_function = type(name + '_backward', (BackwardFunction, ),
                                      {"_forward_cls": cls})

        return super(FunctionMeta, cls).__init__(name, bases, attrs)


class PyLayer(with_mateclass(FunctionMeta, CFunction)):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError("`forward` of `PyLayer` is not Implemented.")

    @staticmethod
    def backward(ctx, *args, **kwargs):
        raise NotImplementedError("`backward` of `PyLayer` is not Implemented")
