# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import functools
import sys

__all__ = ['deprecated']


def deprecated(since, instead, extra_message=""):
    def decorator(func):
        err_msg = "API {0} is deprecated since {1}. Please use {2} instead.".format(
            func.__name__, since, instead)
        if len(extra_message) != 0:
            err_msg += "\n"
            err_msg += extra_message

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(err_msg, file=sys.stderr)
            return func(*args, **kwargs)

        wrapper.__doc__ += "\n    "
        wrapper.__doc__ += err_msg
        return wrapper

    return decorator
