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
"""
decorator to deprecate a function or class
"""

import warnings
import functools
import paddle
import inspect
import importlib

# NOTE(zhiqiu): Since python 3.2, DeprecationWarning is ignored by default,
# and since python 3.7, it is once again shown by default when triggered directly by code in __main__.
# See details: https://docs.python.org/3/library/warnings.html#default-warning-filter
# The following line set DeprecationWarning to show once, which is expected to work in python 3.2 -> 3.6
# However, doing this could introduce one samll side effect, i.e., the DeprecationWarning which is not issued by @deprecated.
# The side effect is acceptable, and we will find better way to do this if we could.
warnings.filterwarnings("once", category=DeprecationWarning, module="__main__")


def deprecated(update_to="", since="", reason=""):
    """Decorate a function to signify its deprecation.

       This function wraps a method that will soon be removed and does two things:
           - The docstring of the API will be modified to include a notice
             about deprecation."
           - Raises a :class:`~exceptions.DeprecatedWarning` when old API is called.
       Args:
           since(str): The version at which the decorated method is considered deprecated.
           update_to(str): The new API users should use.
           reason(str): The reason why the API is deprecated.
       Returns:
           decorator: decorated function or class.
    """

    def decorator(wrapped):
        # NOTE(zhiqiu): Currently, some unary operators in Paddle are generated dynamicly 
        # (see details in ops.py), for example, paddle.tanh. 
        # While, paddle.tanh is alias of paddle.fluid.layers.tanh. And paddle.fluid.layers.tanh is 
        # marked as deprecated. So, it may lead to the following warnings when using paddle.tanh():
        #   API "paddle.fluid.layers.tanh" is deprecated since 2.0.0, 
        #   and may be removed in future versions. Please use "paddle.tanh" instead.
        # This problem should be fixed.

        # TODO(zhiqiu) We temporally disable the warnings by return, and it should be re-enabled after 
        # the above problem fixed.
        return wrapped
        """construct warning message, and return a decorated function or class."""
        assert isinstance(update_to, str), 'type of "update_to" must be str.'
        assert isinstance(since, str), 'type of "since" must be str.'
        assert isinstance(reason, str), 'type of "reason" must be str.'

        _since = since.strip()
        _update_to = update_to.strip()
        _reason = reason.strip()

        msg = 'API "{}.{}" is deprecated'.format(wrapped.__module__,
                                                 wrapped.__name__)
        if len(_since) > 0:
            msg += " since {}".format(_since)
        msg += ", and may be removed in future versions."
        if len(_update_to) > 0:
            assert _update_to.startswith(
                "paddle."
            ), 'Argument update_to must start with "paddle.", your value is "{}"'.format(
                update_to)
            msg += ' Please use "{}" instead.'.format(_update_to)
        if len(_reason) > 0:
            msg += "\n reason: {}".format(_reason)

        need_warn = False
        """deprecated warning should be fired in 3 circumstances:
        1. current version is develop version, i.e. "0.0.0", because we assume develop version is always the latest version.
        2. since version is empty, in this case, API is deprecated in all versions.
        3. current version is newer than since version.
        """
        v_current = [int(i) for i in paddle.__version__.split(".")]
        v_current += [0] * (4 - len(v_current))
        v_since = [int(i) for i in _since.split(".")]
        v_since += [0] * (4 - len(v_since))
        if paddle.__version__ == "0.0.0" or _since == "" or v_current >= v_since:
            need_warn = True

        if inspect.isclass(wrapped):
            origin_new = wrapped.__new__

            @functools.wraps(origin_new)
            def wrapped_cls(cls, *args, **kwargs):
                if need_warn:
                    warnings.warn(
                        msg, category=DeprecationWarning, stacklevel=2)

                return origin_new(cls, *args, **kwargs)

            wrapped.__new__ = staticmethod(wrapped_cls)
            return wrapped

        elif inspect.isroutine(wrapped):

            @functools.wraps(wrapped)
            def wrapper(*args, **kwargs):
                if need_warn:
                    warnings.warn(
                        msg, category=DeprecationWarning, stacklevel=2)
                return wrapped(*args, **kwargs)

            return wrapper
        else:
            raise TypeError(
                "Currently, '@deprecated' only supports class, method and function. but got {}".
                format(type(wrapped)))

    return decorator
