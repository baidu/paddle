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

import importlib
import inspect
import collections
import sys
import pydoc
import hashlib
import six
import functools
import platform

__all__ = ['get_apis_with_and_without_core_ops', ]

# APIs that should not be printed into API.spec 
omitted_list = [
    "paddle.fluid.LoDTensor.set",  # Do not know why it should be omitted
]


def md5(doc):
    try:
        hashinst = hashlib.md5()
        if platform.python_version()[0] == "2":
            hashinst.update(str(doc))
        else:
            hashinst.update(str(doc).encode('utf-8'))
        md5sum = hashinst.hexdigest()
    except UnicodeDecodeError as e:
        md5sum = None
        print(
            "Error({}) occurred when `md5({})`, discard it.".format(
                str(e), doc),
            file=sys.stderr)
    return md5sum


def split_with_and_without_core_ops(member, cur_name):
    if cur_name in omitted_list:
        return

    if inspect.isclass(member):
        pass
    else:
        try:
            source = inspect.getsource(member)
            if source.find('append_op') != -1:
                if source.find('core.ops') != -1:
                    api_with_ops.append(cur_name)
                else:
                    api_without_ops.append(cur_name)
        except:
            # If getsource failed (pybind API or function inherit from father class), just skip
            pass


def get_md5_of_func(member, cur_name):
    if cur_name in omitted_list:
        return

    doc_md5 = md5(member.__doc__)

    if inspect.isclass(member):
        pass
    else:
        try:
            source = inspect.getsource(member)
            func_dict[cur_name] = md5(source)
        except:
            # If getsource failed (pybind API or function inherit from father class), just skip
            pass


def visit_member(parent_name, member, func):
    cur_name = ".".join([parent_name, member.__name__])
    if inspect.isclass(member):
        func(member, cur_name)
        for name, value in inspect.getmembers(member):
            if hasattr(value, '__name__') and (not name.startswith("_") or
                                               name == "__init__"):
                visit_member(cur_name, value, func)
    elif inspect.ismethoddescriptor(member):
        return
    elif callable(member):
        func(member, cur_name)
    elif inspect.isgetsetdescriptor(member):
        return
    else:
        raise RuntimeError("Unsupported generate signature of member, type {0}".
                           format(str(type(member))))


def is_primitive(instance):
    int_types = (int, long) if six.PY2 else (int, )
    pritimitive_types = int_types + (float, str)
    if isinstance(instance, pritimitive_types):
        return True
    elif isinstance(instance, (list, tuple, set)):
        for obj in instance:
            if not is_primitive(obj):
                return False

        return True
    else:
        return False


def visit_all_module(mod, visited, func):
    mod_name = mod.__name__
    if mod_name != 'paddle' and not mod_name.startswith('paddle.'):
        return

    if mod_name.startswith('paddle.fluid.core'):
        return

    if mod in visited:
        return

    visited.add(mod)

    for member_name in (
            name
            for name in (mod.__all__ if hasattr(mod, "__all__") else dir(mod))
            if not name.startswith("_")):
        instance = getattr(mod, member_name, None)
        if instance is None:
            continue

        if is_primitive(instance):
            continue

        if not hasattr(instance, "__name__"):
            continue

        if inspect.ismodule(instance):
            visit_all_module(instance, visited, func)
        else:
            visit_member(mod.__name__, instance, func)


def get_apis_with_and_without_core_ops(modules):
    global api_with_ops, api_without_ops
    api_with_ops = []
    api_without_ops = []
    for m in modules:
        visit_all_module(
            importlib.import_module(m), set(), split_with_and_without_core_ops)
    return api_with_ops, api_without_ops


def get_api_source_desc(modules):
    global func_dict
    func_dict = collections.OrderedDict()
    for m in modules:
        visit_all_module(importlib.import_module(m), set(), get_md5_of_func)
    return func_dict


if __name__ == "__main__":
    if len(sys.argv) > 1:
        modules = sys.argv[2].split(",")
        if sys.argv[1] == '-c':
            api_with_ops, api_without_ops = get_apis_with_and_without_core_ops(
                modules)

            print('api_with_ops:', len(api_with_ops))
            print('\n'.join(api_with_ops))
            print('\n==============\n')
            print('api_without_ops:', len(api_without_ops))
            print('\n'.join(api_without_ops))

        if sys.argv[1] == '-p':
            func_dict = get_api_source_desc(modules)
            for name in func_dict:
                print(name, func_dict[name])

    else:
        print("""Usage: 
            1. Count and list all operator-raleated APIs that contains append_op but not core.ops.xx. 
                python ./count_api_without_core_ops.py -c paddle
            2. Print api and the md5 of source code of the api.
                python ./count_api_without_core_ops.py -p paddle
            """)
