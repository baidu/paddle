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

import numpy as np
import contextlib
from .framework import Program, default_main_program, Variable
from . import core

__all__ = [
    'Executor', 'global_scope', 'scope_guard', '_switch_scope', 'fetch_var'
]

g_scope = core.Scope()


def global_scope():
    """
    Get the global/default scope instance. There are a lot of APIs use
    :code:`global_scope` as its default value, e.g., :code:`Executor.run`

    Returns:
        Scope: The global/default scope instance.
    """
    return g_scope


def _switch_scope(scope):
    global g_scope
    ex = g_scope
    g_scope = scope
    return ex


@contextlib.contextmanager
def scope_guard(scope):
    """
    Change the global/default scope instance by Python `with` statement. All
    variable in runtime will assigned to the new scope.

    Examples:
        >>> import paddle.fluid as fluid
        >>> new_scope = fluid.Scope()
        >>> with fluid.scope_guard(new_scope):
        >>>     ...

    Args:
        scope: The new global/default scope.
    """
    ex = _switch_scope(scope)
    yield
    _switch_scope(ex)


def as_numpy(tensor):
    """
    Convert a Tensor to a numpy.ndarray, its only support Tensor without LoD information.
    For higher dimensional sequence data, please use LoDTensor directly.
    Examples:
        >>> import paddle.fluid as fluid
        >>> outs = executor.run(...)
        >>> np_outs = map(lambda x: as_numpy(x), outs)
        >>>     ...

    Args:
       tensor(Variable): a instance of Tensor

    Returns:
        numpy.ndarray
    """
    if isinstance(tensor, core.LoDTensorArray):
        return [as_numpy(t) for t in tensor]
    if isinstance(tensor, list):
        return [as_numpy(t) for t in tensor]
    assert isinstance(tensor, core.LoDTensor)
    lod = tensor.lod()
    if len(lod) > 0:
        raise RuntimeError("Some of your fetched tensors hold LoD information. \
            They can not be completely cast to Python ndarray. \
            Please set the parameter 'return_numpy' as 'False' to \
            return LoDTensor itself directly.")
    return np.array(tensor)


def has_feed_operators(block, feed_targets, feed_holder_name):
    """ Check whether the block already has feed operators.

    Return false if the block does not have any feed operators.
    If some feed operators have been prepended to the block, check that
    the info contained in these feed operators matches the feed_targets
    and feed_holder_name. Raise exception when any mismatch is found.
    Return true when the block has feed operators with matching info.

    Args:
        block: a block instance (typically global block of a program)
        feed_targets: a dictionary of {feed_target_name: feed_target_data}
        feed_holder_name: the name of the variable that holds the data of
            all feed targets. The type of this feed_holder variable is
            FEED_MINIBATCH, which is essentially vector<LoDTensor>.

    Returns:
        A boolean value that indicates whether a block has feed operators
        that match the info contained in feed_targets and feed_holder_name.
    """

    feed_count = 0
    for op in block.ops:
        if op.desc.type() == 'feed':
            feed_count += 1
            assert op.desc.input('X')[0] == feed_holder_name
            feed_target_name = op.desc.output('Out')[0]
            if feed_target_name not in feed_targets:
                raise Exception("'feed_targets' does not have {} variable".
                                format(feed_target_name))
        else:
            break
    if feed_count > 0 and feed_count != len(feed_targets):
        raise Exception(
            "Feed operators in program desc do not match 'feed_targets'")
    return feed_count > 0


def has_fetch_operators(block, fetch_targets, fetch_holder_name):
    """ Check whether the block already has fetch operators.

    Return false if the block does not have any fetch operators.
    If some fetch operators have been appended to the block, check that
    the info contained in these fetch operators matches the fetch_targets
    and fetch_holder_name. Raise exception when any mismatch is found.
    Return true when the block has fetch operators with matching info.

    Args:
        block: a block instance (typically global block of a program)
        fetch_targets: a dictionary of {fetch_target_name: fetch_target_data}
        fetch_holder_name: the name of the variable that holds the data of
            all fetch targets. The type of this fetch_holder variable is
            FETCH_LIST, which is essentially vector<LoDTensor>.

    Return:
        A boolean value that indicates whether a block has fetch operators
        that match the info contained in fetch_targets and fetch_holder_name.
    """

    fetch_count = 0
    for op in block.ops:
        if op.desc.type() == 'fetch':
            fetch_count += 1
            assert op.desc.output('Out')[0] == fetch_holder_name
            fetch_target_name = op.desc.input('X')[0]
            if fetch_target_name not in [
                    var.desc.name() for var in fetch_targets
            ]:
                raise Exception("'fetch_targets' does not have {} variable".
                                format(fetch_target_name))
            idx = op.desc.attr('col')
            assert fetch_target_name == fetch_targets[idx].desc.name()
    if fetch_count > 0 and fetch_count != len(fetch_targets):
        raise Exception(
            "Fetch operators in program desc do not match 'fetch_targets'")
    return fetch_count > 0


def fetch_var(name, scope=None, return_numpy=True):
    """
    Fetch the value of the variable with the given name from the
    given scope.

    Args:
        name(str): name of the variable. Typically, only persistable variables
            can be found in the scope used for running the program.
        scope(core.Scope|None): scope object. It should be the scope where
            you pass to Executor.run() when running your program.
            If None, global_scope() will be used. Default None.
        return_numpy(bool): whether convert the tensor to numpy.ndarray.
            Default True.

    Returns:
       LodTensor|numpy.ndarray
    """
    assert isinstance(name, str)
    if scope is None:
        scope = global_scope()
    assert isinstance(scope, core.Scope)

    var = scope.find_var(name)
    assert var is not None, (
        "Cannot find " + name + " in scope. Perhaps you need to make the"
        " variable persistable by using var.persistable = True in your"
        " program.")
    tensor = var.get_tensor()
    if return_numpy:
        tensor = as_numpy(tensor)
    return tensor


def _get_program_cache_key(feed, fetch_list):
    feed_var_names = list(feed.keys())

    def to_name_str(var):
        if isinstance(var, Variable):
            return var.desc.name()
        elif isinstance(var, str):
            return var
        elif isinstance(var, str):
            return str(var)
        else:
            raise TypeError(str(var) + " should be Variable or str")

    fetch_var_names = list(map(to_name_str, fetch_list))

    return str(feed_var_names + fetch_var_names)


class Executor(object):
    """
    An Executor in Python, only support the single-GPU running. For multi-cards, please refer to
    ParallelExecutor.
    Python executor takes a program, add feed operators and fetch operators to this program according
    to feed map and fetch_list. Feed map provides input data for the program. fetch_list provides
    the variables(or names) that user want to get after program run. Note: the executor will run all
    operators in the program but not only the operators dependent by the fetch_list.
    It store the global variables into the global scope, and create a local scope for the temporary 
    variables. The local scope contents will be discarded after every minibatch forward/backward finished. 
    But the global scope variables will be persistent through different runs.
    All of ops in program will be running in sequence.

    Args:
        place(core.CPUPlace|core.CUDAPlace(n)): indicate the executor run on which device

    Note: For debugging complicated network in parallel-GPUs, you can test it on the executor.
    They has the exactly same arguments, and expected the same results.
    """

    def __init__(self, place):
        self.place = place
        p = core.Place()
        p.set_place(place)
        self.executor = core.Executor(p)
        self.program_caches = dict()
        self._closed = False

    def as_lodtensor(self, data):
        """
        Convert numpy.ndarray to Tensor, its only support Tensor without LoD information.
        For higher dimensional sequence data, please use LoDTensor directly.

        Examples:
            >>> import paddle.fluid as fluid
            >>> exe = fluid.executor(fluid.CPUPlace())
            >>> data = np.array(size=(100, 200, 300))
            >>> np_outs = map(lambda x: exe.as_lodtensor(x), data)
            >>>     ...

        Args:
            data(numpy.ndarray): a instance of array

        Returns:
            LoDTensor
        """
        if isinstance(data, list):
            raise RuntimeError("Some of your feed data hold LoD information. \
                They can not be completely cast from a list of Python \
                ndarray to LoDTensor. Please convert data to LoDTensor \
                directly before feeding the data.\
                ")
        # single tensor case
        tensor = core.LoDTensor()
        tensor.set(data, self.place)
        return tensor

    def _get_program_cache(self, program_cache_key):
        return self.program_caches.get(program_cache_key, None)

    def _add_program_cache(self, program_cache_key, program):
        self.program_caches[program_cache_key] = program

    def _add_feed_fetch_ops(self, program, feed, fetch_list, feed_var_name,
                            fetch_var_name):
        tmp_program = program.clone()

        global_block = tmp_program.global_block()

        if feed_var_name in global_block.vars:
            feed_var = global_block.var(feed_var_name)
        else:
            feed_var = global_block.create_var(
                name=feed_var_name,
                type=core.VarDesc.VarType.FEED_MINIBATCH,
                persistable=True)

        if fetch_var_name in global_block.vars:
            fetch_var = global_block.var(fetch_var_name)
        else:
            fetch_var = global_block.create_var(
                name=fetch_var_name,
                type=core.VarDesc.VarType.FETCH_LIST,
                persistable=True)

        # prepend feed operators
        if not has_feed_operators(global_block, feed, feed_var_name):
            for i, name in enumerate(feed):
                out = global_block.var(name)
                global_block._prepend_op(
                    type='feed',
                    inputs={'X': [feed_var]},
                    outputs={'Out': [out]},
                    attrs={'col': i})

        # append fetch_operators
        if not has_fetch_operators(global_block, fetch_list, fetch_var_name):
            for i, var in enumerate(fetch_list):
                assert isinstance(var, Variable) or isinstance(var, str), (
                    "Wrong type for fetch_list[%s]: %s" % (i, type(var)))
                global_block.append_op(
                    type='fetch',
                    inputs={'X': [var]},
                    outputs={'Out': [fetch_var]},
                    attrs={'col': i})

        return tmp_program

    def _feed_data(self, program, feed, feed_var_name, scope):
        # feed var to framework
        for op in program.global_block().ops:
            if op.desc.type() == 'feed':
                feed_target_name = op.desc.output('Out')[0]
                cur_feed = feed[feed_target_name]
                if not isinstance(cur_feed, core.LoDTensor):
                    cur_feed = self.as_lodtensor(cur_feed)
                idx = op.desc.attr('col')
                core.set_feed_variable(scope, cur_feed, feed_var_name, idx)
            else:
                break

    def _fetch_data(self, fetch_list, fetch_var_name, scope):
        outs = [
            core.get_fetch_variable(scope, fetch_var_name, i)
            for i in range(len(fetch_list))
        ]
        return outs

    def close(self):
        """
        Close this executor.

        You can no long use this executor after calling this method.
        For the distributed training, this method would free the resource on PServers related to
        the current Trainer.

        Example:
            >>> cpu = core.CPUPlace()
            >>> exe = Executor(cpu)
            >>> ...
            >>> exe.close()
        """
        if not self._closed:
            self.executor.close()
            self._closed = True

    def run(self,
            program=None,
            feed=None,
            fetch_list=None,
            feed_var_name='feed',
            fetch_var_name='fetch',
            scope=None,
            return_numpy=True,
            use_program_cache=False):
        """
        Run program by this Executor. Feed data by feed map, fetch result by fetch_list.
        Python executor takes a program, add feed operators and fetch operators to this program according
        to feed map and fetch_list. Feed map provides input data for the program. fetch_list provides
        the variables(or names) that user want to get after program run.

        Note: the executor will run all
        operators in the program but not only the operators dependent by the fetch_list

        Args:
            program(Program): the program that need to run, if not provied, then default_main_program will be used.
            feed(dict): feed variable map, e.g. {"image": ImageData, "label": LableData}
            fetch_list(list): a list of variable or variable names that user want to get, run will return them according to this list.
            feed_var_name(str): the name for the input variable of feed Operator.
            fetch_var_name(str): the name for the output variable of fetch Operator.
            scope(Scope): the scope used to run this program, you can switch it to different scope. default is global_scope
            return_numpy(bool): if convert the fetched tensor to numpy
            use_program_cache(bool): set use_program_cache to true if program not changed compare to the last step.

        Returns:

            list(numpy.array): fetch result according to fetch_list.


        Examples:

            >>> data = layers.data(name='X', shape=[1], dtype='float32')
            >>> hidden = layers.fc(input=data, size=10)
            >>> layers.assign(hidden, out)
            >>> loss = layers.mean(out)
            >>> adam = fluid.optimizer.Adam()
            >>> adam.minimize(loss)

            >>> cpu = core.CPUPlace()
            >>> exe = Executor(cpu)
            >>> exe.run(default_startup_program())

            >>> x = numpy.random.random(size=(10, 1)).astype('float32')
            >>> outs = exe.run(
            >>>     feed={'X': x},
            >>>     fetch_list=[loss.name])
        """

        if self._closed:
            raise RuntimeError("Attempted to use a closed Executor")

        if feed is None:
            feed = {}
        if not isinstance(feed, dict):
            raise TypeError(
                "feed requires dict as its Parameter. But you passed in %s" %
                (type(feed)))
        if fetch_list is None:
            fetch_list = []
        if program is None:
            program = default_main_program()

        if not isinstance(program, Program):
            raise TypeError(
                "Executor requires Program as its Parameter. But you passed in %s"
                % (type(program)))

        if scope is None:
            scope = global_scope()

        cache_key = _get_program_cache_key(feed, fetch_list)
        if use_program_cache:
            cached_program = self._get_program_cache(cache_key)
            if cached_program is None:
                cached_program = self._add_feed_fetch_ops(
                    program=program,
                    feed=feed,
                    fetch_list=fetch_list,
                    feed_var_name=feed_var_name,
                    fetch_var_name=fetch_var_name)
                self._add_program_cache(cache_key, cached_program)
            program = cached_program
        else:
            self.program_caches.pop(cache_key, None)
            program = self._add_feed_fetch_ops(
                program=program,
                feed=feed,
                fetch_list=fetch_list,
                feed_var_name=feed_var_name,
                fetch_var_name=fetch_var_name)

        self._feed_data(program, feed, feed_var_name, scope)
        self.executor.run(program.desc, scope, 0, True, True)
        outs = self._fetch_data(fetch_list, fetch_var_name, scope)
        if return_numpy:
            outs = as_numpy(outs)
        return outs
