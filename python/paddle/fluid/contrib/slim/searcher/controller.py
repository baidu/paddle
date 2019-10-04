#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
"""The controller used to search hyperparameters or neural architecture"""

import numpy as np
import copy
import math
import logging
from ....log_helper import get_logger

__all__ = ['EvolutionaryController', 'SAController']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class EvolutionaryController(object):
    """Abstract controller for all evolutionary searching method.
    """

    def __init__(self, *args, **kwargs):
        pass

    def update(self, tokens, reward):
        """Update the status of controller according current tokens and reward.
        Args:
            tokens(list<int>): A solution of searching task.
            reward(list<int>): The reward of tokens.
        """
        raise NotImplementedError('Abstract method.')

    def reset(self, range_table, constrain_func=None):
        """Reset the controller.
        Args:
            range_table(list<int>): It is used to define the searching space of controller.
                                    The tokens[i] generated by controller should be in [0, range_table[i]).
            constrain_func(function): It is used to check whether tokens meet the constraint.
                                     None means there is no constraint. Default: None.
        """
        raise NotImplementedError('Abstract method.')

    def next_tokens(self):
        """Generate new tokens.
        """
        raise NotImplementedError('Abstract method.')


class SAController(EvolutionaryController):
    """Simulated annealing controller."""

    def __init__(self,
                 range_table=None,
                 reduce_rate=0.85,
                 init_temperature=1024,
                 max_iter_number=300):
        """Initialize.
        Args:
            range_table(list<int>): Range table.
            reduce_rate(float): The decay rate of temperature.
            init_temperature(float): Init temperature.
            max_iter_number(int): max iteration number.
        """
        super(SAController, self).__init__()
        self._range_table = range_table
        self._reduce_rate = reduce_rate
        self._init_temperature = init_temperature
        self._max_iter_number = max_iter_number
        self._reward = -1
        self._tokens = None
        self._max_reward = -1
        self._best_tokens = None
        self._iter = 0

    def __getstate__(self):
        d = {}
        for key in self.__dict__:
            if key != "_constrain_func":
                d[key] = self.__dict__[key]
        return d

    def reset(self, range_table, init_tokens, constrain_func=None):
        """
        Reset the status of current controller.
        Args:
            range_table(list<int>): The range of value in each position of tokens generated by current controller. The range of tokens[i] is [0, range_table[i]).
            init_tokens(list<int>): The initial tokens.
            constrain_func(function): The callback function used to check whether the tokens meet constraint. None means there is no constraint. Default: None.
        """
        self._range_table = range_table
        self._constrain_func = constrain_func
        self._tokens = init_tokens
        self._iter = 0

    def update(self, tokens, reward):
        """
        Update the controller according to latest tokens and reward.
        Args:
            tokens(list<int>): The tokens generated in last step.
            reward(float): The reward of tokens.
        """
        self._iter += 1
        temperature = self._init_temperature * self._reduce_rate**self._iter
        if (reward > self._reward) or (np.random.random() <= math.exp(
            (reward - self._reward) / temperature)):
            self._reward = reward
            self._tokens = tokens
        if reward > self._max_reward:
            self._max_reward = reward
            self._best_tokens = tokens
        _logger.info("iter: {}; max_reward: {}; best_tokens: {}".format(
            self._iter, self._max_reward, self._best_tokens))
        _logger.info("current_reward: {}; current tokens: {}".format(
            self._reward, self._tokens))

    def next_tokens(self, control_token=None):
        """
        Get next tokens.
        """
        if control_token:
            tokens = control_token[:]
        else:
            tokens = self._tokens
        new_tokens = tokens[:]
        index = int(len(self._range_table) * np.random.random())
        new_tokens[index] = (
            new_tokens[index] + np.random.randint(self._range_table[index] - 1)
            + 1) % self._range_table[index]
        _logger.info("change index[{}] from {} to {}".format(index, tokens[
            index], new_tokens[index]))
        if self._constrain_func is None:
            return new_tokens
        for _ in range(self._max_iter_number):
            if not self._constrain_func(new_tokens):
                index = int(len(self._range_table) * np.random.random())
                new_tokens = tokens[:]
                new_tokens[index] = np.random.randint(self._range_table[index])
            else:
                break
        return new_tokens
