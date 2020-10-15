#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.contrib.mixed_precision as mixed_precision
from .meta_optimizer_base import MetaOptimizerBase


class AMPOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(AMPOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.wrapped_opt = None
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = [
            "LarsOptimizer",
            "LambOptimizer",
            "RecomputeOptimizer",
            "GradientMergeOptimizer",
            "GraphExecutionOptimizer",
        ]
        self.meta_optimizers_black_list = ["DGCOptimizer"]

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(AMPOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)

    def _init_wrapped_opt(self):
        if self.wrapped_opt is not None:
            return

        config = self.user_defined_strategy.amp_configs

        custom_white_list = set(config['custom_white_list'])
        custom_black_list = set(config['custom_black_list'])
        custom_black_varnames = set(config['custom_black_varnames'])
        amp_lists = mixed_precision.AutoMixedPrecisionLists(
            custom_white_list, custom_black_list, custom_black_varnames)

        self.wrapped_opt = mixed_precision.decorate(
            self.inner_opt, amp_lists, config['init_loss_scaling'],
            config['incr_every_n_steps'], config['decr_every_n_nan_or_inf'],
            config['incr_ratio'], config['decr_ratio'],
            config['use_dynamic_loss_scaling'])

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False

        if self.user_defined_strategy.amp:
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.amp = False
        dist_strategy.amp_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        dist_strategy.amp = True
        dist_strategy.amp_configs = {
            "init_loss_scaling": 32768.0,
            "incr_every_n_steps": 1000,
            "decr_every_n_nan_or_inf": 2,
            "incr_ratio": 2.0,
            "decr_ratio": 8.0,
            "use_dynamic_loss_scaling": True
        }

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        # maybe inner_opt of other meta optimizer
        self._init_wrapped_opt()
        return self.wrapped_opt.backward(loss, startup_program, parameter_list,
                                         no_grad_set, callbacks)

    def apply_gradients(self, params_grads):
        return self.wrapped_opt.apply_gradients(params_grads=params_grads)

    def apply_optimize(self, loss, startup_program, params_grads):
        return self.wrapped_opt.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        self._init_wrapped_opt()
        optimize_ops, params_grads = \
            self.wrapped_opt.minimize(loss, startup_program,
                                  parameter_list, no_grad_set)
        return optimize_ops, params_grads
