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
# limitations under the License.

from __future__ import print_function

import os
import sys
import time
import subprocess
import unittest
import numpy as np
import paddle


class TestDirectory(unittest.TestCase):
    def get_import_command(self, module):
        paths = module.split('.')
        if len(paths) == 1:
            return 'import {}'.format(module)
        package = '.'.join(paths[:-1])
        func = paths[-1]
        cmd = 'from {} import {}'.format(package, func)
        return cmd

    def test_new_directory(self):
        new_directory = [
            'paddle.enable_static', 'paddle.disable_static',
            'paddle.in_dynamic_mode', 'paddle.to_tensor', 'paddle.grad',
            'paddle.no_grad', 'paddle.static.save', 'paddle.static.load',
            'paddle.distributed.ParallelEnv',
            'paddle.distributed.prepare_context', 'paddle.DataParallel',
            'paddle.jit', 'paddle.jit.TracedLayer', 'paddle.jit.to_static',
            'paddle.jit.ProgramTranslator', 'paddle.jit.TranslatedLayer',
            'paddle.jit.save', 'paddle.jit.load',
            'paddle.optimizer.lr.LRScheduler', 'paddle.optimizer.lr.NoamDecay',
            'paddle.optimizer.lr.PiecewiseDecay',
            'paddle.optimizer.lr.NaturalExpDecay',
            'paddle.optimizer.lr.ExponentialDecay',
            'paddle.optimizer.lr.InverseTimeDecay',
            'paddle.optimizer.lr.PolynomialDecay',
            'paddle.optimizer.lr.CosineAnnealingDecay',
            'paddle.optimizer.lr.MultiStepDecay',
            'paddle.optimizer.lr.StepDecay', 'paddle.optimizer.lr.LambdaDecay',
            'paddle.optimizer.lr.ReduceOnPlateau',
            'paddle.optimizer.lr.LinearWarmup', 'paddle.static.Executor',
            'paddle.static.global_scope', 'paddle.static.scope_guard',
            'paddle.static.append_backward', 'paddle.static.gradients',
            'paddle.static.BuildStrategy', 'paddle.static.CompiledProgram',
            'paddle.static.ExecutionStrategy',
            'paddle.static.default_main_program',
            'paddle.static.default_startup_program', 'paddle.static.Program',
            'paddle.static.name_scope', 'paddle.static.program_guard',
            'paddle.static.Print', 'paddle.static.py_func',
            'paddle.static.ParallelExecutor',
            'paddle.static.WeightNormParamAttr', 'paddle.static.nn.fc',
            'paddle.static.nn.batch_norm',
            'paddle.static.nn.bilinear_tensor_product',
            'paddle.static.nn.conv2d', 'paddle.static.nn.conv2d_transpose',
            'paddle.static.nn.conv3d', 'paddle.static.nn.conv3d_transpose',
            'paddle.static.nn.create_parameter',
            'paddle.static.nn.crf_decoding', 'paddle.static.nn.data_norm',
            'paddle.static.nn.deform_conv2d', 'paddle.static.nn.group_norm',
            'paddle.static.nn.instance_norm', 'paddle.static.nn.layer_norm',
            'paddle.static.nn.multi_box_head', 'paddle.static.nn.nce',
            'paddle.static.nn.prelu', 'paddle.static.nn.row_conv',
            'paddle.static.nn.spectral_norm', 'paddle.static.nn.embedding'
        ]

        import_file = 'run_import_modules.py'

        with open(import_file, "w") as wb:
            for module in new_directory:
                run_cmd = self.get_import_command(module)
                wb.write("{}\n".format(run_cmd))

        _python = sys.executable

        ps_cmd = "{} {}".format(_python, import_file)
        ps_proc = subprocess.Popen(
            ps_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = ps_proc.communicate()

        self.assertFalse("Error" in str(stderr),
                         "ErrorMessage:\n{}".format(bytes.decode(stderr)))

    def test_old_directory(self):
        old_directory = [
            'paddle.enable_imperative', 'paddle.disable_imperative',
            'paddle.in_imperative_mode', 'paddle.imperative.to_variable',
            'paddle.imperative.enable', 'paddle.imperative.guard',
            'paddle.imperative.grad', 'paddle.imperative.no_grad',
            'paddle.imperative.save', 'paddle.imperative.load',
            'paddle.imperative.ParallelEnv',
            'paddle.imperative.prepare_context',
            'paddle.imperative.DataParalell', 'paddle.imperative.jit',
            'paddle.imperative.TracedLayer', 'paddle.imperative.declarative',
            'paddle.imperative.ProgramTranslator',
            'paddle.imperative.TranslatedLayer', 'paddle.imperative.jit.save',
            'paddle.imperative.jit.load', 'paddle.imperative.NoamDecay'
            'paddle.imperative.PiecewiseDecay',
            'paddle.imperative.NaturalExpDecay',
            'paddle.imperative.ExponentialDecay',
            'paddle.imperative.InverseTimeDecay',
            'paddle.imperative.PolynomialDecay',
            'paddle.imperative.CosineDecay', 'paddle.Executor',
            'paddle.global_scope', 'paddle.scope_guard',
            'paddle.append_backward', 'paddle.gradients',
            'paddle.BuildStrategy', 'paddle.CompiledProgram',
            'paddle.ExecutionStrategy', 'paddle.name_scope',
            'paddle.program_guard', 'paddle.Print', 'paddle.py_func',
            'paddle.ParallelExecutor', 'paddle.default_main_program',
            'paddle.default_startup_program', 'paddle.Program',
            'paddle.WeightNormParamAttr', 'paddle.declarative.fc',
            'paddle.declarative.batch_norm',
            'paddle.declarative.bilinear_tensor_product',
            'paddle.declarative.conv2d', 'paddle.declarative.conv2d_transpose',
            'paddle.declarative.conv3d', 'paddle.declarative.conv3d_transpose',
            'paddle.declarative.create_parameter',
            'paddle.declarative.crf_decoding', 'paddle.declarative.data_norm',
            'paddle.declarative.deformable_conv',
            'paddle.declarative.group_norm', 'paddle.declarative.hsigmoid',
            'paddle.declarative.instance_norm', 'paddle.declarative.layer_norm',
            'paddle.declarative.multi_box_head', 'paddle.declarative.nce',
            'paddle.declarative.prelu', 'paddle.declarative.row_conv',
            'paddle.declarative.spectral_norm', 'paddle.declarative.embedding'
        ]

        import_file = 'run_old_import_modules.py'

        with open(import_file, "w") as wb:
            cmd_context_count = """
count = 0
err_module = ""
"""
            wb.write(cmd_context_count)
            for module in old_directory:
                run_cmd = self.get_import_command(module)
                cmd_context_loop_template = """
try:
    {run_cmd}
except:
    count += 1
else:
    err_module = "{module}"
"""
                cmd_context_loop = cmd_context_loop_template.format(
                    run_cmd=run_cmd, module=module)
                wb.write(cmd_context_loop)
            cmd_context_print_template = """
if count != {len_old_directory}:
    print("Error: Module " + err_module + " should not be imported")
"""
            cmd_context_print = cmd_context_print_template.format(
                len_old_directory=str(len(old_directory)))
            wb.write(cmd_context_print)

        _python = sys.executable

        ps_cmd = "{} {}".format(_python, import_file)
        ps_proc = subprocess.Popen(
            ps_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = ps_proc.communicate()

        self.assertFalse("Error" in str(stdout), bytes.decode(stdout))


if __name__ == '__main__':
    unittest.main()
