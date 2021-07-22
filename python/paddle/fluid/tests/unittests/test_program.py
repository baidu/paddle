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

from __future__ import print_function
import unittest

from paddle.fluid.framework import Program, default_main_program, program_guard, grad_var_name
import paddle.fluid.layers as layers
import paddle.fluid as fluid

main_program = default_main_program()


class TestProgram(unittest.TestCase):
    def test_program(self):
        b = main_program.current_block()
        self.assertEqual(-1, b.parent_idx)
        self.assertEqual(0, b.idx)

        b = main_program._create_block()
        self.assertEqual(1, b.idx)
        self.assertEqual(0, b.parent_idx)

        b = main_program._create_block()
        self.assertEqual(2, b.idx)
        self.assertEqual(1, b.parent_idx)

        main_program._rollback()

        b = main_program.current_block()
        self.assertEqual(1, b.idx)
        self.assertEqual(0, b.parent_idx)

        b = main_program._create_block()
        self.assertEqual(3, b.idx)
        self.assertEqual(1, b.parent_idx)

        main_program._rollback()
        b = main_program.current_block()
        self.assertEqual(1, b.idx)
        self.assertEqual(0, b.parent_idx)

    def test_program_clone(self):
        prog = Program()

        x = prog.global_block().create_var(
            name='X', shape=[1000, 784], dtype='float32')

        y = prog.global_block().create_var(
            name='Y', shape=[784, 100], dtype='float32')
        out = prog.global_block().create_var(name='Out', dtype='float32')
        prog.global_block().append_op(
            type="mul", inputs={'X': [x],
                                'Y': [y]}, outputs={'Out': [out]})

        # FIXME(yuyang18): We manual compare the output string, since the order
        # of variable could be changed.
        print(prog)
        print(prog.clone())

    def test_parse_program_from_string(self):
        prog = Program()

        x = prog.global_block().create_var(
            name='X', shape=[1000, 784], dtype='float32')

        y = prog.global_block().create_var(
            name='Y', shape=[784, 100], dtype='float32')
        out = prog.global_block().create_var(name='Out', dtype='float32')
        prog.global_block().append_op(
            type="mul", inputs={'X': [x],
                                'Y': [y]}, outputs={'Out': [out]})

        binary_str = prog.desc.serialize_to_string()
        prog_restored = Program.parse_from_string(binary_str)

        print(prog)
        print(prog_restored)

    def test_program_clone_with_parameter(self):
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            d = layers.data(name='x', shape=[784], dtype='float32')
            hidden = layers.fc(input=d, size=100)
            layers.fc(input=hidden, size=100)

        new_program = main_program.clone()
        self.assertNotEqual(0, len(new_program.blocks[0].all_parameters()))

    def test_program_inference_optimize(self):
        def net():
            reader = fluid.layers.py_reader(
                capacity=10,
                shapes=[[-1, 10], [-1, 1]],
                lod_levels=[0, 0],
                dtypes=['float32', 'int64'],
                use_double_buffer=True)
            in_data, label = fluid.layers.read_file(reader)
            predict_label = fluid.layers.fc(in_data, size=2, act='softmax')
            loss = fluid.layers.mean(
                fluid.layers.cross_entropy(
                    input=predict_label, label=label))

            optimizer = fluid.optimizer.Adam()
            optimizer.minimize(loss)

        startup_program = fluid.Program()
        main_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            net()
        no_read_program = main_program._inference_optimize()
        keep_read_program = main_program._inference_optimize(
            prune_read_op=False)
        no_read_ops = no_read_program.global_block().ops
        keep_read_ops = keep_read_program.global_block().ops
        self.assertEqual(len(keep_read_ops) - len(no_read_ops), 2)
        self.assertEqual(keep_read_ops[0].type, 'create_double_buffer_reader')
        self.assertEqual(keep_read_ops[1].type, 'read')

        for i in range(len(no_read_ops)):
            self.assertEqual(no_read_ops[i].type, keep_read_ops[i + 2].type)

    def test_program_all_parameters(self):
        program = fluid.default_main_program()
        data = fluid.data(name='x', shape=[None, 13], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = fluid.layers.mean(hidden)
        fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

        # NOTE: here the parameters are fc_0.w_0 and fc_0.b_0
        param_list = program.all_parameters()
        self.assertEqual(len(param_list), 2)
        self.assertEqual(param_list[0].name, "fc_0.w_0")
        self.assertEqual(param_list[1].name, "fc_0.b_0")

    def test_prune_with_input_type_error(self):
        program = fluid.default_main_program()
        feed_var_names = [2, 3, 4]
        self.assertRaises(ValueError, program._prune_with_input, feed_var_names,
                          [])

    def test_random_seed_error(self):
        program = fluid.default_main_program()
        with self.assertRaises(ValueError):
            program.random_seed = "seed"

    def test_copy_info_from_error(self):
        program = fluid.default_main_program()
        self.assertRaises(TypeError, program._copy_param_info_from, "program")
        self.assertRaises(TypeError, program._copy_dist_param_info_from,
                          "program")

    def test_remove_training_info(self):
        def net():
            reader = fluid.layers.py_reader(
                capacity=10,
                shapes=[[-1, 10], [-1, 1]],
                lod_levels=[0, 0],
                dtypes=['float32', 'int64'],
                use_double_buffer=True)
            in_data, label = fluid.layers.read_file(reader)
            predict_label = fluid.layers.fc(in_data, size=2, act='softmax')
            loss = fluid.layers.mean(
                fluid.layers.cross_entropy(
                    input=predict_label, label=label))

            optimizer = fluid.optimizer.Adam()
            optimizer.minimize(loss)

        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            net()

        removed_program = main_program._remove_training_info()

        for i in range(removed_program.num_blocks):
            block = removed_program.block(i)
            for var in block.desc.all_vars():
                self.assertFalse(var.has_is_parameter())
                self.assertFalse(var.has_stop_gradient())


if __name__ == '__main__':
    unittest.main()
