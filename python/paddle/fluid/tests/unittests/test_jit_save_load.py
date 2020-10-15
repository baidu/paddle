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

import os
import pickle
import unittest
import numpy as np
import paddle
from paddle.static import InputSpec
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph import declarative, ProgramTranslator
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX, INFER_PARAMS_INFO_SUFFIX

BATCH_SIZE = 32
BATCH_NUM = 10
SEED = 10


def random_batch_reader(input_size, label_size):
    def _get_random_inputs_and_labels(input_size, label_size):
        np.random.seed(SEED)
        input = np.random.random(size=input_size).astype('float32')
        label = np.random.random(size=label_size).astype('int64')
        return input, label

    def __reader__():
        for _ in range(BATCH_NUM):
            batch_input, batch_label = _get_random_inputs_and_labels(
                [BATCH_SIZE, input_size], [BATCH_SIZE, label_size])
            yield batch_input, batch_label

    return __reader__


class LinearNet(fluid.dygraph.Layer):
    def __init__(self, in_size, out_size):
        super(LinearNet, self).__init__()
        self._linear = Linear(in_size, out_size)

    @declarative
    def forward(self, x):
        return self._linear(x)


class LinearNetWithInputSpec(fluid.dygraph.Layer):
    def __init__(self, in_size, out_size):
        super(LinearNetWithInputSpec, self).__init__()
        self._linear = Linear(in_size, out_size)

    @declarative(input_spec=[InputSpec(shape=[None, 784], dtype='float32')])
    def forward(self, x):
        return self._linear(x)


class LinearNetNotDeclarative(fluid.dygraph.Layer):
    def __init__(self, in_size, out_size):
        super(LinearNetNotDeclarative, self).__init__()
        self._linear = Linear(in_size, out_size)

    def forward(self, x):
        return self._linear(x)


class LinerNetWithLabel(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super(LinerNetWithLabel, self).__init__()
        self._linear = Linear(in_size, out_size)

    @declarative(input_spec=[
        InputSpec(
            shape=[None, 784], dtype='float32', name="image"), InputSpec(
                shape=[None, 1], dtype='int64', name="label")
    ])
    def forward(self, x, label):
        out = self._linear(x)
        loss = fluid.layers.cross_entropy(out, label)
        avg_loss = fluid.layers.mean(loss)
        return out, avg_loss


class LinearNetReturnLoss(fluid.dygraph.Layer):
    def __init__(self, in_size, out_size):
        super(LinearNetReturnLoss, self).__init__()
        self._linear = Linear(in_size, out_size)

    @declarative
    def forward(self, x):
        y = self._linear(x)
        z = self._linear(y)
        loss = fluid.layers.mean(z)
        return z, loss


class LinearNetMultiInput(fluid.dygraph.Layer):
    def __init__(self, in_size, out_size):
        super(LinearNetMultiInput, self).__init__()
        self._linear1 = Linear(in_size, out_size)
        self._linear2 = Linear(in_size, out_size)

    @declarative(input_spec=[
        InputSpec(
            [None, 8], dtype='float32'), InputSpec(
                [None, 8], dtype='float32')
    ])
    def forward(self, x, y):
        x_out = self._linear1(x)
        y_out = self._linear2(y)
        loss = fluid.layers.mean(x_out + y_out)
        return x_out, y_out, loss


class MultiLoadingLinearNet(fluid.dygraph.Layer):
    def __init__(self, size, model_path):
        super(MultiLoadingLinearNet, self).__init__()
        self._linear = Linear(size, size)
        self._load_linear1 = paddle.jit.load(model_path)
        self._load_linear2 = paddle.jit.load(model_path)

    @declarative
    def forward(self, x):
        tmp1 = self._linear(x)
        tmp2 = self._load_linear1(tmp1)
        tmp3 = self._load_linear2(tmp2)
        y = self._linear(tmp3)
        return y


class LinearNetReturnHidden(fluid.dygraph.Layer):
    def __init__(self, in_size, out_size):
        super(LinearNetReturnHidden, self).__init__()
        self._linear_1 = Linear(in_size, out_size)
        self._linear_2 = Linear(in_size, out_size)

    @declarative
    def forward(self, x):
        y = self._linear_1(x)
        z = self._linear_2(y)
        loss = fluid.layers.mean(z)
        return y, loss


class EmptyLayer(paddle.nn.Layer):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    @paddle.jit.to_static
    def forward(self, x):
        return x


class NoParamLayer(paddle.nn.Layer):
    def __init__(self):
        super(NoParamLayer, self).__init__()

    @paddle.jit.to_static
    def forward(self, x, y):
        return x + y


def train(layer, input_size=784, label_size=1):
    # create optimizer
    sgd = fluid.optimizer.SGDOptimizer(
        learning_rate=0.01, parameter_list=layer.parameters())
    # create data loader
    train_loader = fluid.io.DataLoader.from_generator(capacity=5)
    train_loader.set_batch_generator(
        random_batch_reader(input_size, label_size))
    # train
    for data in train_loader():
        img, label = data
        label.stop_gradient = True

        cost = layer(img)

        loss = fluid.layers.cross_entropy(cost, label)
        avg_loss = fluid.layers.mean(loss)

        avg_loss.backward()
        sgd.minimize(avg_loss)
        layer.clear_gradients()
    return [img], layer, avg_loss


def train_with_label(layer, input_size=784, label_size=1):
    # create optimizer
    sgd = fluid.optimizer.SGDOptimizer(
        learning_rate=0.01, parameter_list=layer.parameters())
    # create data loader
    train_loader = fluid.io.DataLoader.from_generator(capacity=5)
    train_loader.set_batch_generator(
        random_batch_reader(input_size, label_size))
    # train
    for data in train_loader():
        img, label = data
        label.stop_gradient = True

        out, avg_loss = layer(img, label)

        avg_loss.backward()
        sgd.minimize(avg_loss)
        layer.clear_gradients()
    return out


class TestJitSaveLoad(unittest.TestCase):
    def setUp(self):
        self.model_path = "test_jit_save_load/model"
        # enable dygraph mode
        fluid.enable_dygraph()
        # config seed
        paddle.manual_seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def train_and_save_model(self, model_path=None):
        layer = LinearNet(784, 1)
        example_inputs, layer, _ = train(layer)
        final_model_path = model_path if model_path else self.model_path
        orig_input_types = [type(x) for x in example_inputs]
        paddle.jit.save(
            layer=layer, path=final_model_path, input_spec=example_inputs)
        new_input_types = [type(x) for x in example_inputs]
        self.assertEqual(orig_input_types, new_input_types)
        return layer

    def test_save_load(self):
        # train and save model
        train_layer = self.train_and_save_model()
        # load model
        loaded_layer = paddle.jit.load(self.model_path)
        self.load_and_inference(train_layer, loaded_layer)
        self.load_dygraph_state_dict(train_layer)
        self.load_and_finetune(train_layer, loaded_layer)

    def load_and_inference(self, train_layer, infer_layer):
        train_layer.eval()
        infer_layer.eval()
        # inference & compare
        x = fluid.dygraph.to_variable(
            np.random.random((1, 784)).astype('float32'))
        self.assertTrue(
            np.array_equal(train_layer(x).numpy(), infer_layer(x).numpy()))

    def load_and_finetune(self, train_layer, load_train_layer):
        train_layer.train()
        load_train_layer.train()
        # train & compare
        img0, _, train_loss = train(train_layer)
        img1, _, load_train_loss = train(load_train_layer)
        self.assertTrue(
            np.array_equal(train_loss.numpy(), load_train_loss.numpy()))

    def load_dygraph_state_dict(self, train_layer):
        train_layer.eval()
        # construct new model
        new_layer = LinearNet(784, 1)
        orig_state_dict = new_layer.state_dict()
        load_state_dict = paddle.load(self.model_path)
        for structured_name in orig_state_dict:
            self.assertTrue(structured_name in load_state_dict)
        new_layer.set_state_dict(load_state_dict)
        new_layer.eval()
        # inference & compare
        x = fluid.dygraph.to_variable(
            np.random.random((1, 784)).astype('float32'))
        self.assertTrue(
            np.array_equal(train_layer(x).numpy(), new_layer(x).numpy()))

    def test_load_dygraph_no_path(self):
        model_path = "test_jit_save_load.no_path/model_path"
        with self.assertRaises(ValueError):
            model_dict, _ = fluid.dygraph.load_dygraph(model_path)

    def test_jit_load_model_incomplete(self):
        model_path = "test_jit_save_load.remove_variables/model"
        self.train_and_save_model(model_path)
        # remove `.pdiparams`	
        var_path = model_path + INFER_PARAMS_SUFFIX
        os.remove(var_path)
        with self.assertRaises(ValueError):
            paddle.jit.load(model_path)

    def test_jit_load_no_path(self):
        path = "test_jit_save_load.no_path/model_path"
        with self.assertRaises(ValueError):
            loaded_layer = paddle.jit.load(path)


class TestSaveLoadWithInputSpec(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        fluid.enable_dygraph()

    def test_with_input_spec(self):
        net = LinearNetReturnLoss(8, 8)
        # set x.shape = [None, 8]
        net.forward = declarative(
            net.forward, input_spec=[InputSpec(
                [None, 8], name='x')])

        model_path = "input_spec.output_spec/model"
        # check inputs and outputs
        self.assertTrue(len(net.forward.inputs) == 1)
        input_x = net.forward.inputs[0]
        self.assertTrue(input_x.shape == (-1, 8))
        self.assertTrue(input_x.name == 'x')

        # 1. prune loss
        output_spec = net.forward.outputs[:1]
        paddle.jit.save(net, model_path, output_spec=output_spec)

        # 2. load to infer
        infer_layer = paddle.jit.load(model_path)
        x = fluid.dygraph.to_variable(
            np.random.random((4, 8)).astype('float32'))
        pred = infer_layer(x)

    def test_multi_in_out(self):
        net = LinearNetMultiInput(8, 8)

        model_path = "multi_inout.output_spec1/model"
        # 1. check inputs and outputs
        self.assertTrue(len(net.forward.inputs) == 2)
        input_x = net.forward.inputs[0]
        input_y = net.forward.inputs[1]
        self.assertTrue(input_x.shape == (-1, 8))
        self.assertTrue(input_y.shape == (-1, 8))

        # 2. prune loss
        output_spec = net.forward.outputs[:2]
        paddle.jit.save(net, model_path, output_spec=output_spec)

        # 3. load to infer
        infer_layer = paddle.jit.load(model_path)
        x = fluid.dygraph.to_variable(
            np.random.random((4, 8)).astype('float32'))
        y = fluid.dygraph.to_variable(
            np.random.random((4, 8)).astype('float32'))
        # 4. predict
        pred_x, pred_y = infer_layer(x, y)

        # 1. prune y and loss
        model_path = "multi_inout.output_spec2/model"
        output_spec = net.forward.outputs[:1]
        paddle.jit.save(net, model_path, [input_x], output_spec=output_spec)
        # 2. load again
        infer_layer2 = paddle.jit.load(model_path)
        # 3. predict
        pred_xx = infer_layer2(x)

        # 4. assert pred_x == pred_xx
        self.assertTrue(np.allclose(pred_x.numpy(), pred_xx.numpy()))


class TestJitSaveLoadConfig(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        fluid.enable_dygraph()
        # config seed
        paddle.manual_seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def test_output_spec(self):
        train_layer = LinearNetReturnLoss(8, 8)
        adam = fluid.optimizer.AdamOptimizer(
            learning_rate=0.1, parameter_list=train_layer.parameters())
        x = fluid.dygraph.to_variable(
            np.random.random((4, 8)).astype('float32'))
        for i in range(10):
            out, loss = train_layer(x)
            loss.backward()
            adam.minimize(loss)
            train_layer.clear_gradients()

        model_path = "save_load_config.output_spec"
        output_spec = [out]
        paddle.jit.save(
            layer=train_layer,
            path=model_path,
            input_spec=[x],
            output_spec=output_spec)

        train_layer.eval()
        infer_layer = paddle.jit.load(model_path)
        x = fluid.dygraph.to_variable(
            np.random.random((4, 8)).astype('float32'))
        self.assertTrue(
            np.array_equal(train_layer(x)[0].numpy(), infer_layer(x).numpy()))

    def test_save_no_support_config_error(self):
        layer = LinearNet(784, 1)
        path = "no_support_config_test"
        with self.assertRaises(ValueError):
            paddle.jit.save(layer=layer, path=path, model_filename="")

    def test_load_empty_model_filename_error(self):
        path = "error_model_filename_test"
        with self.assertRaises(ValueError):
            paddle.jit.load(path, model_filename="")

    def test_load_empty_params_filename_error(self):
        path = "error_params_filename_test"
        with self.assertRaises(ValueError):
            paddle.jit.load(path, params_filename="")

    def test_load_with_no_support_config(self):
        path = "no_support_config_test"
        with self.assertRaises(ValueError):
            paddle.jit.load(path, separate_params=True)


class TestJitMultipleLoading(unittest.TestCase):
    def setUp(self):
        self.linear_size = 4
        self.model_path = "jit_multi_load/model"
        # enable dygraph mode
        fluid.enable_dygraph()
        # config seed
        paddle.manual_seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        # train and save base model
        self.train_and_save_orig_model()

    def train_and_save_orig_model(self):
        layer = LinearNet(self.linear_size, self.linear_size)
        example_inputs, layer, _ = train(layer, self.linear_size, 1)
        paddle.jit.save(
            layer=layer, path=self.model_path, input_spec=example_inputs)

    def test_load_model_retransform_inference(self):
        multi_loaded_layer = MultiLoadingLinearNet(self.linear_size,
                                                   self.model_path)
        state_dict = multi_loaded_layer.state_dict()
        name_set = set()
        for _, var in state_dict.items():
            self.assertTrue(var.name not in name_set)
            name_set.add(var.name)


class TestJitPruneModelAndLoad(unittest.TestCase):
    def setUp(self):
        self.linear_size = 4
        self.model_path = "jit_prune_model_and_load/model"
        # enable dygraph mode
        fluid.enable_dygraph()
        # config seed
        paddle.manual_seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def train_and_save(self):
        train_layer = LinearNetReturnHidden(8, 8)
        adam = fluid.optimizer.AdamOptimizer(
            learning_rate=0.1, parameter_list=train_layer.parameters())
        x = fluid.dygraph.to_variable(
            np.random.random((4, 8)).astype('float32'))
        for i in range(10):
            hidden, loss = train_layer(x)
            loss.backward()
            adam.minimize(loss)
            train_layer.clear_gradients()

        output_spec = [hidden]
        paddle.jit.save(
            layer=train_layer,
            path=self.model_path,
            input_spec=[x],
            output_spec=output_spec)

        return train_layer

    def test_load_pruned_model(self):
        train_layer = self.train_and_save()
        train_layer.eval()

        infer_layer = paddle.jit.load(self.model_path)

        x = fluid.dygraph.to_variable(
            np.random.random((4, 8)).astype('float32'))
        self.assertTrue(
            np.array_equal(train_layer(x)[0].numpy(), infer_layer(x).numpy()))

    def test_load_var_not_in_extra_var_info(self):
        self.train_and_save()

        # chage extra var info
        var_info_path = self.model_path + INFER_PARAMS_INFO_SUFFIX
        with open(var_info_path, 'rb') as f:
            extra_var_info = pickle.load(f)
            extra_var_info.clear()
        with open(var_info_path, 'wb') as f:
            pickle.dump(extra_var_info, f, protocol=2)

        with self.assertRaises(RuntimeError):
            paddle.jit.load(self.model_path)


class TestJitSaveMultiCases(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        fluid.enable_dygraph()
        # config seed
        paddle.manual_seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def verify_inference_correctness(self, layer, model_path, with_label=False):
        layer.eval()
        loaded_layer = paddle.jit.load(model_path)
        loaded_layer.eval()
        # inference & compare
        x = paddle.to_tensor(np.random.random((1, 784)).astype('float32'))
        if with_label:
            y = paddle.to_tensor(np.random.random((1, 1)).astype('int64'))
            pred, _ = layer(x, y)
            pred = pred.numpy()
        else:
            pred = layer(x).numpy()
        loaded_pred = loaded_layer(x).numpy()
        self.assertTrue(
            np.array_equal(pred, loaded_pred),
            msg="Result diff when load and inference:\nlayer result:\n{}\n" \
                "loaded layer result:\n{}".format(pred, loaded_pred))

    def test_no_prune_to_static_after_train(self):
        layer = LinearNet(784, 1)

        train(layer)

        model_path = "test_no_prune_to_static_after_train/model"
        paddle.jit.save(layer, model_path)

        self.verify_inference_correctness(layer, model_path)

    def test_no_prune_to_static_no_train(self):
        layer = LinearNetWithInputSpec(784, 1)

        model_path = "test_no_prune_to_static_no_train/model"
        paddle.jit.save(layer, model_path)

        self.verify_inference_correctness(layer, model_path)

    def test_no_prune_no_to_static_after_train(self):
        layer = LinearNetNotDeclarative(784, 1)

        train(layer)

        model_path = "test_no_prune_no_to_static_after_train/model"
        paddle.jit.save(
            layer,
            model_path,
            input_spec=[InputSpec(
                shape=[None, 784], dtype='float32')])

        self.verify_inference_correctness(layer, model_path)

    def test_no_prune_no_to_static_after_train_with_examples(self):
        layer = LinearNetNotDeclarative(784, 1)

        example_inputs, _, _ = train(layer)

        model_path = "test_no_prune_no_to_static_after_train_with_examples/model"
        paddle.jit.save(layer=layer, path=model_path, input_spec=example_inputs)

        self.verify_inference_correctness(layer, model_path)

    def test_no_prune_no_to_static_no_train(self):
        layer = LinearNetNotDeclarative(784, 1)

        model_path = "test_no_prune_no_to_static_no_train/model"
        paddle.jit.save(
            layer,
            model_path,
            input_spec=[InputSpec(
                shape=[None, 784], dtype='float32')])

        self.verify_inference_correctness(layer, model_path)

    def test_prune_to_static_after_train(self):
        layer = LinerNetWithLabel(784, 1)

        out = train_with_label(layer)

        model_path = "test_prune_to_static_after_train/model"
        paddle.jit.save(
            layer,
            model_path,
            input_spec=[
                InputSpec(
                    shape=[None, 784], dtype='float32', name="image")
            ],
            output_spec=[out])

        self.verify_inference_correctness(layer, model_path, True)

    def test_prune_to_static_no_train(self):
        layer = LinerNetWithLabel(784, 1)

        model_path = "test_prune_to_static_no_train/model"
        # TODO: no train, cannot get output_spec var here
        # now only can use index
        output_spec = layer.forward.outputs[:1]
        paddle.jit.save(
            layer,
            model_path,
            input_spec=[
                InputSpec(
                    shape=[None, 784], dtype='float32', name="image")
            ],
            output_spec=output_spec)

        self.verify_inference_correctness(layer, model_path, True)

    def test_no_prune_input_spec_name_warning(self):
        layer = LinearNetWithInputSpec(784, 1)

        train(layer)

        model_path = "test_no_prune_input_spec_name_warning/model"
        paddle.jit.save(
            layer,
            model_path,
            input_spec=[InputSpec(
                shape=[None, 784], dtype='float32')])
        paddle.jit.save(
            layer,
            model_path,
            input_spec=[
                InputSpec(
                    shape=[None, 784], dtype='float32', name='feed_input')
            ])

        self.verify_inference_correctness(layer, model_path)

    def test_not_prune_output_spec_name_warning(self):
        layer = LinearNet(784, 1)

        train(layer)

        model_path = "test_not_prune_output_spec_name_warning/model"
        out = paddle.to_tensor(np.random.random((1, 1)).astype('float'))
        paddle.jit.save(layer, model_path, output_spec=[out])

        self.verify_inference_correctness(layer, model_path)

    def test_prune_input_spec_name_error(self):
        layer = LinerNetWithLabel(784, 1)

        model_path = "test_prune_input_spec_name_error/model"
        with self.assertRaises(ValueError):
            paddle.jit.save(
                layer,
                model_path,
                input_spec=[InputSpec(
                    shape=[None, 784], dtype='float32')])
        with self.assertRaises(ValueError):
            paddle.jit.save(
                layer,
                model_path,
                input_spec=[
                    InputSpec(
                        shape=[None, 784], dtype='float32', name='feed_input')
                ])

    def test_prune_output_spec_name_error(self):
        layer = LinerNetWithLabel(784, 1)

        train_with_label(layer)

        model_path = "test_prune_to_static_after_train/model"
        out = paddle.to_tensor(np.random.random((1, 1)).astype('float'))
        with self.assertRaises(ValueError):
            paddle.jit.save(
                layer,
                model_path,
                input_spec=[
                    InputSpec(
                        shape=[None, 784], dtype='float32', name="image")
                ],
                output_spec=[out])


class TestJitSaveLoadEmptyLayer(unittest.TestCase):
    def setUp(self):
        self.model_path = "jit_save_load_empty_layer/model"
        # enable dygraph mode
        paddle.disable_static()

    def test_save_load_empty_layer(self):
        layer = EmptyLayer()
        x = paddle.to_tensor(np.random.random((10)).astype('float32'))
        out = layer(x)
        paddle.jit.save(layer, self.model_path)
        load_layer = paddle.jit.load(self.model_path)
        load_out = load_layer(x)
        self.assertTrue(np.array_equal(out, load_out))


class TestJitSaveLoadNoParamLayer(unittest.TestCase):
    def setUp(self):
        self.model_path = "jit_save_load_no_param_layer/model"
        # enable dygraph mode
        paddle.disable_static()

    def test_save_load_no_param_layer(self):
        layer = NoParamLayer()
        x = paddle.to_tensor(np.random.random((5)).astype('float32'))
        y = paddle.to_tensor(np.random.random((5)).astype('float32'))
        out = layer(x, y)
        paddle.jit.save(layer, self.model_path)
        load_layer = paddle.jit.load(self.model_path)
        load_out = load_layer(x, y)
        self.assertTrue(np.array_equal(out, load_out))


if __name__ == '__main__':
    unittest.main()
