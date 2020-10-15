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

import os
import numbers

from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.utils import try_import

from .progressbar import ProgressBar

__all__ = ['Callback', 'ProgBarLogger', 'ModelCheckpoint', 'VisualDL']


def config_callbacks(callbacks=None,
                     model=None,
                     batch_size=None,
                     epochs=None,
                     steps=None,
                     log_freq=2,
                     verbose=2,
                     save_freq=1,
                     save_dir=None,
                     metrics=None,
                     mode='train'):
    cbks = callbacks or []
    cbks = cbks if isinstance(cbks, (list, tuple)) else [cbks]
    if not any(isinstance(k, ProgBarLogger) for k in cbks) and verbose:
        cbks = [ProgBarLogger(log_freq, verbose=verbose)] + cbks

    if not any(isinstance(k, ModelCheckpoint) for k in cbks):
        cbks = cbks + [ModelCheckpoint(save_freq, save_dir)]

    cbk_list = CallbackList(cbks)
    cbk_list.set_model(model)
    metrics = metrics or [] if mode != 'test' else []
    params = {
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps,
        'verbose': verbose,
        'metrics': metrics,
    }
    cbk_list.set_params(params)
    return cbk_list


class CallbackList(object):
    def __init__(self, callbacks=None):
        # copy
        self.callbacks = [c for c in callbacks]
        self.params = {}
        self.model = None

    def append(self, callback):
        self.callbacks.append(callback)

    def __iter__(self):
        return iter(self.callbacks)

    def set_params(self, params):
        for c in self.callbacks:
            c.set_params(params)

    def set_model(self, model):
        for c in self.callbacks:
            c.set_model(model)

    def _call(self, name, *args):
        for c in self.callbacks:
            func = getattr(c, name)
            func(*args)

    def _check_mode(self, mode):
        assert mode in ['train', 'eval', 'test'], \
            'mode should be train, eval or test'

    def on_begin(self, mode, logs=None):
        self._check_mode(mode)
        name = 'on_{}_begin'.format(mode)
        self._call(name, logs)

    def on_end(self, mode, logs=None):
        self._check_mode(mode)
        name = 'on_{}_end'.format(mode)
        self._call(name, logs)

    def on_epoch_begin(self, epoch=None, logs=None):
        self._call('on_epoch_begin', epoch, logs)

    def on_epoch_end(self, epoch=None, logs=None):
        self._call('on_epoch_end', epoch, logs)

    def on_batch_begin(self, mode, step=None, logs=None):
        self._check_mode(mode)
        name = 'on_{}_batch_begin'.format(mode)
        self._call(name, step, logs)

    def on_batch_end(self, mode, step=None, logs=None):
        self._check_mode(mode)
        name = 'on_{}_batch_end'.format(mode)
        self._call(name, step, logs)


class Callback(object):
    """
    Base class used to build new callbacks.

    Examples:

        .. code-block:: python
            
            import paddle

            # build a simple model checkpoint callback
            class ModelCheckpoint(paddle.callbacks.Callback):
                def __init__(self, save_freq=1, save_dir=None):
                    self.save_freq = save_freq
                    self.save_dir = save_dir

                def on_epoch_end(self, epoch, logs=None):
                    if self.model is not None and epoch % self.save_freq == 0:
                        path = '{}/{}'.format(self.save_dir, epoch)
                        print('save checkpoint at {}'.format(path))
                        self.model.save(path)

    """

    def __init__(self):
        self.model = None
        self.params = {}

    def set_params(self, params):
        """
        Set parameters, which is dict. The keys contain:

          - 'batch_size': an integer. Number of samples per batch.
          - 'epochs': an integer. Number of epochs.
          - 'steps': an integer. Number of steps of one epoch.
          - 'verbose': an integer. Verbose mode is 0, 1 or 2.
             0 = silent, 1 = progress bar, 2 = one line per epoch.
          - 'metrics': a list of str. Names of metrics, including 'loss'
              and the names of paddle.metric.Metric.
        """
        self.params = params

    def set_model(self, model):
        """model is instance of paddle.Model.
        """
        self.model = model

    def on_train_begin(self, logs=None):
        """Called at the start of training.

        Args:
            logs (dict): The logs is a dict or None.
        """

    def on_train_end(self, logs=None):
        """Called at the end of training.

        Args:
            logs (dict): The logs is a dict or None. The keys of logs
                passed by paddle.Model contains 'loss', metric names and
                `batch_size`.
        """

    def on_eval_begin(self, logs=None):
        """Called at the start of evaluation.

        Args:
            logs (dict): The logs is a dict or None. The keys of logs
                passed by paddle.Model contains 'steps' and 'metrics',
                The `steps` is number of total steps of validation dataset.
                The `metrics` is a list of str including 'loss' and the names
                of paddle.metric.Metric.
        """

    def on_eval_end(self, logs=None):
        """Called at the end of evaluation.

        Args:
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is a dict contains 'loss', metrics and 'batch_size'
                of last batch of validation dataset.
        """

    def on_test_begin(self, logs=None):
        """Called at the beginning of predict.

        Args:
            logs (dict): The logs is a dict or None.
        """

    def on_test_end(self, logs=None):
        """Called at the end of predict.

        Args:
            logs (dict): The logs is a dict or None.
        """

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch.

        Args:
            epoch (int): The index of epoch.
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is None.
        """

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch.

        Args:
            epoch (int): The index of epoch.
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is a dict, contains 'loss', metrics and 'batch_size'
                of last batch.
        """

    def on_train_batch_begin(self, step, logs=None):
        """Called at the beginning of each batch in training.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is empty.
        """

    def on_train_batch_end(self, step, logs=None):
        """Called at the end of each batch in training.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is a dict, contains 'loss', metrics and 'batch_size'
                of current batch.
        """

    def on_eval_batch_begin(self, step, logs=None):
        """Called at the beginning of each batch in evaluation.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is empty.
        """

    def on_eval_batch_end(self, step, logs=None):
        """Called at the end of each batch in evaluation.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is a dict, contains 'loss', metrics and 'batch_size'
                of current batch.
        """

    def on_test_batch_begin(self, step, logs=None):
        """Called at the beginning of each batch in predict.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None.
        """

    def on_test_batch_end(self, step, logs=None):
        """Called at the end of each batch in predict.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None.
        """


class ProgBarLogger(Callback):
    """Logger callback function
    Args:
        log_freq (int): The frequency, in number of steps, the logs such as `loss`, 
                `metrics` are printed. Default: 1.
        verbose (int): The verbosity mode, should be 0, 1, or 2.
                0 = silent, 1 = progress bar, 2 = one line per epoch. Default: 2.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.static import InputSpec

            inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
            labels = [InputSpec([None, 1], 'int64', 'label')]

            train_dataset = paddle.vision.datasets.MNIST(mode='train')

            lenet = paddle.vision.LeNet()
            model = paddle.Model(lenet,
                inputs, labels)

            optim = paddle.optimizer.Adam(0.001, parameters=lenet.parameters())
            model.prepare(optimizer=optim,
                        loss=paddle.nn.CrossEntropyLoss(),
                        metrics=paddle.metric.Accuracy())

            callback = paddle.callbacks.ProgBarLogger(log_freq=10)
            model.fit(train_dataset, batch_size=64, callbacks=callback)
    """

    def __init__(self, log_freq=1, verbose=2):
        self.epochs = None
        self.steps = None
        self.progbar = None
        self.verbose = verbose
        self.log_freq = log_freq

    def _is_print(self):
        return self.verbose and ParallelEnv().local_rank == 0

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        assert self.epochs
        self.train_metrics = self.params['metrics']
        assert self.train_metrics

    def on_epoch_begin(self, epoch=None, logs=None):
        self.steps = self.params['steps']
        self.epoch = epoch
        self.train_step = 0
        if self.epochs and self._is_print():
            print('Epoch %d/%d' % (epoch + 1, self.epochs))
        self.train_progbar = ProgressBar(num=self.steps, verbose=self.verbose)

    def _updates(self, logs, mode):
        values = []
        metrics = getattr(self, '%s_metrics' % (mode))
        progbar = getattr(self, '%s_progbar' % (mode))
        steps = getattr(self, '%s_step' % (mode))

        for k in metrics:
            if k in logs:
                values.append((k, logs[k]))

        progbar.update(steps, values)

    def on_train_batch_end(self, step, logs=None):
        logs = logs or {}
        self.train_step += 1

        if self._is_print() and self.train_step % self.log_freq == 0:
            if self.steps is None or self.train_step < self.steps:
                self._updates(logs, 'train')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self._is_print() and (self.steps is not None):
            self._updates(logs, 'train')

    def on_eval_begin(self, logs=None):
        self.eval_steps = logs.get('steps', None)
        self.eval_metrics = logs.get('metrics', [])
        self.eval_step = 0
        self.evaled_samples = 0

        self.eval_progbar = ProgressBar(
            num=self.eval_steps, verbose=self.verbose)
        if self._is_print():
            print('Eval begin...')

    def on_eval_batch_end(self, step, logs=None):
        logs = logs or {}
        self.eval_step += 1
        samples = logs.get('batch_size', 1)
        self.evaled_samples += samples

        if self._is_print() and self.eval_step % self.log_freq == 0:
            if self.eval_steps is None or self.eval_step < self.eval_steps:
                self._updates(logs, 'eval')

    def on_test_begin(self, logs=None):
        self.test_steps = logs.get('steps', None)
        self.test_metrics = logs.get('metrics', [])
        self.test_step = 0
        self.tested_samples = 0
        self.test_progbar = ProgressBar(
            num=self.test_steps, verbose=self.verbose)
        if self._is_print():
            print('Predict begin...')

    def on_test_batch_end(self, step, logs=None):
        logs = logs or {}
        self.test_step += 1
        samples = logs.get('batch_size', 1)
        self.tested_samples += samples

        if self.test_step % self.log_freq == 0 and self._is_print():
            if self.test_steps is None or self.test_step < self.test_steps:
                self._updates(logs, 'test')

    def on_eval_end(self, logs=None):
        logs = logs or {}
        if self._is_print() and (self.eval_steps is not None):
            self._updates(logs, 'eval')
            print('Eval samples: %d' % (self.evaled_samples))

    def on_test_end(self, logs=None):
        logs = logs or {}
        if self._is_print():
            if self.test_step % self.log_freq != 0 or self.verbose == 1:
                self._updates(logs, 'test')
            print('Predict samples: %d' % (self.tested_samples))


class ModelCheckpoint(Callback):
    """Model checkpoint callback function
    Args:
        save_freq(int): The frequency, in number of epochs, the model checkpoint 
                        are saved. Default: 1.
        save_dir(str|None): The directory to save checkpoint during training.
                If None, will not save checkpoint. Default: None.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.static import InputSpec

            inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
            labels = [InputSpec([None, 1], 'int64', 'label')]

            train_dataset = paddle.vision.datasets.MNIST(mode='train')

            lenet = paddle.vision.LeNet()
            model = paddle.Model(lenet,
                inputs, labels)

            optim = paddle.optimizer.Adam(0.001, parameters=lenet.parameters())
            model.prepare(optimizer=optim,
                        loss=paddle.nn.CrossEntropyLoss(),
                        metrics=paddle.metric.Accuracy())

            callback = paddle.callbacks.ModelCheckpoint(save_dir='./temp')
            model.fit(train_dataset, batch_size=64, callbacks=callback)
    """

    def __init__(self, save_freq=1, save_dir=None):
        self.save_freq = save_freq
        self.save_dir = save_dir

    def on_epoch_begin(self, epoch=None, logs=None):
        self.epoch = epoch

    def _is_save(self):
        return self.model and self.save_dir and ParallelEnv().local_rank == 0

    def on_epoch_end(self, epoch, logs=None):
        if self._is_save() and self.epoch % self.save_freq == 0:
            path = '{}/{}'.format(self.save_dir, epoch)
            print('save checkpoint at {}'.format(os.path.abspath(path)))
            self.model.save(path)

    def on_train_end(self, logs=None):
        if self._is_save():
            path = '{}/final'.format(self.save_dir)
            print('save checkpoint at {}'.format(os.path.abspath(path)))
            self.model.save(path)


class VisualDL(Callback):
    """VisualDL callback function
    Args:
        log_dir (str): The directory to save visualdl log file.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.static import InputSpec

            inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
            labels = [InputSpec([None, 1], 'int64', 'label')]

            train_dataset = paddle.vision.datasets.MNIST(mode='train')
            eval_dataset = paddle.vision.datasets.MNIST(mode='test')

            net = paddle.vision.LeNet()
            model = paddle.Model(net, inputs, labels)

            optim = paddle.optimizer.Adam(0.001, parameters=net.parameters())
            model.prepare(optimizer=optim,
                        loss=paddle.nn.CrossEntropyLoss(),
                        metrics=paddle.metric.Accuracy())
            
            ## uncomment following lines to fit model with visualdl callback function
            # callback = paddle.callbacks.VisualDL(log_dir='visualdl_log_dir')
            # model.fit(train_dataset, eval_dataset, batch_size=64, callbacks=callback)

    """

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.epochs = None
        self.steps = None
        self.epoch = 0

    def _is_write(self):
        return ParallelEnv().local_rank == 0

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        assert self.epochs
        self.train_metrics = self.params['metrics']
        assert self.train_metrics
        self._is_fit = True
        self.train_step = 0

    def on_epoch_begin(self, epoch=None, logs=None):
        self.steps = self.params['steps']
        self.epoch = epoch

    def _updates(self, logs, mode):
        if not self._is_write():
            return
        if not hasattr(self, 'writer'):
            visualdl = try_import('visualdl')
            self.writer = visualdl.LogWriter(self.log_dir)

        metrics = getattr(self, '%s_metrics' % (mode))
        current_step = getattr(self, '%s_step' % (mode))

        if mode == 'train':
            total_step = current_step
        else:
            total_step = self.epoch

        for k in metrics:
            if k in logs:
                temp_tag = mode + '/' + k

                if isinstance(logs[k], (list, tuple)):
                    temp_value = logs[k][0]
                elif isinstance(logs[k], numbers.Number):
                    temp_value = logs[k]
                else:
                    continue

                self.writer.add_scalar(
                    tag=temp_tag, step=total_step, value=temp_value)

    def on_train_batch_end(self, step, logs=None):
        logs = logs or {}
        self.train_step += 1

        if self._is_write():
            self._updates(logs, 'train')

    def on_eval_begin(self, logs=None):
        self.eval_steps = logs.get('steps', None)
        self.eval_metrics = logs.get('metrics', [])
        self.eval_step = 0
        self.evaled_samples = 0

    def on_train_end(self, logs=None):
        if hasattr(self, 'writer'):
            self.writer.close()
            delattr(self, 'writer')

    def on_eval_end(self, logs=None):
        if self._is_write():
            self._updates(logs, 'eval')

            if (not hasattr(self, '_is_fit')) and hasattr(self, 'writer'):
                self.writer.close()
                delattr(self, 'writer')
