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

import copy
import functools
import inspect
import io
import json
import os
import six
import logging

import collections
import math
import numpy as np
from paddle.fluid.dygraph.parallel import ParallelEnv

import paddle
from paddle.fluid.dygraph import Layer
from paddle.dataset.common import DATA_HOME
from paddle.incubate.hapi.download import get_path_from_url

from .vocab import Vocab

__all__ = [
    'PreTrainedTokenizer',
    'PreTrainedModel',
]

logger = logging.getLogger(__name__)


def _get_init_dict(init_func, *args, **kwargs):
    """
    Inspect function `init_func` and its arguments for running, and extract a
    dict mapping between argument names and keys. 
    """
    if hasattr(inspect, 'getfullargspec'):
        (spec_args, spec_varargs, spec_varkw, spec_defaults, _, _,
         _) = inspect.getfullargspec(init_func)
    else:
        (spec_args, spec_varargs, spec_varkw,
         spec_defaults) = inspect.getargspec(init_func)
    # add positional argument values
    init_dict = dict(zip(spec_args, args))
    # add default argument values
    kwargs_dict = dict(zip(spec_args[-len(spec_defaults):],
                           spec_defaults)) if spec_defaults else {}
    kwargs_dict.update(kwargs)
    init_dict.update(kwargs_dict)
    return init_dict


def register_base_model(cls):
    """
    Add a `base_model_class` attribute for the base class of decorated class,
    representing the base model class in derived classes of the same architecture.

    Args:
        cls (class): the name of the model
    """
    base_cls = cls.__bases__[0]
    assert issubclass(
        base_cls, PreTrainedModel
    ), "`register_base_model` should be used on subclasses of PreTrainedModel."
    base_cls.base_model_class = cls
    return cls


class InitTrackerMeta(type(Layer)):
    """
    This metaclass wraps the `__init__` method of a class to add `init_config`
    attribute for instances of that class, and `init_config` use a dict to track
    the initial configuration. If the class has `_wrap_init` method, it would be
    hooked after `__init__` and called as `_wrap_init(self, init_fn, init_args)`.

    Since InitTrackerMeta would be used as metaclass for pretrained model classes,
    which always are Layer and `type(Layer)` is not `type`, thus use `type(Layer)`
    rather than `type` as base class for it to avoid inheritance metaclass
    conflicts.
    """

    def __init__(cls, name, bases, attrs):
        init_func = cls.__init__
        # If attrs has `__init__`, wrap it using accessable `_wrap_init`.
        # Otherwise, no need to wrap again since the super cls has been wraped.
        # TODO: remove reduplicated tracker if using super cls `__init__`
        help_func = getattr(cls, '_wrap_init',
                            None) if '__init__' in attrs else None
        cls.__init__ = InitTrackerMeta.init_and_track_conf(init_func, help_func)
        super(InitTrackerMeta, cls).__init__(name, bases, attrs)

    @staticmethod
    def init_and_track_conf(init_func, help_func=None):
        """
        wraps `init_func` which is `__init__` method of a class to add `init_config`
        attribute for instances of that class.

        Args:
            init_func (callable): It should be the `__init__` method of a class.
            help_func (callable, optional): If provided, it would be hooked after
                `init_func` and called as `_wrap_init(self, init_func, *init_args, **init_args)`.
                Default None.
        
        Returns:
            function: the wrapped function
        """

        @functools.wraps(init_func)
        def __impl__(self, *args, **kwargs):
            # keep full configuration
            init_func(self, *args, **kwargs)
            # registed helper by `_wrap_init`
            if help_func:
                help_func(self, init_func, *args, **kwargs)
            self.init_config = kwargs
            if args:
                kwargs['init_args'] = args
            kwargs['init_class'] = self.__class__.__name__

        return __impl__


@six.add_metaclass(InitTrackerMeta)
class PreTrainedTokenizer(object):
    """
    The base class for all pretrained tokenizers. It provides some attributes
    and common methods for all pretrained tokenizers, including attributes for
    and special tokens (arguments of `__init__` whose name ends with `_token`)
    and methods for saving and loading.

    It also includes some class attributes (should be set by derived classes):

    - `tokenizer_config_file` (str): represents the file name for saving and loading
      tokenizer configuration, it's value is `tokenizer_config.json`.

    - `resource_files_names` (dict): use this to map resource related arguments
      of `__init__` to specific file names for saving and loading.

    - `pretrained_resource_files_map` (dict): The dict has the same keys as
      `resource_files_names`, the values are also dict mapping specific pretrained
      model name to URL linking to vocabulary or other resources.

    - `pretrained_init_configuration` (dict): The dict has pretrained model names
      as keys, and the values are also dict preserving corresponding configuration
      for tokenizer initialization.
    """
    tokenizer_config_file = "tokenizer_config.json"
    pretrained_init_configuration = {}
    resource_files_names = {}  # keys are arguments of __init__
    pretrained_resource_files_map = {}

    def _wrap_init(self, original_init, *args, **kwargs):
        """
        It would be hooked after `__init__` to add specials tokens (arguments of
        `__init__` whose name ends with `_token`) as attributes of the tokenizer
        instance.
        """
        # expose tokens as attributes
        init_dict = _get_init_dict(original_init, *args, **kwargs)
        for identifier, token in init_dict.items():
            if identifier.endswith('_token'):
                setattr(self, identifier, token)

    def convert_tokens_to_ids(self, tokens):
        """
        Converts a sequence of tokens into ids using the vocab. The tokenizer
        should has the `vocab` attribute.

        Args：
            tokens (list(str)): List of tokens.

        Returns:
            list: Converted id list.

        """
        return [self.vocab[token] for token in tokens]

    def convert_tokens_to_string(self, tokens):
        """ 
        Converts a sequence of tokens (list of string) to a single string by
        using :code:`' '.join(tokens)` .

        Args:
            tokens (list(str)): List of tokens.

        Returns:
            str: Converted string.
        """
        return " ".join(tokens)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Instantiate an instance of `PreTrainedTokenizer` from a predefined
        tokenizer specified by name or path., and it always corresponds to a
        pretrained model.

        Args:
            pretrained_model_name_or_path (str): A name of or a file path to a
                pretrained model.
            *args (tuple): position arguments for `__init__`. If provide, use
                this as position argument values for tokenizer initialization.
            **kwargs (dict): keyword arguments for `__init__`. If provide, use
                this to update pre-defined keyword argument values for tokenizer
                initialization.

        Returns:
            PreTrainedTokenizer: An instance of PreTrainedTokenizer.
        """
        pretrained_models = list(cls.pretrained_init_configuration.keys())
        vocab_files = {}
        init_configuration = {}
        if pretrained_model_name_or_path in pretrained_models:
            for file_id, map_list in cls.pretrained_resource_files_map.items():
                vocab_files[file_id] = map_list[pretrained_model_name_or_path]
            init_configuration = copy.deepcopy(
                cls.pretrained_init_configuration[
                    pretrained_model_name_or_path])
        else:
            if os.path.isdir(pretrained_model_name_or_path):
                for file_id, file_name in cls.resource_files_names.items():
                    full_file_name = os.path.join(pretrained_model_name_or_path,
                                                  file_name)
                    vocab_files[file_id] = full_file_name
                vocab_files["tokenizer_config_file"] = os.path.join(
                    pretrained_model_name_or_path, cls.tokenizer_config_file)
            else:
                raise ValueError(
                    "Calling {}.from_pretrained() with a model identifier or the "
                    "path to a directory instead. The supported model "
                    "identifiers are as follows: {}".format(
                        cls.__name__, cls.pretrained_init_configuration.keys()))

        default_root = os.path.join(DATA_HOME, pretrained_model_name_or_path)
        resolved_vocab_files = {}
        for file_id, file_path in vocab_files.items():
            resolved_vocab_files[
                file_id] = file_path if file_path is None or os.path.isfile(
                    file_path) else get_path_from_url(file_path, default_root,
                                                      None)

        # Prepare tokenizer initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        tokenizer_config_file = resolved_vocab_files.pop(
            "tokenizer_config_file", None)
        if tokenizer_config_file is not None:
            with io.open(tokenizer_config_file, encoding="utf-8") as f:
                init_kwargs = json.load(f)
        else:
            init_kwargs = init_configuration
        # position args are stored in kwargs, maybe better not include
        init_args = init_kwargs.pop("init_args", ())

        # Update with newly provided args and kwargs
        init_args = init_args if not args else args
        init_kwargs.update(kwargs)

        # Merge resolved_vocab_files arguments in init_kwargs if not including.
        # Maybe need more ways to load resources.
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path
        # TODO(guosheng): avoid reduplication of position args and key word args
        tokenizer = cls(*init_args, **init_kwargs)
        return tokenizer

    def save_pretrained(self, save_directory):
        """
        Save tokenizer configuration and related resources to files under
        `save_directory`.

        Args:
            save_directory (str): Directory to save files into.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving directory ({}) should be a directory".format(save_directory)
        tokenizer_config_file = os.path.join(save_directory,
                                             self.tokenizer_config_file)
        # init_config is set in metaclass created `__init__`,
        tokenizer_config = self.init_config
        with io.open(tokenizer_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(tokenizer_config, ensure_ascii=False))

        self.save_resources(save_directory)

    def save_resources(self, save_directory):
        """
        Save tokenizer related resources to files under `save_directory`.

        Args:
            save_directory (str): Directory to save files into.
        """
        assert hasattr(self, 'vocab') and len(
            self.resource_files_names) == 1, "Must overwrite `save_resources`"
        file_name = os.path.join(save_directory,
                                 list(self.resource_files_names.values())[0])
        self.save_vocabulary(file_name, self.vocab)

    @staticmethod
    def load_vocabulary(filepath,
                        unk_token=None,
                        pad_token=None,
                        bos_token=None,
                        eos_token=None,
                        **kwargs):
        """
        Instantiate an instance of `Vocab` from a file reserving all tokens
        by using `Vocab.from_dict`. The file contains a token per line, and the
        line number would be the index of corresponding token.

        Args:
            filepath (str): path of file to construct vocabulary.
            unk_token (str): special token for unknow token. If no need, it also
                could be None. Default: None.
            pad_token (str): special token for padding token. If no need, it also
                could be None. Default: None.
            bos_token (str): special token for bos token. If no need, it also
                could be None. Default: None.
            eos_token (str): special token for eos token. If no need, it also
                could be None. Default: None.
            **kwargs (dict): keyword arguments for `Vocab.from_dict`.

        Returns:
            Vocab: An instance of `Vocab`.
        """
        token_to_idx = {}
        with io.open(filepath, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                token = line.rstrip('\n')
                token_to_idx[token] = int(index)
        vocab = Vocab.from_dict(
            token_to_idx,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs)
        return vocab

    @staticmethod
    def save_vocabulary(filepath, vocab):
        """
        Save all tokens to a vocabulary file. The file contains a token per line,
        and the line number would be the index of corresponding token.

        Agrs:
            filepath (str): File path to be saved to.
            vocab (Vocab|dict): the Vocab or dict instance to be saved.
        """
        if isinstance(vocab, Vocab):
            tokens = vocab.idx_to_token
        else:
            tokens = sorted(vocab.keys(), key=lambda token: vocab[token])
        with io.open(filepath, 'w', encoding='utf-8') as f:
            for token in tokens:
                f.write(token + '\n')


@six.add_metaclass(InitTrackerMeta)
class PreTrainedModel(Layer):
    """
    The base class for all pretrained models. It provides some attributes and
    common methods for all pretrained models, including attributes `init_config`,
    `config` for initialized arguments and methods for saving, loading.

    It also includes some class attributes (should be set by derived classes):

    - `model_config_file` (str): represents the file name for saving and loading
      model configuration, it's value is `model_config.json`.

    - `resource_files_names` (dict): use this to map resources to specific file
      names for saving and loading.

    - `pretrained_resource_files_map` (dict): The dict has the same keys as
      `resource_files_names`, the values are also dict mapping specific pretrained
      model name to URL linking to pretrained model.

    - `pretrained_init_configuration` (dict): The dict has pretrained model names
      as keys, and the values are also dict preserving corresponding configuration
      for model initialization.
    
    - `base_model_prefix` (str): represents the the attribute associated to the
      base model in derived classes of the same architecture adding layers on
      top of the base model.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {}
    # TODO: more flexible resource handle, namedtuple with fileds as:
    # resource_name, saved_file, handle_name_for_load(None for used as __init__
    # arguments), handle_name_for_save
    resource_files_names = {"model_state": "model_state"}
    pretrained_resource_files_map = {}
    base_model_prefix = ""

    def _wrap_init(self, original_init, *args, **kwargs):
        """
        It would be hooked after `__init__` to add a dict including arguments of
        `__init__` as a attribute named `config` of the prtrained model instance.
        """
        # expose tokens as attributes
        init_dict = _get_init_dict(original_init, *args, **kwargs)
        self.config = init_dict

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix, self)

    @property
    def model_name_list(self):
        return list(self.pretrained_init_configuration.keys())

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Instantiate an instance of `PreTrainedModel` from a predefined
        model specified by name or path.

        Args:
            pretrained_model_name_or_path (str): A name of or a file path to a
                pretrained model.
            *args (tuple): position arguments for `__init__`. If provide, use
                this as position argument values for model initialization.
            **kwargs (dict): keyword arguments for `__init__`. If provide, use
                this to update pre-defined keyword argument values for model
                initialization.

        Returns:
            PreTrainedModel: An instance of PreTrainedModel.
        """
        pretrained_models = list(cls.pretrained_init_configuration.keys())
        resource_files = {}
        init_configuration = {}
        if pretrained_model_name_or_path in pretrained_models:
            for file_id, map_list in cls.pretrained_resource_files_map.items():
                resource_files[file_id] = map_list[
                    pretrained_model_name_or_path]
            init_configuration = copy.deepcopy(
                cls.pretrained_init_configuration[
                    pretrained_model_name_or_path])
        else:
            if os.path.isdir(pretrained_model_name_or_path):
                for file_id, file_name in cls.resource_files_names.items():
                    full_file_name = os.path.join(pretrained_model_name_or_path,
                                                  file_name)
                    resource_files[file_id] = full_file_name
            else:
                raise ValueError(
                    "Calling {}.from_pretrained() with a model identifier or the "
                    "path to a directory instead. The supported model "
                    "identifiers are as follows: {}".format(
                        cls.__name__, cls.pretrained_init_configuration.keys()))

        default_root = os.path.join(DATA_HOME, pretrained_model_name_or_path)
        resolved_resource_files = {}
        for file_id, file_path in resource_files.items():
            resolved_resource_files[
                file_id] = file_path if file_path is None or os.path.isfile(
                    file_path) else get_path_from_url(file_path, default_root,
                                                      None)

        # Prepare model initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        model_config_file = resolved_resource_files.pop("model_config_file",
                                                        None)
        if model_config_file is not None:
            with io.open(model_config_file, encoding="utf-8") as f:
                init_kwargs = json.load(f)
        else:
            init_kwargs = init_configuration
        # position args are stored in kwargs, maybe better not include
        init_args = init_kwargs.pop("init_args", ())
        # class name corresponds to this configuration
        init_class = init_kwargs.pop("init_class",
                                     cls.base_model_class.__name__)

        # Check if the loaded config matches the current model class's __init__
        # arguments. If not match, the loaded config is for the base model class.
        if init_class == cls.base_model_class.__name__:
            base_args = init_args
            base_kwargs = init_kwargs
            derived_args = ()
            derived_kwargs = {}
            base_arg_index = None
        else:  # extract config for base model
            derived_args = list(init_args)
            derived_kwargs = init_kwargs
            for i, arg in enumerate(init_args):
                if isinstance(arg, dict) and "init_class" in arg:
                    assert arg.pop(
                        "init_class") == cls.base_model_class.__name__, (
                            "pretrained base model should be {}"
                        ).format(cls.base_model_class.__name__)
                    base_arg_index = i
            for arg_name, arg in init_kwargs.items():
                if isinstance(arg, dict) and "init_class" in arg:
                    assert arg.pop(
                        "init_class") == cls.base_model_class.__name__, (
                            "pretrained base model should be {}"
                        ).format(cls.base_model_class.__name__)
                    base_arg_index = arg_name
            base_args = arg.pop("init_args", ())
            base_kwargs = arg
        if cls == cls.base_model_class:
            # Update with newly provided args and kwargs for base model
            base_args = base_args if not args else args
            base_kwargs.update(kwargs)
            model = cls(*base_args, **base_kwargs)
        else:
            # Update with newly provided args and kwargs for derived model
            base_model = cls.base_model_class(*base_args, **base_kwargs)
            if base_arg_index is not None:
                derived_args[base_arg_index] = base_model
            else:
                derived_args = (base_model, )  # assume at the first position
            derived_args = derived_args if not args else args
            derived_kwargs.update(kwargs)
            model = cls(*derived_args, **derived_kwargs)

        # Maybe need more ways to load resources.
        weight_path = list(resolved_resource_files.values())[0]
        assert weight_path.endswith(
            ".pdparams"), "suffix of weight must be .pdparams"
        state_dict, _ = paddle.fluid.load_dygraph(weight_path)

        # Make sure we are able to load base models as well as derived models
        # (with heads)
        start_prefix = ""
        model_to_load = model
        state_to_load = state_dict
        unexpected_keys = []
        missing_keys = []
        if not hasattr(model, cls.base_model_prefix) and any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            # base model
            state_to_load = {}
            start_prefix = cls.base_model_prefix + "."
            for k, v in state_dict.items():
                if k.startswith(cls.base_model_prefix):
                    state_to_load[k[len(start_prefix):]] = v
                else:
                    unexpected_keys.append(k)
        if hasattr(model, cls.base_model_prefix) and not any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            # derived model (base model with heads)
            model_to_load = getattr(model, cls.base_model_prefix)
            for k in model.state_dict().keys():
                if not k.startswith(cls.base_model_prefix):
                    missing_keys.append(k)
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".
                format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".
                        format(model.__class__.__name__, unexpected_keys))
        model_to_load.load_dict(state_to_load)
        return model

    def save_pretrained(self, save_directory):
        """
        Save model configuration and related resources (model state) to files
        under `save_directory`.

        Args:
            save_directory (str): Directory to save files into.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving directory ({}) should be a directory".format(save_directory)
        # save model config
        model_config_file = os.path.join(save_directory, self.model_config_file)
        model_config = self.init_config
        # If init_config contains a Layer, use the layer's init_config to save
        for key, value in model_config.items():
            if key == "input_args":
                args = []
                for arg in value:
                    args.append(
                        arg.init_config
                        if isinstance(arg, PreTrainedModel) else arg)
                model_config[key] = tuple(args)
            elif isinstance(value, PreTrainedModel):
                model_config[key] = value.init_config
        with io.open(model_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(model_config, ensure_ascii=False))
        # save model
        file_name = os.path.join(save_directory,
                                 list(self.resource_files_names.values())[0])
        paddle.fluid.save_dygraph(self.state_dict(),
                                  os.path.join(save_directory, file_name))


class SamplerHelper(object):
    """
    SamplerHelper is to help construct iterable sampler used for `DataLoader`. It wraps
    a dataset and uses its :code:`__getitem__`


    Every SamplerHelper subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    Also can be used as batch iterator instead of indices iterator when `iterator`
    yield samples rather than indices by initializing `iterator` with a iterable
    dataset.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`DataLoader`, but is expected in any
              calculation involving the length of a :class:`DataLoader`.

    Args:
        dataset (Dataset): Input dataset for SamplerHelper.
        iterable (collections.Iterable|callable, optional): Iterator of dataset. Default: None.
    """

    # chain sampler
    def __init__(self, dataset, iterable=None):
        self.data_source = dataset
        self.iterable = iterable
        if isinstance(dataset, collections.Iterable) and iterable is None:
            # iterable-style datasets
            self.iterable = dataset

    def __iter__(self):
        if self.iterable is None:
            return iter(range(len(self.data_source)))
        elif isinstance(self.iterable, collections.Iterable):
            return iter(self.iterable)
        elif callable(self.iterable):
            return self.iterable()
        else:
            raise ValueError(
                "`iterable` should be None, instance of Iterable or callable "
                "producing generator.")

    def __len__(self):
        # Allow some samplers have different length with `len(data_source)`,
        # such as batch sampler.
        if hasattr(self, "_length"):
            return self._length
        else:
            return len(self.data_source)

    @property
    def length(self):
        """
        Returns:
            the length of the SamplerHelper.
        """

        # since `len()` only produce integer, use length property to get None
        # for uncertain length. samplers can set length if necessary.
        try:
            length = len(self)
        except Exception:
            length = None
        return length

    @length.setter
    def length(self, length):
        self._length = length

    def apply(self, fn):
        """
        Transformations would be performed. It includes `Shuffle`, `sort`, `fit` and `shard`.

        Args:
            fn (callable): Transformations to be performed. It returns transformed iterable (and data_source).

        Returns:
            SamplerHelper: A new transformed object.
        """
        rs = fn(self)
        if isinstance(rs, (list, tuple)):
            iterable, data_source = rs
        else:
            iterable, data_source = rs, self.data_source
        sampler = type(self)(data_source, iterable)
        return sampler

    def shuffle(self, buffer_size=-1, seed=None):
        """
        Shuffle the dataset according to the given buffer size and random seed.
        Args:
            buffer_size (int): Buffer size for shuffle. if buffer_size < 0 or more than the length of the dataset, 
                buffer_size is the length of the dataset. Default: -1. 
            seed (int, optional): Seed for the random. Default: None.
        Returns:
            SamplerHelper
         """
        if seed is not None:
            random_generator = np.random.RandomState(seed)
        else:  # use the global random generator
            random_generator = np.random

        def _impl():
            buf = []
            for idx in iter(self):
                buf.append(idx)
                if buffer_size > 0 and len(buf) >= buffer_size:
                    random_generator.shuffle(buf)
                    for b in buf:
                        yield b
                    buf = []
            if len(buf) > 0:
                random_generator.shuffle(buf)
                for b in buf:
                    yield b

        return type(self)(self.data_source, _impl)

    def sort(self, cmp=None, key=None, reverse=False, buffer_size=-1):
        """
        Sort samples according to given callable cmp or key.

        Args:
            cmp (callable): The funcation of comparison. Default: None. 
            key (callable): Return element to be compared. Default: None.
            reverse (bool): If True, it means in descending order, and False means in ascending order. Default: False.
            buffer_size (int): Buffer size for sort. If buffer_size < 0 or buffer_size is more than the length of the data, 
                buffer_size will be set to the length of the data. Default: -1.
        Returns:
            SamplerHelper
        """
        if key:
            key = (lambda x: key(x, data_source))
        elif cmp:
            key = functools.cmp_to_key(lambda x, y: cmp(x, y, data_source))

        def _impl():
            data_source = self.data_source
            buf = []
            for idx in iter(self):
                buf.append(idx)
                if buffer_size > 0 and len(buf) >= buffer_size:
                    buf = sorted(buf, key=key, reverse=reverse)
                    for b in buf:
                        yield b
                    buf = []
            if len(buf) > 0:
                buf = sorted(buf, key=key, reverse=reverse)
                for b in buf:
                    yield b

        return type(self)(self.data_source, _impl)

    def batch(self,
              batch_size,
              drop_last=False,
              batch_size_fn=None,
              batch_fn=None):
        """
        To produce a BatchSampler.

        Agrs:
            batch_size (int): Batch size.
            drop_last (bool): Whether to drop the last mini batch. Default: False.
            batch_size_fn (callable, optional): Return the size of mini batch so far. Default: None.
            batch_fn (callable, optional): Transformations to be performed. Default: None.
        Returns:
            SamplerHelper
        """
        if batch_size_fn is None:
            batch_size_fn = lambda new, count, sofar, data_source: count

        def _impl():
            data_source = self.data_source
            minibatch, size_so_far = [], 0
            for idx in iter(self):
                minibatch.append(idx)
                size_so_far = batch_size_fn(idx,
                                            len(minibatch), size_so_far,
                                            data_source)
                if size_so_far == batch_size:
                    yield minibatch
                    minibatch, size_so_far = [], 0
                elif size_so_far > batch_size:
                    yield minibatch[:-1]
                    minibatch, size_so_far = minibatch[-1:], batch_size_fn(
                        idx, 1, 0, data_source)
            if minibatch and not drop_last:
                yield minibatch

        sampler = type(self)(
            self.data_source,
            _impl) if batch_fn is None else self.apply(batch_fn)
        if batch_size_fn is None and batch_fn is None and self.length is not None:
            sampler.length = (self.length + int(not drop_last) *
                              (batch_size - 1)) // batch_size
        else:
            sampler.length = None

        return sampler

    def shard(self, num_replicas=None, rank=None):
        """
        Operates slice using multi GPU.

        Args:
            num_replicas (int, optional): The number of training process, and is also the number of GPU cards used in training. 
                Default: None.
            rank (int, optional): Number of training process. Equal to the value of the environment variable PADDLE_TRAINER_ID.
                Default: None.
        Returns:
            SamplerHelper
        """
        if num_replicas is None:
            num_replicas = ParallelEnv().nranks
        if rank is None:
            rank = ParallelEnv().local_rank

        def _impl():
            for i, idx in enumerate(self):
                if i % num_replicas == rank:
                    yield idx
            if i % num_replicas != num_replicas - 1 and rank > i % num_replicas:
                # use last samples to make it evenly divisible
                yield idx

        sampler = type(self)(self.data_source, _impl)
        if self.length is not None:
            sampler.length = int(math.ceil(self.length * 1.0 / num_replicas))
        else:
            sampler.length = None
        return sampler

    def list(self):
        """
        Produce a sampler with a `listiterator` when calling `iter`. Since `list`
        would fetch all contents at time, thus it can get accurate length.

        Returns:
            SamplerHelper
        """

        def _impl():
            indices = list(iter(self))
            self.length = len(indices)
            return iter(indices)

        return type(self)(self.data_source, _impl)
