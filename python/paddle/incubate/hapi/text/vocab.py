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

import collections
import io
import json
import os


class Vocab(object):
    """
    Vocab is for mapping between text tokens and ids.

    Args:
        counter (collections.Counter, optional): A Counter intance describes
            the tokens and their frequencies. Its keys will be indexed accroding
            to the order of frequency sorting to construct mapping relationship. 
            If None, `token_to_idx` must be provided as the mapping relationship.
            Default: None.
        max_size (int, optional): Max size of vocab, not including special tokens.
            Default: None.
        min_freq (int): Ignore tokens whose frequencies are less than `min_freq`.
            Default: 1.
        token_to_idx (dict, optional): A dict specifies the mapping relationship
            between tokens and indices to be used. If provided, adjust the tokens
            and indices mapping according to it. If None, counter must be provided.
            Default: None.
        unk_token (str): special token for unknow token. If no need, it also
            could be None. Default: '<unk>'.
        pad_token (str): special token for padding token. If no need, it also
            could be None. Default: '<pad>'.
        bos_token (str): special token for bos token. If no need, it also
            could be None. Default: <bos>'.
        eos_token (str): special token for eos token. If no need, it also
            could be None. Default: '<eos>'.
        **kwargs (dict): Keyword arguments ending with `_token`. It can be used
            to specify further special tokens that will be exposed as attribute
            of the vocabulary and associated with an index.
    """

    def __init__(self,
                 counter=None,
                 max_size=None,
                 min_freq=1,
                 token_to_idx=None,
                 unk_token='<unk>',
                 pad_token='<pad>',
                 bos_token='<bos>',
                 eos_token='<eos>',
                 **kwargs):
        # Handle special tokens
        combs = (('unk_token', unk_token), ('pad_token', pad_token),
                 ('bos_token', bos_token), ('eos_token', eos_token))
        for name, value in combs:
            kwargs[name] = value
        special_tokens = []
        special_iter = kwargs.keys()
        # sort alphabetically
        special_iter = sorted(special_iter)
        for special_token_name in special_iter:
            # Test if kwarg specifies a special token
            if not special_token_name.endswith('_token'):
                raise ValueError('{} is invalid. Only keyword arguments '
                                 'that end in \'_token\' are supported '
                                 'to declare special tokens.'.format(
                                     special_token_name))

            special_token = kwargs[special_token_name]
            if special_token is not None and special_token not in special_tokens:
                special_tokens.append(special_token)

        if counter is None:
            # use token_to_idx as dict to import pretrained vocabulary
            assert token_to_idx, (
                'token_to_idx should not be None when counter is None')
            for special_token in special_tokens:
                assert special_token in token_to_idx, '{} is not in token_to_idx'.format(
                    special_token)
            self._token_to_idx = token_to_idx
            self._idx_to_token = sorted(
                self._token_to_idx.keys(),
                key=lambda token: self._token_to_idx[token])
            if unk_token:
                unk_index = self._token_to_idx[unk_token]
                self._token_to_idx = collections.defaultdict(lambda: unk_index)
                self._token_to_idx.update(token_to_idx)
        else:
            self._idx_to_token = list(special_tokens)
            self._token_to_idx = collections.defaultdict()
            self._token_to_idx.update(
                (token, idx) for idx, token in enumerate(self._idx_to_token))
            self._index_counter_keys(counter, special_tokens, max_size,
                                     min_freq)
            if token_to_idx:
                self._sort_index_according_to_user_specification(token_to_idx)
            if unk_token:
                self._token_to_idx.default_factory = lambda: self._token_to_idx[unk_token]

        # _expose_tokens_as_attributes
        self._identifiers_to_tokens = kwargs
        for identifier, token in kwargs.items():
            if identifier.startswith('_'):
                raise ValueError(
                    'It is not allowed to use identifiers starting with '
                    'underscore. In Python identifier names beginning with '
                    'underscore are internal.')
            if hasattr(self, identifier):
                raise ValueError(
                    'vocab.{} already exists. '
                    'Please choose a different identifier for token {}'.format(
                        identifier, token))
            setattr(self, identifier, token)

    def _index_counter_keys(self, counter, special_tokens, max_size, min_freq):
        # sort by frequency, then alphabetically
        token_freqs = sorted(counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        special_tokens = set(special_tokens)
        max_size = None if max_size is None else max_size + len(special_tokens)
        for token, freq in token_freqs:
            if freq < min_freq or len(self._idx_to_token) == max_size:
                break
            if token not in special_tokens:
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

    def _sort_index_according_to_user_specification(self, token_to_idx):
        # Sanity checks
        if not set(token_to_idx.keys()).issubset(self.token_to_idx.keys()):
            raise ValueError(
                'User-specified token_to_idx mapping can only contain '
                'tokens that will be part of the vocabulary.')
        if len(set(token_to_idx.values())) != len(token_to_idx):
            raise ValueError(
                'User-specified indices must not contain duplicates.')
        if min(token_to_idx.values()) < 0 or max(token_to_idx.values()) >= len(
                self.token_to_idx):
            raise ValueError(
                'User-specified indices must not be < 0 or >= the number of tokens '
                'that will be in the vocabulary. The current vocab contains {}'
                'tokens.'.format(len(self.token_to_idx)))

        # Update index ordering
        for token, new_idx in token_to_idx.items():
            old_idx = self.token_to_idx[token]
            ousted_token = self.idx_to_token[new_idx]

            self.token_to_idx[token] = new_idx
            self.token_to_idx[ousted_token] = old_idx
            self.idx_to_token[old_idx] = ousted_token
            self.idx_to_token[new_idx] = token

    def to_tokens(self, indices):
        """
        Map the input indices to token list.

        Args:
            indices (list|tuple|int): input indices for mapping.

        Returns:
            list|str: obtained token(s).
        """
        to_reduce = False
        if not isinstance(indices, (list, tuple)):
            indices = [indices]
            to_reduce = True

        max_idx = len(self._idx_to_token) - 1

        tokens = []
        for idx in indices:
            if not isinstance(idx, int) or idx > max_idx:
                raise ValueError(
                    'Token index {} in the provided `indices` is invalid.'.
                    format(idx))
            tokens.append(self._idx_to_token[idx])

        return tokens[0] if to_reduce else tokens

    def to_indices(self, tokens):
        """
        Map the input tokens into indices

        Args:
            tokens (list|tuple, optional): input tokens for mapping.

        Returns:
            list|int: obationed indice list.
        """
        return self[tokens]

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx[tokens]
        else:
            return [self._token_to_idx[token] for token in tokens]

    def __len__(self):
        return len(self._idx_to_token)

    def __contains__(self, token):
        return token in self._token_to_idx

    def __call__(self, tokens):
        return self[tokens]

    @property
    def idx_to_token(self):
        """
        Return index-token dict
        """
        return self._idx_to_token

    @property
    def token_to_idx(self):
        """
        Return token-index dict
        """
        return self._token_to_idx

    def to_json(self, path=None):
        """
        Summarize some information of vocab as JSON string. If path is gaven,
        the JSON string will be saved into files.

        Args:
            path (str, optional): the path to save JSON string. If None, the
                JSON will not be saved. Default: None.

        Returns:
                str: JSON string.
        """
        vocab_dict = {}
        vocab_dict['idx_to_token'] = self.idx_to_token
        vocab_dict['token_to_idx'] = dict(self.token_to_idx)
        vocab_dict['unk_token'] = self.unk_token
        vocab_dict['identifiers_to_tokens'] = self._identifiers_to_tokens
        json_str = json.dumps(vocab_dict)
        if path:
            with io.open(path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str):
        """
        Load vocab from JSON string or JSON file.

        Args:
            json_str (str): JSON string or file path of JSON string.

        Returns:
            Vocab: vocab generated from information contained in JSON string.
        """
        if os.path.isfile(json_str):
            with io.open(json_str, 'w', encoding='utf-8') as f:
                vocab_dict = json.load(f)
        else:
            vocab_dict = json.loads(json_str)
        token_to_idx = vocab_dict.get('token_to_idx')
        unk_token = vocab_dict.get('unk_token')
        identifiers_to_tokens = vocab_dict.get('identifiers_to_tokens', dict())
        vocab = cls(counter=None,
                    token_to_idx=token_to_idx,
                    unk_token=unk_token,
                    **identifiers_to_tokens)
        return vocab

    @classmethod
    def from_dict(cls,
                  token_to_idx,
                  unk_token=None,
                  pad_token=None,
                  bos_token=None,
                  eos_token=None,
                  **kwargs):
        """
        Generate vocab from a dict.

        Args:
            token_to_idx (dict): A dict describes the mapping relationship between
                tokens to indices.
            unk_token (str): special token for unknow token. If no need, it also
                could be None. Default: None.
            pad_token (str): special token for padding token. If no need, it also
                could be None. Default: None.
            bos_token (str): special token for bos token. If no need, it also
                could be None. Default: None.
            eos_token (str): special token for eos token. If no need, it also
                could be None. Default: None.
            **kwargs (dict): Keyword arguments ending with `_token`. It can be used
                to specify further special tokens that will be exposed as attribute
                of the vocabulary and associated with an index.

        Returns:
            Vocab: vocab generated from the given dict and special tokens.
        """
        vocab = cls(counter=None,
                    token_to_idx=token_to_idx,
                    unk_token=unk_token,
                    pad_token=pad_token,
                    bos_token=bos_token,
                    eos_token=eos_token,
                    **kwargs)
        return vocab

    @staticmethod
    def build_vocab(iterator,
                    max_size=None,
                    min_freq=1,
                    token_to_idx=None,
                    unk_token='<unk>',
                    pad_token='<pad>',
                    bos_token='<bos>',
                    eos_token='<eos>',
                    **kwargs):
        """
        Building vocab accoring to given iterator and other information. Iterate
        over the `iterator` to construct a `Counter` and as `__init__`

        Args:
            iterator (collections.Iterable): Iterator of tokens.
            max_size (int, optional): Max size of vocab, not including special tokens. Default: None.
            min_freq (int): Ignore tokens whose frequencies are less than `min_freq`. Default: 1.
            token_to_idx (dict, optional): A dict specifies the mapping relationship
                between tokens and indices to be used. If provided, adjust the tokens
                and indices mapping according to it. If None, counter must be provided.
                Default: None.
            unk_token (str): special token for unknow token. If no need, it also
                could be None. Default: '<unk>'.
            pad_token (str): special token for padding token. If no need, it also
                could be None. Default: '<pad>'.
            bos_token (str): special token for bos token. If no need, it also
                could be None. Default: <bos>'.
            eos_token (str): special token for eos token. If no need, it also
                could be None. Default: '<eos>'.
            **kwargs (dict): Keyword arguments ending with `_token`. It can be used
                to specify further special tokens that will be exposed as attribute
                of the vocabulary and associated with an index.

        Returns:
            Vocab: Generated vocab from given iterator and other informations.
        """
        counter = collections.Counter()
        for tokens in iterator:
            counter.update(tokens)
        vocab = Vocab(
            counter,
            max_size=max_size,
            min_freq=min_freq,
            token_to_idx=token_to_idx,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs)
        return vocab
