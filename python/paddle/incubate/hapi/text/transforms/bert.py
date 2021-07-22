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
import six
import unicodedata

from ..utils import PreTrainedTokenizer

__all__ = ['BertTokenizer', ]


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming utf-8 input.

    Args:
        text (str|bytes): Text to be converted to unicode.

    Returns: 
        str: converted text.
    """
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def whitespace_tokenize(text):
    """
    Runs basic whitespace cleaning and splitting on a peice of text.

    Args:
        text (str): Text to be tokened.

    Returns:
        list(str): Token list.
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_whitespace(char):
    """
    Checks whether `chars` is a whitespace character.
    """
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class BertBasicTokenizer(object):
    """
    Runs basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (bool): Whether the text strips accents and convert to
            lower case. If you use the BERT pretrained model, lower is set to
            Flase when using the cased model, otherwise it is set to True.
            Default: True.
    """

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer."""

        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """
        Tokenizes a piece of text using basic tokenizer.

        Args:
            text (str): A piece of text.

        Returns: 
            list(str): A list of tokens.
        """
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """
        Strips accents from a piece of text.
        """
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """
        Splits punctuation on a piece of text.
        """
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """
        Adds whitespace around any CJK character.
        """
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """
        Checks whether CP is the codepoint of a CJK character.
        """

        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """
        Performs invalid character removal and whitespace cleanup on text.
        """
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """
    Runs WordPiece tokenization.

    Args:
        vocab (Vocab|dict): Vocab of the word piece tokenizer.
        unk_token (str):  A specific token to replace all unkown tokens.
        max_input_chars_per_word (int):  If a word's length is more than max_input_chars_per_word, it will be 
            dealt as unknown word. Default: 100.
    """

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `BasicTokenizer`.

        Returns:
            list (str): A list of wordpiece tokens.

        Example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class BertTokenizer(PreTrainedTokenizer):
    """
    Constructs a BERT tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.

    Args:
        vocab_file (str): file path of the vocabulary
        do_lower_case (bool): Whether the text strips accents and convert to
            lower case. If you use the BERT pretrained model, lower is set to
            Flase when using the cased model, otherwise it is set to True.
            Default: True.
        unk_token (str): The special token for unkown words. Default: "[UNK]".
        sep_token (str): The special token for separator token . Default: "[SEP]".
        pad_token (str): The special token for padding. Default: "[PAD]".
        cls_token (str): The special token for cls. Default: "[CLS]".
        mask_token (str): The special token for mask. Default: "[MASK]".
    
    Examples:

        .. code-block:: python

            from paddle.hapi.text import BertTokenizer

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # the following line get: ['he', 'was', 'a', 'puppet', '##eer']
            tokens = tokenizer('He was a puppeteer')
            # the following line get: 'he was a puppeteer'
            tokenizer.convert_tokens_to_string(tokens)
    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "bert-base-uncased":
            "https://paddle-hapi.bj.bcebos.com/models/bert/bert-base-uncased-vocab.txt",
            "bert-large-uncased":
            "https://paddle-hapi.bj.bcebos.com/models/bert/bert-large-uncased-vocab.txt",
            "bert-base-cased":
            "https://paddle-hapi.bj.bcebos.com/models/bert/bert-base-cased-vocab.txt",
            "bert-large-cased":
            "https://paddle-hapi.bj.bcebos.com/models/bert/bert-large-cased-vocab.txt",
            "bert-base-multilingual-uncased":
            "https://paddle-hapi.bj.bcebos.com/models/bert/bert-base-multilingual-uncased-vocab.txt",
            "bert-base-multilingual-cased":
            "https://paddle-hapi.bj.bcebos.com/models/bert/bert-base-multilingual-cased-vocab.txt",
            "bert-base-chinese":
            "https://paddle-hapi.bj.bcebos.com/models/bert/bert-base-chinese-vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "bert-base-uncased": {
            "do_lower_case": True
        },
        "bert-large-uncased": {
            "do_lower_case": True
        },
        "bert-base-cased": {
            "do_lower_case": False
        },
        "bert-large-cased": {
            "do_lower_case": False
        },
        "bert-base-multilingual-uncased": {
            "do_lower_case": True
        },
        "bert-base-multilingual-cased": {
            "do_lower_case": False
        },
        "bert-base-chinese": {
            "do_lower_case": False
        }
    }

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]"):

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)
        self.basic_tokenizer = BertBasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token=unk_token)

    @property
    def vocab_size(self):
        """
        return the size of vocabulary.

        Returns:
            int: the size of vocabulary.
        """
        return len(self.vocab)

    def _tokenize(self, text):
        """
        End-to-end tokenization for BERT models.

        Args:
            text (str): The text to be tokenized.
        
        Returns:
            list: A list of string representing converted tokens.
        """
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def __call__(self, text):
        """
        End-to-end tokenization for BERT models.

        Args:
            text (str): The text to be tokenized.
        
        Returns:
            list: A list of string representing converted tokens.
        """
        return self._tokenize(text)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (list of string) in a single string. Since
        the usage of WordPiece introducing `##` to concat subwords, also remove
        `##` when converting.

        Args:
            tokens (list): A list of string representing tokens to be converted.

        Returns:
            str: Converted string from tokens.
        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string
