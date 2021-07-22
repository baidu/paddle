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

from paddle.incubate.hapi.text.data_utils import PreTrainedTokenizer, BertBasicTokenizer, WordpieceTokenizer, BertTokenizer, convert_to_unicode, whitespace_tokenize, Vocab
import unittest
import chardet
import tempfile
import os
import shutil
import json
MODEL_NAME_OR_PATH = "bert-base-uncased"


class TestPreProcess(unittest.TestCase):
    def test_convert_to_unicode(self):
        text1 = "中国".encode("utf-8")
        text1 = convert_to_unicode(text1)
        text2 = "abc"
        text2 = convert_to_unicode(text2)

    def test_whitespace_tokenize(self):
        text1 = " I love PaddlePaddle  "
        tokens1 = whitespace_tokenize(text1)
        for token in tokens1:
            assert " " not in token
        text2 = ""
        tokens2 = whitespace_tokenize(text2)
        assert tokens2 == []


class TestBertBasicTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BertBasicTokenizer(do_lower_case=True)

    def test_tokenize(self):
        text = "Absolutely  useful 我觉得很好用Absolutelyuseful"
        output_tokens = self.tokenizer.tokenize(text)
        assert output_tokens == [
            'absolutely', 'useful', '我', '觉', '得', '很', '好', '用',
            'absolutelyuseful'
        ]


class TestWordpieceTokenizer(unittest.TestCase):
    def setUp(self):
        vocab = ["AI", "Paddle", "I", "Deep", "Learning", "##Paddle", "."]
        unk_token = "UNK"
        self.tokenizer = WordpieceTokenizer(vocab, unk_token)

    def test_tokenize(self):
        text = "PaddlePaddle Deep learning ."
        output_tokens = self.tokenizer.tokenize(text)
        assert output_tokens == ['Paddle', '##Paddle', 'Deep', 'UNK', '.']


class TestBertTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

    def test_vocab_size(self):
        vocab_size = self.tokenizer.vocab_size

    def test_call(self):
        text = "AI and PaddlePaddle."
        tokens = self.tokenizer(text)
        assert tokens == ['ai', 'and', 'paddle', '##pad', '##dle', '.']

    def test_convert_tokens_to_string(self):
        tokens = ['ai', 'and', 'paddle', '##pad', '##dle', '.']
        output_string = self.tokenizer.convert_tokens_to_string(tokens)
        assert output_string == 'ai and paddlepaddle .'

    def test_tokens_to_ids(self):
        tokens = ["a", "an", "the"]
        converted_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

    def test_tokens_to_str(self):
        tokens = ["a", "an", "the"]
        string = self.tokenizer.convert_tokens_to_string(tokens)
        print(string)
        assert string == "a an the"

    def test_save_pretrained_and_resources(self):
        save_dir = tempfile.mkdtemp()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test_load_save_vocabulary(self):
        vocab_file = "vocab"
        token_to_idx = {
            "Paddle": 0,
            "AI": 1,
            "Baidu": 2,
            "Deep": 3,
            "Learning": 4,
            '<unk>': 5
        }
        vocab = Vocab(token_to_idx=token_to_idx)
        self.tokenizer.save_vocabulary(vocab_file, vocab)
        vocab = self.tokenizer.load_vocabulary(vocab_file)
        os.remove(vocab_file)


class TestVocab(unittest.TestCase):
    def setUp(self):
        # special token should be listed.
        token_to_idx = {
            "Paddle": 0,
            "AI": 1,
            "Baidu": 2,
            "Deep": 3,
            "Learning": 4,
            '<unk>': 5
        }
        self.vocab = Vocab(token_to_idx=token_to_idx)
        counter = {"Paddle": 2, "AI": 3, "Learning": 2, "Deep": 1}
        vocab = Vocab(counter, min_freq=2)
        assert "Deep" not in vocab.idx_to_token

    def test_to_tokens(self):
        indices = [1, 3, 4]
        tokens = self.vocab.to_tokens(indices)
        assert tokens == ['AI', 'Deep', 'Learning']
        indices = 1
        tokens = self.vocab.to_tokens(indices)
        assert tokens == 'AI'

    def test_to_indices(self):
        tokens = ['AI', 'Deep', 'Learning']
        indices = self.vocab.to_indices(tokens)
        assert indices == [1, 3, 4]

    def test_get_item(self):
        assert self.vocab['AI'] == 1
        assert self.vocab[['AI', 'Deep', 'Learning']] == [1, 3, 4]

    def test_call(self):
        assert self.vocab('AI') == 1

    def test_idx_to_token(self):
        # test in est_build_vocab
        # print(self.vocab.token_to_idx)
        pass

    def test_token_to_dix(self):
        # test in test_from_dict
        # print(self.vocab.token_to_idx)
        pass

    def test_to_from_json(self):
        path = "json_str"
        json_str = self.vocab.to_json(path)
        vocab_dict = json.loads(json_str)
        vocab_dict["identifiers_to_tokens"].pop("unk_token")  # it's needed.
        json_str = json.dumps(vocab_dict)
        vocab = Vocab.from_json(json_str)
        os.remove(path) if path else None

    def test_from_dict(self):
        token_to_idx = self.vocab.token_to_idx
        vocab = Vocab.from_dict(token_to_idx, unk_token='<unk>')

    def test_build_vocab(self):
        iterator = [['Deep', 'Learning'], ['Paddle']]
        vocab = Vocab.build_vocab(iterator)
        vocab.idx_to_token == [
            '<bos>', '<eos>', '<pad>', '<unk>', 'Deep', 'Learning', 'Paddle'
        ]


if __name__ == '__main__':
    unittest.main()
