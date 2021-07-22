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
import collections
import io
import os
import warnings

from paddle.io import Dataset
from paddle.dataset.common import DATA_HOME, md5file
from paddle.incubate.hapi.download import get_path_from_url

from .dataset import TSVDataset

__all__ = [
    'GlueCoLA',
    'GlueSST2',
    'GlueMRPC',
    'GlueSTSB',
    'GlueQQP',
    'GlueMNLI',
    'GlueQNLI',
    'GlueRTE',
    'GlueWNLI',
]


class _GlueDataset(TSVDataset):
    URL = None
    MD5 = None
    SEGMENT_INFO = collections.namedtuple(
        'SEGMENT_INFO', ('file', 'md5', 'field_indices', 'num_discard_samples'))
    SEGMENTS = {}  # mode: file, md5, field_indices, num_discard_samples

    def __init__(self,
                 segment='train',
                 root=None,
                 return_all_fields=False,
                 **kwargs):
        if return_all_fields:
            # self.SEGMENTS = copy.deepcopy(self.__class__.SEGMENTS)
            # self.SEGMENTS[segment].field_indices = segments
            segments = copy.deepcopy(self.__class__.SEGMENTS)
            segment_info = list(segments[segment])
            segment_info[2] = None
            segments[segment] = self.SEGMENT_INFO(*segment_info)
            self.SEGMENTS = segments

        self._get_data(root, segment, **kwargs)

    def _get_data(self, root, segment, **kwargs):
        default_root = os.path.join(DATA_HOME, 'glue')
        filename, data_hash, field_indices, num_discard_samples = self.SEGMENTS[
            segment]
        fullname = os.path.join(default_root,
                                filename) if root is None else os.path.join(
                                    os.path.expanduser(root), filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            if root is not None:  # not specified, and no need to warn
                warnings.warn(
                    'md5 check failed for {}, download {} data to {}'.format(
                        filename, self.__class__.__name__, default_root))
            path = get_path_from_url(self.URL, default_root, self.MD5)
            fullname = os.path.join(default_root, filename)
        super(_GlueDataset, self).__init__(
            fullname,
            field_indices=field_indices,
            num_discard_samples=num_discard_samples,
            **kwargs)


class GlueCoLA(_GlueDataset):
    """
    The Corpus of Linguistic Acceptability (Warstadt et al., 2018) consists of
    English acceptability judgments drawn from books and journal articles on
    linguistic theory.
    Each example is a sequence of words annotated with whether it is a
    grammatical English sentence. From https://gluebenchmark.com/tasks

    Args:
        segment ('train'|'dev'|'test'): Dataset segment. Default: 'train'.
        root (str): Path to temp folder for storing data.
        return_all_fields (bool): Return all fields available in the dataset.
            Default: False.

    Example:

        .. code-block:: python
            from paddle.incubate.hapi.text.glue import GlueCoLA
            cola_dev = GlueCoLA('dev', root='./datasets/cola')
            len(cola_dev) # 1043
            len(cola_dev[0]) # 2

            # ['The sailors rode the breeze clear of the rocks.', '1']
            cola_dev[0] 
            cola_test = GlueCoLA('test', root='./datasets/cola')
            len(cola_test) # 1063
            len(cola_test[0]) # 1
            cola_test[0] # ['Bill whistled past the house.']

    """
    URL = "https://dataset.bj.bcebos.com/glue/CoLA.zip"
    MD5 = 'b178a7c2f397b0433c39c7caf50a3543'
    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('CoLA', 'train.tsv'),
            'c79d4693b8681800338aa044bf9e797b', (3, 1), 0),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('CoLA', 'dev.tsv'), 'c5475ccefc9e7ca0917294b8bbda783c',
            (3, 1), 0),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('CoLA', 'test.tsv'),
            'd8721b7dedda0dcca73cebb2a9f4259f', (1, ), 1)
    }

    def get_labels(self):
        """
        Return labels of the GlueCoLA object.
        """
        return ["0", "1"]


class GlueSST2(_GlueDataset):
    """
    The Stanford Sentiment Treebank (Socher et al., 2013) consists of sentences
    from movie reviews and human annotations of their sentiment.
    From https://gluebenchmark.com/tasks

    Args:
        segment ('train'|'dev'|'test'): Dataset segment. Default: 'train'.
        root (str): Path to temp folder for storing data.
        return_all_fields (bool): Return all fields available in the dataset.
            Default: False.

    Examples:
        .. code-block:: python
            from paddle.incubate.hapi.text.glue import GlueSST2
            sst_dev = GlueSST2('dev', root='./datasets/sst')
            len(sst_dev) # 872
            len(sst_dev[0]) # 2
            # ["it 's a charming and often affecting journey . ", '1']
            sst_dev[0] 
            sst_test = GlueSST2('test', root='./datasets/sst')
            len(sst_test) # 1821
            len(sst_test[0]) # 1
            sst_test[0] # ['uneasy mishmash of styles and genres .']

    """

    URL = 'https://dataset.bj.bcebos.com/glue/SST.zip'
    MD5 = '9f81648d4199384278b86e315dac217c'

    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('SST-2', 'train.tsv'),
            'da409a0a939379ed32a470bc0f7fe99a', (0, 1), 1),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('SST-2', 'dev.tsv'),
            '268856b487b2a31a28c0a93daaff7288', (0, 1), 1),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('SST-2', 'test.tsv'),
            '3230e4efec76488b87877a56ae49675a', (1, ), 1)
    }

    def get_labels(self):
        """
        Return labels of the GlueSST2 object.
        """
        return ["0", "1"]


class GlueMRPC(_GlueDataset):
    """
    The Microsoft Research Paraphrase Corpus dataset.
    From https://gluebenchmark.com/tasks

    Args:
        root (str): Path to temp folder for storing data.
        segment ('train'|'dev'|'test'): Dataset segment. Default: 'train'.

    Example:
        .. code-block:: python

            from paddle.incubate.hapi.text.glue import  GlueMRPC
            mrpc_dev = GlueMRPC('dev', root='./datasets/mrpc')
            len(mrpc_dev) # 408
            len(mrpc_dev[0]) # 3
            mrpc_dev[0] # ["He said the foodservice pie business doesn 't fit
                        # the company 's long-term growth strategy .", 
                        # '" The foodservice pie business does not fit our 
                        # long-term growth strategy .', '1']
            mrpc_test = GlueMRPC('test', root='./datasets/mrpc')
            len(mrpc_test) # 1725
            len(mrpc_test[0]) # 2
            mrpc_test[0] 
            # ["PCCW 's chief operating officer , Mike Butcher , and Alex Arena ,
            #  the chief financial officer , will report directly to Mr So .", 
            # 'Current Chief Operating Officer Mike Butcher and Group Chief
            # Financial Officer Alex Arena will report to So .']
    """

    DEV_ID_URL = 'https://dataset.bj.bcebos.com/glue/mrpc/dev_ids.tsv'
    DEV_ID_MD5 = '7ab59a1b04bd7cb773f98a0717106c9b'
    TRAIN_DATA_URL = 'https://dataset.bj.bcebos.com/glue/mrpc/msr_paraphrase_train.txt'
    TRAIN_DATA_MD5 = '793daf7b6224281e75fe61c1f80afe35'
    TEST_DATA_URL = 'https://dataset.bj.bcebos.com/glue/mrpc/msr_paraphrase_test.txt'
    TEST_DATA_MD5 = 'e437fdddb92535b820fe8852e2df8a49'

    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('MRPC', 'train.tsv'),
            'dc2dac669a113866a6480a0b10cd50bf', (3, 4, 0), 1),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('MRPC', 'dev.tsv'), '185958e46ba556b38c6a7cc63f3a2135',
            (3, 4, 0), 1),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('MRPC', 'test.tsv'),
            '4825dab4b4832f81455719660b608de5', (3, 4), 1)
    }

    def _get_data(self, root, segment, **kwargs):
        default_root = os.path.join(DATA_HOME, 'glue')
        filename, data_hash, field_indices, num_discard_samples = self.SEGMENTS[
            segment]
        fullname = os.path.join(default_root,
                                filename) if root is None else os.path.join(
                                    os.path.expanduser(root), filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            if root is not None:  # not specified, and no need to warn
                warnings.warn(
                    'md5 check failed for {}, download {} data to {}'.format(
                        filename, self.__class__.__name__, default_root))
            if segment in ('train', 'dev'):
                dev_id_path = get_path_from_url(
                    self.DEV_ID_URL,
                    os.path.join(default_root, 'MRPC'), self.DEV_ID_MD5)
                train_data_path = get_path_from_url(
                    self.TRAIN_DATA_URL,
                    os.path.join(default_root, 'MRPC'), self.TRAIN_DATA_MD5)
                # read dev data ids
                dev_ids = []
                with io.open(dev_id_path, encoding='utf-8') as ids_fh:
                    for row in ids_fh:
                        dev_ids.append(row.strip().split('\t'))

                # generate train and dev set
                train_path = os.path.join(default_root, 'MRPC', 'train.tsv')
                dev_path = os.path.join(default_root, 'MRPC', 'dev.tsv')
                with io.open(train_data_path, encoding='utf-8') as data_fh:
                    with io.open(train_path, 'w', encoding='utf-8') as train_fh:
                        with io.open(dev_path, 'w', encoding='utf8') as dev_fh:
                            header = data_fh.readline()
                            train_fh.write(header)
                            dev_fh.write(header)
                            for row in data_fh:
                                label, id1, id2, s1, s2 = row.strip().split(
                                    '\t')
                                example = '%s\t%s\t%s\t%s\t%s\n' % (label, id1,
                                                                    id2, s1, s2)
                                if [id1, id2] in dev_ids:
                                    dev_fh.write(example)
                                else:
                                    train_fh.write(example)
            else:
                test_data_path = get_path_from_url(
                    self.TEST_DATA_URL,
                    os.path.join(default_root, 'MRPC'), self.TEST_DATA_MD5)
                test_path = os.path.join(default_root, 'MRPC', 'test.tsv')
                with io.open(test_data_path, encoding='utf-8') as data_fh:
                    with io.open(test_path, 'w', encoding='utf-8') as test_fh:
                        header = data_fh.readline()
                        test_fh.write(
                            'index\t#1 ID\t#2 ID\t#1 String\t#2 String\n')
                        for idx, row in enumerate(data_fh):
                            label, id1, id2, s1, s2 = row.strip().split('\t')
                            test_fh.write('%d\t%s\t%s\t%s\t%s\n' %
                                          (idx, id1, id2, s1, s2))
            root = default_root
        super(GlueMRPC, self)._get_data(root, segment, **kwargs)

    def get_labels(self):
        """
        Return labels of the GlueMRPC object.
        """
        return ["0", "1"]


class GlueSTSB(_GlueDataset):
    """
    The Semantic Textual Similarity Benchmark (Cer et al., 2017) is a
    collection of sentence pairs drawn from news headlines, video and image
    captions, and natural language inference data. Each pair is human-annotated
    with a similarity score from 1 to 5.

    From https://gluebenchmark.com/tasks

    Args:
        segment ('train'|'dev'|'test'): Dataset segment. Default: 'train'.
        root (str): Path to temp folder for storing data.
        return_all_fields (bool): Return all fields available in the dataset. Default: False.

    Example:
        .. code-block:: python
            from paddle.incubate.hapi.text.glue import GlueSTSB
            stsb_dev = GlueSTSB('dev', root='./datasets/stsb')
            len(stsb_dev) # 1500
            len(stsb_dev[0]) # 3
            stsb_dev[0] # ['A man with a hard hat is dancing.', 'A man wearing a hard hat is dancing.', '5.000']
            stsb_test = GlueSTSB('test', root='./datasets/stsb')
            len(stsb_test) # 1379
            len(stsb_test[0]) # 2
            stsb_test[0] # ['A girl is styling her hair.', 'A girl is brushing her hair.']
    """
    URL = 'https://dataset.bj.bcebos.com/glue/STS.zip'
    MD5 = 'd573676be38f1a075a5702b90ceab3de'

    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('STS-B', 'train.tsv'),
            '4f7a86dde15fe4832c18e5b970998672', (7, 8, 9), 1),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('STS-B', 'dev.tsv'),
            '5f4d6b0d2a5f268b1b56db773ab2f1fe', (7, 8, 9), 1),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('STS-B', 'test.tsv'),
            '339b5817e414d19d9bb5f593dd94249c', (7, 8), 1)
    }

    def get_labels(self):
        """
        Return labels of the GlueSTSB object.
        """
        return None


class GlueQQP(_GlueDataset):
    """
    The Quora Question Pairs dataset is a collection of question pairs from the
    community question-answering website Quora.
    From https://gluebenchmark.com/tasks

    Args:
        segment ({'train', 'dev', 'test'}): Dataset segment. Default: 'train'.
        root (str): Path to temp folder for storing data.
        return_all_fields (bool): Return all fields available in the dataset.
            Default: False.

    Example:
        .. code-block:: python

            from paddle.incubate.hapi.text.glue import GlueQQP
            import warnings
            with warnings.catch_warnings():
                # Ignore warnings triggered by invalid entries in GlueQQP dev set
                warnings.simplefilter("ignore")
                qqp_dev = GlueQQP('dev', root='./datasets/qqp')

            len(qqp_dev) # 40430
            len(qqp_dev[0]) # 3
            qqp_dev[0] # ['Why are African-Americans so beautiful?', 
                    # 'Why are hispanics so beautiful?', '0']
            qqp_test = GlueQQP('test', root='./datasets/qqp')
            len(qqp_test) # 390965
            len(qqp_test[3]) # 2
            qqp_test[3] # ['Is it safe to invest in social trade biz?',
                    # 'Is social trade geniune?']
    """
    URL = 'https://dataset.bj.bcebos.com/glue/QQP.zip'
    MD5 = '884bf26e39c783d757acc510a2a516ef'

    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('QQP', 'train.tsv'),
            'e003db73d277d38bbd83a2ef15beb442', (3, 4, 5), 1),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('QQP', 'dev.tsv'), 'cff6a448d1580132367c22fc449ec214',
            (3, 4, 5), 1),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('QQP', 'test.tsv'), '73de726db186b1b08f071364b2bb96d0',
            (1, 2), 1)
    }

    def __init__(self, segment='train', root=None, return_all_fields=False):
        # QQP may include broken samples
        super(GlueQQP, self).__init__(
            segment, root, return_all_fields, allow_missing=True)

    def get_labels(self):
        """
        Return labels of the GlueQQP object.
        """
        return ["0", "1"]


class GlueMNLI(_GlueDataset):
    """
    The Multi-Genre Natural Language Inference Corpus (Williams et al., 2018)
    is a crowdsourced collection of sentence pairs with textual entailment
    annotations.
    From https://gluebenchmark.com/tasks

    Args:
        segment ('train'|'dev_matched'|'dev_mismatched'|'test_matched'|
            'test_mismatched'): Dataset segment. Default: ‘train’.
        root (str, default '$MXNET_HOME/datasets/glue_mnli'): Path to temp
            folder for storing data.
        return_all_fields (bool): Return all fields available in the dataset.
            Default: False.

    Example:
        .. code-block:: python
            from paddle.incubate.hapi.text.glue import GlueMNLI
            mnli_dev = GlueMNLI('dev_matched', root='./datasets/mnli')
            len(mnli_dev) # 9815
            len(mnli_dev[0]) # 3
            mnli_dev[0] # ['The new rights are nice enough', 
                        # 'Everyone really likes the newest benefits ', 
                        # 'neutral']
            mnli_test = GlueMNLI('test_matched', root='./datasets/mnli')
            len(mnli_test) # 9796
            len(mnli_test[0]) # 2
            mnli_test[0] # ['Hierbas, ans seco, ans dulce, and frigola are 
                            # just a few names worth keeping a look-out for.', 
                            # 'Hierbas is a name worth looking out for.']

    """
    URL = 'https://dataset.bj.bcebos.com/glue/MNLI.zip'
    MD5 = 'e343b4bdf53f927436d0792203b9b9ff'

    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('MNLI', 'train.tsv'),
            '220192295e23b6705f3545168272c740', (8, 9, 11), 1),
        'dev_matched': _GlueDataset.SEGMENT_INFO(
            os.path.join('MNLI', 'dev_matched.tsv'),
            'c3fa2817007f4cdf1a03663611a8ad23', (8, 9, 15), 1),
        'dev_mismatched': _GlueDataset.SEGMENT_INFO(
            os.path.join('MNLI', 'dev_mismatched.tsv'),
            'b219e6fe74e4aa779e2f417ffe713053', (8, 9, 15), 1),
        'test_matched': _GlueDataset.SEGMENT_INFO(
            os.path.join('MNLI', 'test_matched.tsv'),
            '33ea0389aedda8a43dabc9b3579684d9', (8, 9), 1),
        'test_mismatched': _GlueDataset.SEGMENT_INFO(
            os.path.join('MNLI', 'test_mismatched.tsv'),
            '7d2f60a73d54f30d8a65e474b615aeb6', (8, 9), 1),
    }

    def get_labels(self):
        """
        Return labels of the GlueMNLI object.
        """
        return ["contradiction", "entailment", "neutral"]


class GlueQNLI(_GlueDataset):
    """
    The Question-answering NLI dataset converted from Stanford Question
    Answering Dataset (Rajpurkar et al. 2016).
    From https://gluebenchmark.com/tasks

    Args:
        segment ('train'|'dev'|'test'): Dataset segment. Dataset segment.
            Default: 'train'.
        root (str): Path to temp folder for storing data.
        return_all_fields (bool): Return all fields available in the dataset.
            Default: False.
       
    Example:
        .. code-block:: python
            from paddle.incubate.hapi.text.glue import GlueQNLI
            qnli_dev = GlueQNLI('dev', root='./datasets/qnli')
            len(qnli_dev) # 5732
            len(qnli_dev[0]) # 3
            qnli_dev[0] # ['Which NFL team represented the AFC at Super Bowl 
                        # 50?', 'The American Football Conference (AFC) 
                        # champion Denver Broncos defeated the National 
                        # Football Conference (NFC) champion Carolina Panthers
                        # 24\u201310 to earn their third Super Bowl title.', 
                        # 'entailment']
            qnli_test = GlueQNLI('test', root='./datasets/qnli')
            len(qnli_test) # 5740
            len(qnli_test[0]) # 2
            qnli_test[0] # ['What seldom used term of a unit of force equal to
                         # 1000 pound s of force?', 
                         # 'Other arcane units of force include the sthène,
                         # which is equivalent to 1000 N, and the kip, which
                         # is equivalent to 1000 lbf.']
    """
    URL = 'https://dataset.bj.bcebos.com/glue/QNLI.zip'
    MD5 = 'b4efd6554440de1712e9b54e14760e82'
    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('QNLI', 'train.tsv'),
            '5e6063f407b08d1f7c7074d049ace94a', (1, 2, 3), 1),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('QNLI', 'dev.tsv'), '1e81e211959605f144ba6c0ad7dc948b',
            (1, 2, 3), 1),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('QNLI', 'test.tsv'),
            'f2a29f83f3fe1a9c049777822b7fa8b0', (1, 2), 1)
    }

    def get_labels(self):
        """
        Return labels of the GlueQNLI object.
        """
        return ["entailment", "not_entailment"]


class GlueRTE(_GlueDataset):
    """
    The Recognizing Textual Entailment (RTE) datasets come from a series of
    annual textual entailment challenges (RTE1, RTE2, RTE3, and RTE5).
    From https://gluebenchmark.com/tasks
    Args:
        segment ('train'|'dev'|'test'): Dataset segment. Default: 'train'.
        root (str): Path to temp folder for storing data.
        return_all_fields (bool): Return all fields available in the dataset.
            Default: False.

    Examples:
        .. code-block:: python
            from paddle.incubate.hapi.text.glue import GlueRTE
            rte_dev = GlueRTE('dev', root='./datasets/rte')
            len(rte_dev) # 277
            len(rte_dev[0]) # 3
            rte_dev[0] # ['Dana Reeve, the widow of the actor Christopher 
                       # Reeve, has died of lung cancer at age 44, according
                       # to the Christopher Reeve Foundation.', 'Christopher
                       # Reeve had an accident.', 'not_entailment']
            rte_test = GlueRTE('test', root='./datasets/rte')
            len(rte_test) # 3000
            len(rte_test[16]) # 2
            rte_test[16] # ['United failed to progress beyond the group stages
                         # of the Champions League and trail in the Premiership
                         # title race, sparking rumours over its future.', 
                         # 'United won the Champions League.']
    """
    URL = 'https://dataset.bj.bcebos.com/glue/RTE.zip'
    MD5 = 'bef554d0cafd4ab6743488101c638539'

    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('RTE', 'train.tsv'),
            'd2844f558d111a16503144bb37a8165f', (1, 2, 3), 1),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('RTE', 'dev.tsv'), '973cb4178d4534cf745a01c309d4a66c',
            (1, 2, 3), 1),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('RTE', 'test.tsv'), '6041008f3f3e48704f57ce1b88ad2e74',
            (1, 2), 1)
    }

    def get_labels(self):
        """
        Return labels of the GlueRTE object.
        """
        return ["entailment", "not_entailment"]


class GlueWNLI(_GlueDataset):
    """
    The Winograd NLI dataset converted from the dataset in Winograd Schema
    Challenge (Levesque et al., 2011).
    From https://gluebenchmark.com/tasks

    Args:
        segment ('train'|'dev'|'test'): Dataset segment. Default: 'train'.
        root (str): Path to temp folder for storing data.
        return_all_fields (bool): Return all fields available in the dataset.
            Default: False.

    Example:
        .. code-block:: python
            from paddle.incubate.hapi.text.glue import GlueWNLI
            wnli_dev = GlueWNLI('dev', root='./datasets/wnli')
            len(wnli_dev) # 71
            len(wnli_dev[0]) # 3
            wnli_dev[0] # ['The drain is clogged with hair. It has to be 
                        # cleaned.', 'The hair has to be cleaned.', '0']
            wnli_test = GlueWNLI('test', root='./datasets/wnli')
            len(wnli_test) # 146
            len(wnli_test[0]) # 2
            wnli_test[0] # ['Maude and Dora had seen the trains rushing 
                            # across the prairie, with long, rolling puffs 
                            # of black smoke streaming back from the engine.
                            # Their roars and their wild, clear whistles 
                            # could be heard from far away. Horses ran away 
                            # when they came in sight.', 'Horses ran away when
                            # Maude and Dora came in sight.']
    """
    URL = 'https://dataset.bj.bcebos.com/glue/WNLI.zip'
    MD5 = 'a1b4bd2861017d302d29e42139657a42'

    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('WNLI', 'train.tsv'),
            '5cdc5a87b7be0c87a6363fa6a5481fc1', (1, 2, 3), 1),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('WNLI', 'dev.tsv'), 'a79a6dd5d71287bcad6824c892e517ee',
            (1, 2, 3), 1),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('WNLI', 'test.tsv'),
            'a18789ba4f60f6fdc8cb4237e4ba24b5', (1, 2), 1)
    }

    def get_labels(self):
        """
        Return labels of the GlueWNLI object.
        """
        return ["0", "1"]
