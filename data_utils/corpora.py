# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
"""several datasets with preset arguments"""
from olfmlm.data_utils.datasets import json_dataset, csv_dataset

from olfmlm.paths import train_data_path
import os

class wikipedia(json_dataset):
    """
    dataset for wikipedia with arguments configured for convenience

    command line usage: `--train-data wikipedia`
    """
    PATH = os.path.join(train_data_path, "wikipedia_sentences.lazy")
    assert_str = "make sure to set PATH at line 27 of data_utils/corpora.py"
    def __init__(self, **kwargs):
        assert wikipedia.PATH != '<wikipedia_path>', \
                                 wikipedia.assert_str
        if not kwargs:
            kwargs = {}
        kwargs['text_key'] = 'text'
        kwargs['loose_json'] = True
        super(wikipedia, self).__init__(wikipedia.PATH, **kwargs)


class bookcorpus(json_dataset):
    """
    dataset for wikipedia with arguments configured for convenience

    command line usage: `--train-data wikipedia`
    """
    PATH = os.path.join(train_data_path, "bookcorpus.lazy")
    assert_str = "make sure to set PATH at line 27 of data_utils/corpora.py"
    def __init__(self, **kwargs):
        assert bookcorpus.PATH != '<bookcorpus_path>', \
                                   bookcorpus.assert_str
        if not kwargs:
            kwargs = {}
        kwargs['text_key'] = 'text'
        kwargs['loose_json'] = True
        super(bookcorpus, self).__init__(bookcorpus.PATH, **kwargs)


class gutenberg(json_dataset):
    """
    dataset for wikipedia with arguments configured for convenience

    command line usage: `--train-data wikipedia`
    """
    PATH = os.path.join(train_data_path, "Gutenberg.lazy")
    assert_str = "make sure to set PATH at line 27 of data_utils/corpora.py"
    def __init__(self, **kwargs):
        assert gutenberg.PATH != '<gutenberg>', gutenberg.assert_str
        if not kwargs:
            kwargs = {}
        kwargs['text_key'] = 'text'
        kwargs['loose_json'] = True
        super(gutenberg, self).__init__(gutenberg.PATH, **kwargs)


class cnn_dailymail(json_dataset):
    """
    dataset for wikipedia with arguments configured for convenience

    command line usage: `--train-data wikipedia`
    """
    PATH = os.path.join(train_data_path, "cnn_dailymail.lazy")
    assert_str = "make sure to set PATH at line 27 of data_utils/corpora.py"
    def __init__(self, **kwargs):
        assert bookcorpus.PATH != '<cnn_dailymail>', cnn_dailymail.assert_str
        if not kwargs:
            kwargs = {}
        kwargs['text_key'] = 'text'
        kwargs['loose_json'] = True
        super(cnn_dailymail, self).__init__(cnn_dailymail.PATH, **kwargs)


class bert_corpus(json_dataset):
    """
    dataset for bert corpus (wikipedia/book) with arguments configured for convenience

    command line usage: `--train-data wikipedia`
    """
    PATH = os.path.join(train_data_path, "bert_corpus.lazy")
    assert_str = "make sure to set PATH at line 27 of data_utils/corpora.py"
    def __init__(self, **kwargs):
        assert bert_corpus.PATH != '<cnn_dailymail>', bert_corpus.assert_str
        if not kwargs:
            kwargs = {}
        kwargs['text_key'] = 'text'
        kwargs['loose_json'] = True
        super(bert_corpus, self).__init__(bert_corpus.PATH, **kwargs)


NAMED_CORPORA = {
    'wikipedia': wikipedia,
    'bookcorpus': bookcorpus,
    'cnn_dailymail': cnn_dailymail,
    'gutenberg': gutenberg,
    'bert_corpus': bert_corpus
}
