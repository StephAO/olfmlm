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
"""dataset objects for jsons, csvs, and BERT datasets"""

import os
import time
from operator import itemgetter
from bisect import bisect_right
import json
import csv
import math
import random

from torch.utils import data
import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')
from nltk import tokenize

from .lazy_loader import lazy_array_loader, exists_lazy, make_lazy
from .tokenization import Tokenization

class ConcatDataset(data.Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated.
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, **kwargs):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        self._X = None
        self._Y = None

    def SetTokenizer(self, tokenizer):
        for ds in self.datasets:
            ds.SetTokenizer(tokenizer)

    def GetTokenizer(self):
        return self.datasets[0].GetTokenizer()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def X(self):
        if self._X is None:
            self._X = []
            for data in self.datasets:
                self._X.extend(data.X)
        return self._X

    @property
    def Y(self):
        if self._Y is None:
            self._Y = []
            for data in self.datasets:
                self._Y.extend(list(data.Y))
            self._Y = np.array(self._Y)
        return self._Y

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

class SplitDataset(data.Dataset):
    """
    Dataset wrapper to access a subset of another dataset.
    Purpose: useful to index into existing datasets, possibly
    large-scale datasets as the subindexing operation is done in an
    on-the-fly manner.
    Arguments:
        ds (Dataset or array-like): List of datasets to be subindexed
        split_inds (1D array-like): List of indices part of subset
    """
    def __init__(self, ds, split_inds, **kwargs):
        self.split_inds = list(split_inds)
        self.wrapped_data = ds
        self.is_lazy = isinstance(ds, lazy_array_loader)
        if self.is_lazy:
            self.lens = itemgetter(*self.split_inds)(list(self.wrapped_data.lens))
        self._X = None
        self._Y = None

    def __len__(self):
        return len(self.split_inds)

    def __getitem__(self, index):
        return self.wrapped_data[self.split_inds[index]]

    def SetTokenizer(self, tokenizer):
        self.wrapped_data.SetTokenizer(tokenizer)

    def GetTokenizer(self):
        return self.wrapped_data.GetTokenizer()

    @property
    def X(self):
        if self._X is None:
            self._X = itemgetter(*self.split_inds)(self.wrapped_data.X)
        return self._X

    @property
    def Y(self):
        if self._Y is None:
            self._Y = np.array(itemgetter(*self.split_inds)(self.wrapped_data.Y))
        return self._Y

    def __iter__(self):
        for idx in self.split_inds:
            yield self.wrapped_data[idx]

def split_ds(ds, split=[.8,.2,.0], shuffle=True):
    """
    Split a dataset into subsets given proportions of how
    much to allocate per split. If a split is 0% returns None for that split.
    Purpose: Useful for creating train/val/test splits
    Arguments:
        ds (Dataset or array-like): Data to be split.
        split (1D array-like): proportions to split `ds`. `sum(splits) != 0`
        shuffle (boolean): Randomly split dataset. Default: True
    """
    split_sum = sum(split)
    if split_sum == 0:
        raise Exception('Split cannot sum to 0.')
    split = np.array(split)
    split /= split_sum
    ds_len = len(ds)
    inds = np.arange(ds_len)
    if shuffle:
        np.random.shuffle(inds)
    start_idx = 0
    residual_idx = 0
    rtn_ds = [None]*len(split)
    for i, f in enumerate(split):
        if f != 0:
            proportion = ds_len*split[i]
            residual_idx += proportion % 1
            split_ = int(int(proportion) + residual_idx)
            split_inds = inds[start_idx:start_idx+max(split_, 1)]
            rtn_ds[i] = SplitDataset(ds, split_inds)
            start_idx += split_
            residual_idx %= 1
    return rtn_ds

class csv_dataset(data.Dataset):
    """
    Class for loading datasets from csv files.
    Purpose: Useful for loading data for unsupervised modeling or transfer tasks
    Arguments:
        path (str): Path to csv file with dataset.
        tokenizer (data_utils.Tokenizer): Tokenizer to use when processing text. Default: None
        preprocess_fn (callable): Callable that process a string into desired format.
        delim (str): delimiter for csv. Default: ','
        binarize_sent (bool): binarize label values to 0 or 1 if they\'re on a different scale. Default: False
        drop_unlabeled (bool): drop rows with unlabelled values. Always fills remaining empty
            columns with -1 (regardless if rows are dropped based on value) Default: False
        text_key (str): key to get text from csv. Default: 'sentence'
        label_key (str): key to get label from json dictionary. Default: 'label'
    Attributes:
        X (list): all strings from the csv file
        Y (np.ndarray): labels to train with
    """
    def __init__(self, path, tokenizer=None, preprocess_fn=None, delim=',',
                binarize_sent=False, drop_unlabeled=False, text_key='sentence', label_key='label',
                **kwargs):
        self.preprocess_fn = preprocess_fn
        self.SetTokenizer(tokenizer)
        self.path = path
        self.delim = delim
        self.text_key = text_key
        self.label_key = label_key
        self.drop_unlabeled = drop_unlabeled

        if '.tsv' in self.path:
            self.delim = '\t'


        self.X = []
        self.Y = []
        try:
            cols = [text_key]
            if isinstance(label_key, list):
                cols += label_key
            else:
                cols += [label_key]
            data = pd.read_csv(self.path, sep=self.delim, usecols=cols, encoding='latin-1')
        except:
            data = pd.read_csv(self.path, sep=self.delim, usecols=[text_key], encoding='latin-1')

        data = data.dropna(axis=0)

        self.X = data[text_key].values.tolist()
        try:
            self.Y = data[label_key].values
        except Exception as e:
            self.Y = np.ones(len(self.X))*-1

        if binarize_sent:
            self.Y = binarize_labels(self.Y, hard=binarize_sent)

    def SetTokenizer(self, tokenizer):
        if tokenizer is None:
            self.using_tokenizer = False
            if not hasattr(self, '_tokenizer'):
                self._tokenizer = tokenizer
        else:
            self.using_tokenizer = True
            self._tokenizer = tokenizer

    def GetTokenizer(self):
        return self._tokenizer

    @property
    def tokenizer(self):
        if self.using_tokenizer:
            return self._tokenizer
        return None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """process+tokenize string and return string,label,and stringlen"""
        x = self.X[index]
        if self.tokenizer is not None:
            x = self.tokenizer.EncodeAsIds(x, self.preprocess_fn)
        elif self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
        y = self.Y[index]
        if isinstance(y, str):
            if self.tokenizer is not None:
                y = self.tokenizer.EncodeAsIds(y, self.preprocess_fn)
            elif self.preprocess_fn is not None:
                y = self.preprocess_fn(y)
        return {'text': x, 'length': len(x), 'label': y}

    def write(self, writer_gen=None, path=None, skip_header=False):
        """
        given a generator of metrics for each of the data points X_i,
            write the metrics, text, and labels to a csv file
        """
        if path is None:
            path = self.path+'.results'
        print('generating csv at ' + path)
        with open(path, 'w') as csvfile:
            c = csv.writer(csvfile, delimiter=self.delim)
            if writer_gen is not None:
                #if first item of generator is a header of what the metrics mean then write header to csv file
                if not skip_header:
                    header = (self.label_key,)+tuple(next(writer_gen))+(self.text_key,)
                    c.writerow(header)
                for i, row in enumerate(writer_gen):
                    row = (self.Y[i],)+tuple(row)+(self.X[i],)
                    c.writerow(row)
            else:
                c.writerow([self.label_key, self.text_key])
                for row in zip(self.Y, self.X):
                    c.writerow(row)

class json_dataset(data.Dataset):
    """
    Class for loading datasets from a json dump.
    Purpose: Useful for loading data for unsupervised modeling or transfer tasks
    Arguments:
        path (str): path to json file with dataset.
        tokenizer (data_utils.Tokenizer): Tokenizer to use when processing text. Default: None
        preprocess_fn (callable): callable function that process a string into desired format.
            Takes string, maxlen=None, encode=None as arguments. Default: process_str
        text_key (str): key to get text from json dictionary. Default: 'sentence'
        label_key (str): key to get label from json dictionary. Default: 'label'
    Attributes:
        all_strs (list): list of all strings from the dataset
        all_labels (list): list of all labels from the dataset (if they have it)
    """
    def __init__(self, path, tokenizer=None, preprocess_fn=None, binarize_sent=False,
                text_key='sentence', label_key='label', loose_json=False, **kwargs):
        self.preprocess_fn = preprocess_fn
        self.path = path
        self.SetTokenizer(tokenizer)
        self.X = []
        self.Y = []
        self.text_key = text_key
        self.label_key = label_key
        self.loose_json = loose_json

        for j in self.load_json_stream(self.path):
            s = j[text_key]
            self.X.append(s)
            self.Y.append(j[label_key])

        if binarize_sent:
            self.Y = binarize_labels(self.Y, hard=binarize_sent)

    def SetTokenizer(self, tokenizer):
        if tokenizer is None:
            self.using_tokenizer = False
            if not hasattr(self, '_tokenizer'):
                self._tokenizer = tokenizer
        else:
            self.using_tokenizer = True
            self._tokenizer = tokenizer

    def GetTokenizer(self):
        return self._tokenizer

    @property
    def tokenizer(self):
        if self.using_tokenizer:
            return self._tokenizer
        return None

    def __getitem__(self, index):
        """gets the index'th string from the dataset"""
        x = self.X[index]
        if self.tokenizer is not None:
            x = self.tokenizer.EncodeAsIds(x, self.preprocess_fn)
        elif self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
        y = self.Y[index]
        if isinstance(y, str):
            if self.tokenizer is not None:
                y = self.tokenizer.EncodeAsIds(y, self.preprocess_fn)
            elif self.preprocess_fn is not None:
                y = self.preprocess_fn(y)
        return {'text': x, 'length': len(x), 'label': y}

    def __len__(self):
        return len(self.X)

    def write(self, writer_gen=None, path=None, skip_header=False):
        """
        given a generator of metrics for each of the data points X_i,
            write the metrics, text, and labels to a json file
        """
        if path is None:
            path = self.path+'.results'

        jsons = []

        if writer_gen is not None:
            #if first item of generator is a header of what the metrics mean then write header to csv file
            def gen_helper():
                keys = {}
                keys[0] = self.label_key
                if not skip_header:
                    for idx, k in enumerate(tuple(next(writer_gen))):
                        keys[idx+1] = k
                for i, row in enumerate(writer_gen):
                    if i == 0 and skip_header:
                        for idx, _ in enumerate(row):
                            keys[idx+1] = 'metric_%d'%(idx,)
                    j = {}
                    for idx, v in enumerate((self.Y[i],)+tuple(row)):
                        k = keys[idx]
                        j[k] = v
                    yield j
        else:
            def gen_helper():
                for y in self.Y:
                    j = {}
                    j[self.label_key] = y
                    yield j

        def out_stream():
            for i, j in enumerate(gen_helper()):
                j[self.text_key] = self.X[i]
                yield j

        self.save_json_stream(path, out_stream())

    def save_json_stream(self, save_path, json_stream):
        if self.loose_json:
            with open(save_path, 'w') as f:
                for i, j in enumerate(json_stream):
                    write_string = ''
                    if i != 0:
                        write_string = '\n'
                    write_string += json.dumps(j)
                    f.write(write_string)
        else:
            jsons = [j for j in json_stream]
            json.dump(jsons, open(save_path, 'w'), separators=(',', ':'))

    def load_json_stream(self, load_path):
        if not self.loose_json:
            jsons = json.load(open(load_path, 'r'))
            generator = iter(jsons)
        else:
            def gen_helper():
                with open(load_path, 'r') as f:
                    for row in f:
                        yield json.loads(row)
            generator = gen_helper()

        for j in generator:
            if self.label_key not in j:
                j[self.label_key] = -1
            yield j


class bert_dataset(data.Dataset):
    """
    Abstract bert dataset.
    Arguments:
        ds (Dataset or array-like): data corpus to use for training
        max_seq_len (int): maximum sequence length to use for a sentence pair
        mask_lm_prob (float): proportion of tokens to mask for masked LM
        max_preds_per_seq (int): Maximum number of masked tokens per sentence pair. Default: math.ceil(max_seq_len*mask_lm_prob/10)*10
        dataset_size (int): number of random sentencepairs in the dataset. Default: len(ds)*(len(ds)-1)

    """
    def __init__(self, use_types, ds, max_seq_len=512, mask_lm_prob=.15, max_preds_per_seq=None, short_seq_prob=.01, dataset_size=None, presplit_sentences=False, **kwargs):
        self.use_types = use_types
        self.ds = ds
        self.ds_len = len(self.ds)
        self.tokenizer = self.ds.GetTokenizer()
        self.vocab_words = list(self.tokenizer.text_token_vocab.values())
        self.ds.SetTokenizer(None)
        self.max_seq_len = max_seq_len
        self.mask_lm_prob = mask_lm_prob
        if max_preds_per_seq is None:
            max_preds_per_seq = math.ceil(max_seq_len*mask_lm_prob /10)*10
        self.max_preds_per_seq = max_preds_per_seq
        self.dataset_size = dataset_size
        if self.dataset_size is None:
            self.dataset_size = self.ds_len * (self.ds_len-1)
        self.presplit_sentences = presplit_sentences

    def __len__(self):
        return self.dataset_size

    def sentence_tokenize(self, sent, sentence_num=0, beginning=False, ending=False):
        """tokenize sentence and get token types if tokens=True"""
        tokens = self.tokenizer.EncodeAsIds(sent).tokenization
        token_types = None
        if self.use_types:
            str_type = 'str' + str(sentence_num)
            token_types = [self.tokenizer.get_type(str_type).Id]*len(tokens)
        return tokens, token_types

    def sentence_split(self, document):
        """split document into sentences"""
        lines = document.split('\n')
        if self.presplit_sentences:
            return [line for line in lines if line]
        rtn = []
        for line in lines:
            if line != '':
                rtn.extend(tokenize.sent_tokenize(line))
        return rtn

    def get_doc(self, idx):
        """gets text of document corresponding to idx"""
        rtn = self.ds[idx]
        if isinstance(rtn, dict):
            rtn = rtn['text']
        return rtn

    def truncate_seq_pair(self, a, b, max_seq_len, rng):
        """
        Truncate sequence pair according to original BERT implementation:
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L391
        """
        tokens_a, token_types_a = a
        tokens_b, token_types_b = b
        max_num_tokens = max_seq_len - 3
        pop_from = None
        while True:
            len_a = len(tokens_a)
            len_b = len(tokens_b)
            total_length = len_a + len_b
            if total_length <= max_num_tokens:
                break
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                trunc_types = token_types_a
                pop_from = "front"
            else:
                trunc_tokens = tokens_b
                trunc_types = token_types_b
                pop_from = "back"

            assert len(trunc_tokens) >= 1

            if pop_from == "front" or (pop_from is None and rng.random() < 0.5):
                trunc_tokens.pop(0)
                trunc_types.pop(0)
            else:
                trunc_tokens.pop()
                trunc_types.pop()
        return (tokens_a, token_types_a), (tokens_b, token_types_b)

    def truncate_seq(self, a, max_seq_len, rng, pop_from=None):
        """
        Truncate sequence pair according to original BERT implementation:
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L391
        """
        if self.use_types:
            tokens_a, token_types_a = a
        else:
            tokens_a = a
        max_num_tokens = max_seq_len - 2
        while True:
            len_a = len(tokens_a)
            if len_a <= max_num_tokens:
                break
            assert len(tokens_a) >= 1
            if pop_from == "front" or (pop_from is None and rng.random() < 0.5):
                tokens_a.pop(0)
                if self.use_types:
                    token_types_a.pop(0)
            else:
                tokens_a.pop()
                if self.use_types:
                    token_types_a.pop()
        output = (tokens_a, token_types_a) if self.use_types else tokens_a
        return output

    def mask_token(self, idx, tokens, types, vocab_words, rng):
        """
        helper function to mask `idx` token from `tokens` according to
        section 3.3.1 of https://arxiv.org/pdf/1810.04805.pdf
        """
        label = tokens[idx]
        if rng.random() < 0.8:
            new_label = self.tokenizer.get_command('MASK').Id
        else:
            if rng.random() < 0.5:
                new_label = label
            else:
                new_label = rng.choice(vocab_words)

        tokens[idx] = new_label

        return label

    def pad_seq(self, seq):
        """helper function to pad sequence pair"""
        #if len(seq) != self.max_seq_len:
        #    print("----->", len(seq))
        num_pad = max(0, self.max_seq_len - len(seq))
        pad_mask = [0] * len(seq) + [1] * num_pad
        seq += [self.tokenizer.get_command('pad').Id] * num_pad
        return seq, pad_mask

    def get_sentence(self, target_seq_length, rng, sentence_num=0, split=False):

        tokens = []
        token_types = []
        split_points = []

        while not tokens:
            doc = None
            while doc is None:
                doc_idx = rng.randint(0, self.ds_len - 1)
                doc = self.sentence_split(self.get_doc(doc_idx))
                if not doc:
                    doc = None

            end_idx = rng.randint(0, len(doc) - 1)
            start_idx = end_idx - 1
            while len(tokens) < target_seq_length:
                if split and len(tokens) > 0:
                    split_points += [len(tokens)]
                if end_idx < len(doc):
                    sentence = doc[end_idx]
                    sentence, sentence_types = self.sentence_tokenize(sentence, sentence_num, end_idx == 0,
                                                                      end_idx == len(doc))
                    tokens = tokens + sentence
                    if self.use_types:
                         token_types = token_types + sentence_types
                    end_idx += 1
                elif start_idx >= 0:
                    sentence = doc[start_idx]
                    sentence, sentence_types = self.sentence_tokenize(sentence, sentence_num, start_idx == 0,
                                                                      start_idx == len(doc))
                    tokens = sentence + tokens
                    if self.use_types:
                        token_types = sentence_types + token_types
                    start_idx -= 1
                else:
                    #print("Full document is too small, returning a small sequence")
                    #print("Length {}, number of sentences {}, start idx {}, end idx {}" .format(len(tokens), len(doc),
                    #                                                                            start_idx, end_idx))
                    break

            if split:
                if len(split_points) == 0:
                    tokens = []
                    token_types = []
                    split_points = []
                    continue
                target_split = int(target_seq_length / 2)
                spi = bisect_right(split_points, target_seq_length / 2)
                if len(split_points) == 1:
                    split_idx = split_points[0]
                elif spi == len(split_points):
                    split_idx = split_points[spi - 2]
                elif abs(split_points[spi - 1] - target_split) < abs(split_points[spi] - target_split):
                    split_idx = split_points[spi - 1]
                else:
                    split_idx =  split_points[min(spi, len(split_points) - 2)]
                tokens = (tokens[:split_idx], tokens[split_idx:])

                str_type_a = 'str' + str(0)
                str_type_b = 'str' + str(1)
                token_types_a = [self.tokenizer.get_type(str_type_a).Id]*len(tokens[0])
                token_types_b = [self.tokenizer.get_type(str_type_b).Id]*len(tokens[1])
                token_types = (token_types_a, token_types_b)
                if len(tokens[0]) == 0 or len(tokens[1]) == 0:
                    tokens = []
                    token_types = []
                    split_points = []
                    continue

        return (tokens, token_types) if self.use_types else tokens

    def create_masked_lm_predictions(self, a, b, mask_lm_prob, max_preds_per_seq, vocab_words, rng, do_not_mask_tokens=[]):
        """
        Mask sequence pair for BERT training according to:
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L338
        """
        tokens_a, token_types_a = a if self.use_types else (a, None)
        len_a = len(tokens_a)
        if self.use_types:
            tokens_b, token_types_b = b
            len_b = len(tokens_b)
            tokens = [self.tokenizer.get_command('ENC').Id] + tokens_a + [self.tokenizer.get_command('sep').Id] + tokens_b + [self.tokenizer.get_command('sep').Id]
            token_types = [token_types_a[0]] + token_types_a + [token_types_a[0]] + token_types_b + [token_types_b[0]]
            cand_indices = [idx + 1 for idx in range(len_a)] + [idx + 2 + len_a for idx in range(len_b)]
            output_types, _ = self.pad_seq(list(token_types))
        else:
            tokens = [self.tokenizer.get_command('ENC').Id] + tokens_a + [self.tokenizer.get_command('sep').Id]
            cand_indices = [idx + 1 for idx in range(len_a)]
            output_types = None

        rng.shuffle(cand_indices)


        output_tokens, pad_mask = self.pad_seq(list(tokens))


        num_to_predict = min(max_preds_per_seq, max(1, int(round(len(tokens) * mask_lm_prob))))

        mask = [0] * len(output_tokens)
        mask_labels = [-1] * len(output_tokens)

        for idx in sorted(cand_indices[:num_to_predict]):
            while idx in do_not_mask_tokens:
                idx = (idx + 1) % len(mask)
            mask[idx] = 1
            label = self.mask_token(idx, output_tokens, output_types, vocab_words, rng)
            mask_labels[idx] = label

        output = (output_tokens, output_types) if self.use_types else output_tokens

        return output, mask, mask_labels, pad_mask

    def corrupt_replace(self, tokens, rng, num_to_corrupt):
        indices = []
        cand_indices = [idx for idx in range(len(tokens))]
        rng.shuffle(cand_indices)

        for idx in sorted(cand_indices[:num_to_corrupt]):
            tokens[idx] = rng.choice(self.vocab_words)
            indices += [idx - 1, idx, idx + 1]

        return tokens, 1, indices

    def corrupt_permute(self, tokens, rng, num_to_corrupt):
        if len(tokens) < 2:
            return tokens, 0, []

        indices = []
        cand_indices = [idx for idx in range(len(tokens))]
        rng.shuffle(cand_indices)

        for idx in sorted(cand_indices[:num_to_corrupt]):
            if idx + 1 >= len(tokens):
                idx -= 1
            tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
            indices += [idx - 1, idx, idx + 1, idx + 2]

        return tokens, 2, indices

    def corrupt_insert(self, tokens, rng, num_to_corrupt):
        indices = []
        cand_indices = [idx for idx in range(len(tokens))]
        rng.shuffle(cand_indices)

        for idx in sorted(cand_indices[:num_to_corrupt]):
            tokens = tokens[:idx] + [rng.choice(self.vocab_words)] + tokens[idx:]
            indices += [idx - 1, idx, idx + 1]

        return tokens, 3, indices

    def corrupt_delete(self, tokens, rng, num_to_corrupt):
        cand_indices = [idx for idx in range(len(tokens))]
        rng.shuffle(cand_indices)

        for idx in sorted(cand_indices[:num_to_corrupt], reverse=True):
            del tokens[idx]

        return tokens, 4, []

class bert_sentencepair_dataset(bert_dataset):
    """
    Dataset containing sentencepairs for BERT training. Each index corresponds to a randomly generated sentence pair.
    Arguments:
        ds (Dataset or array-like): data corpus to use for training
        max_seq_len (int): maximum sequence length to use for a sentence pair
        mask_lm_prob (float): proportion of tokens to mask for masked LM
        max_preds_per_seq (int): Maximum number of masked tokens per sentence pair. Default: math.ceil(max_seq_len*mask_lm_prob/10)*10
        dataset_size (int): number of random sentencepairs in the dataset. Default: len(ds)*(len(ds)-1)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(True, *args, **kwargs)

    def __getitem__(self, idx):
        # get rng state corresponding to index (allows deterministic random pair)
        rng = random.Random(idx)
        # get seq length
        target_seq_length = self.max_seq_len
        # get sentence pair and label
        is_random_next = None
        lena = 0
        lenb = 0
        while (is_random_next is None) or (lena < 1) or (lenb < 1):
            tokensa, tokensb, is_random_next = self.create_random_sentencepair(target_seq_length, rng)
            lena = len(tokensa[0])
            lenb = len(tokensb[0])
        # truncate sentence pair to max_seq_len
        tokensa, tokensb = self.truncate_seq_pair(tokensa, tokensb, self.max_seq_len, rng)
        # join sentence pair, mask, and pad
        tokens, mask, mask_labels, pad_mask = self.create_masked_lm_predictions(tokensa, tokensb, self.mask_lm_prob, self.max_preds_per_seq, self.vocab_words, rng)
        sample = {'text': np.array(tokens[0]), 'types': np.array(tokens[1]), 'sent_label': int(is_random_next), 'mask': np.array(mask), 'mask_labels': np.array(mask_labels), 'pad_mask': np.array(pad_mask)}
        return sample

    def create_random_sentencepair(self, target_seq_length, rng):
        """
        fetches a random sentencepair corresponding to rng state similar to
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L248-L294
        """
        is_random_next = None
        a_length = rng.randint(int(target_seq_length * 0.4), int(target_seq_length * 0.6))
        if rng.random() < 0.5:
            is_random_next = True
            tokens_a, token_types_a = self.get_sentence(a_length, rng, sentence_num=0)
            tokens_b, token_types_b = self.get_sentence(target_seq_length - a_length, rng, sentence_num=1)
        else:
            is_random_next = False
            tokens, token_types = self.get_sentence(target_seq_length, rng, sentence_num=0, split=True)
            tokens_a, tokens_b = tokens
            token_types_a, token_types_b = token_types


        output_a = (tokens_a, token_types_a)
        output_b = (tokens_b, token_types_b)

        return output_a, output_b, is_random_next

### ADDED BY STEPHANE ###
class bert_split_sentences_dataset(bert_dataset):
    """
    Dataset containing sentencepairs for BERT training. Each index corresponds to a randomly generated sentence pair.
    Arguments:
        ds (Dataset or array-like): data corpus to use for training
        max_seq_len (int): maximum sequence length to use for a sentence pair
        mask_lm_prob (float): proportion of tokens to mask for masked LM
        max_preds_per_seq (int): Maximum number of masked tokens per sentence pair. Default: math.ceil(max_seq_len*mask_lm_prob/10)*10
        dataset_size (int): number of random sentencepairs in the dataset. Default: len(ds)*(len(ds)-1)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(False, *args, **kwargs)

    def __getitem__(self, idx):
        # get rng state corresponding to index (allows deterministic random pair)
        rng = random.Random(idx)
        # get seq length
        target_seq_length = self.max_seq_len
        # get sentence pair and label
        is_random_next = None

        while (is_random_next is None) or (len(a) < 1) or (len(b) < 1):
            a, b, is_random_next = self.create_random_sentencepair(target_seq_length, rng)
        # truncate sentences to max_seq_len
        a = self.truncate_seq(a, self.max_seq_len, rng, pop_from="front")
        b = self.truncate_seq(b, self.max_seq_len, rng, pop_from="back")
        # Mask and pad sentence pair
        sample = {}
        sample['sent_label'] = int(is_random_next)
        # A #
        tok_a, mask_a, m_labs_a, pad_mask_a = self.create_masked_lm_predictions(a, None, self.mask_lm_prob, self.max_preds_per_seq, self.vocab_words, rng)
        # B #
        tok_b, mask_b, m_labs_b, pad_mask_b = self.create_masked_lm_predictions(b, None, self.mask_lm_prob, self.max_preds_per_seq, self.vocab_words, rng)
        sample.update({'text': np.array(tok_a), 'mask': np.array(mask_a), 'mask_labels': np.array(m_labs_a),
                       'pad_mask': np.array(pad_mask_a),
                       'text2': np.array(tok_b), 'mask2': np.array(mask_b), 'mask_labels2': np.array(m_labs_b),
                       'pad_mask2': np.array(pad_mask_b)})
        return sample

    def create_random_sentencepair(self, target_seq_length, rng):
        """
        fetches a random sentencepair corresponding to rng state similar to
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L248-L294
        """
        is_random_next = None
        if rng.random() < 0.5:
            is_random_next = True
            tokens_a = self.get_sentence(target_seq_length, rng, sentence_num=0)
            tokens_b = self.get_sentence(target_seq_length, rng, sentence_num=1)
        else:
            is_random_next = False
            tokens = self.get_sentence(target_seq_length * 2.5, rng, sentence_num=0, split=True)
            tokens_a, tokens_b = tokens

        return tokens_a, tokens_b, is_random_next


### ADDED BY STEPHANE ###
class bert_corrupt_sentences_dataset(bert_dataset):
    """
    Dataset containing sentencepairs for BERT training. Each index corresponds to a randomly generated sentence pair.
    Arguments:
        ds (Dataset or array-like): data corpus to use for training
        max_seq_len (int): maximum sequence length to use for a sentence pair
        mask_lm_prob (float): proportion of tokens to mask for masked LM
        max_preds_per_seq (int): Maximum number of masked tokens per sentence pair. Default: math.ceil(max_seq_len*mask_lm_prob/10)*10
        dataset_size (int): number of random sentencepairs in the dataset. Default: len(ds)*(len(ds)-1)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(False, *args, **kwargs)
        self.corrupt_per_sentence = 0.05

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # get rng state corresponding to index (allows deterministic random pair)
        rng = random.Random(idx)
        # get seq length
        target_seq_length = self.max_seq_len
        # get sentence pair and label
        corrupted = None

        while (corrupted is None) or (len(a) < 1):
            a, corrupted, ids = self.create_sentence(target_seq_length, rng)
        # truncate sentences to max_seq_len
        a = self.truncate_seq(a, self.max_seq_len, rng)

        # Mask and pad sentence pair
        tokens, mask, mask_labels, pad_mask = self.create_masked_lm_predictions(a, None, self.mask_lm_prob, self.max_preds_per_seq, self.vocab_words, rng, do_not_mask_tokens=ids)
        sample = {'text': np.array(tokens), 'sent_label': int(corrupted), 'mask': np.array(mask), 'mask_labels': np.array(mask_labels), 'pad_mask': np.array(pad_mask)}

        return sample

    def create_sentence(self, target_seq_length, rng):
        """
        fetches a random sentencepair corresponding to rng state similar to
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L248-L294
        """
        is_random_next = None

        tokens = self.get_sentence(target_seq_length, rng)

        corrupted = 0
        ids = []
        x = rng.random()
        num_to_corrupt = max(2, int(round(len(tokens) * self.corrupt_per_sentence)))
        if x < 0.2:
            tokens, corrupted, ids = self.corrupt_permute(tokens, rng, num_to_corrupt)
        elif x < 0.4:
            tokens, corrupted, ids = self.corrupt_replace(tokens, rng, num_to_corrupt)
        elif x < 0.6:
            tokens, corrupted, ids = self.corrupt_insert(tokens, rng, num_to_corrupt)
        elif x < 0.8:
            tokens, corrupted, ids = self.corrupt_delete(tokens, rng, num_to_corrupt)

        return tokens, corrupted, ids


class bert_rg_sentences_dataset(bert_dataset):
    """
    Dataset containing sentencepairs for BERT training. Each index corresponds to a randomly generated sentence pair.
    Arguments:
        ds (Dataset or array-like): data corpus to use for training
        max_seq_len (int): maximum sequence length to use for a sentence pair
        mask_lm_prob (float): proportion of tokens to mask for masked LM
        max_preds_per_seq (int): Maximum number of masked tokens per sentence pair. Default: math.ceil(max_seq_len*mask_lm_prob/10)*10
        dataset_size (int): number of random sentencepairs in the dataset. Default: len(ds)*(len(ds)-1)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(False, *args, **kwargs)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # get rng state corresponding to index (allows deterministic random pair)
        rng = random.Random(idx)
        # get seq length
        target_seq_length = self.max_seq_len * 2
        # get sentence pair and label
        a, b = self.create_random_sentencepair(target_seq_length, rng)
        while (len(a) < 1) or (len(b) < 1):
            a, b = self.create_random_sentencepair(target_seq_length, rng)
        # truncate sentences to max_seq_len
        a = self.truncate_seq(a, self.max_seq_len, rng, pop_from="front")
        b = self.truncate_seq(b, self.max_seq_len, rng, pop_from="back")
        # Mask and pad sentence pair
        # A #
        tok_a, mask_a, m_labs_a, pad_mask_a = self.create_masked_lm_predictions(a, None, self.mask_lm_prob,
                                                                                self.max_preds_per_seq,
                                                                                self.vocab_words, rng)
        # B #
        tok_b, mask_b, m_labs_b, pad_mask_b = self.create_masked_lm_predictions(b, None, self.mask_lm_prob,
                                                                                self.max_preds_per_seq,
                                                                                self.vocab_words, rng)
        sample = {'text': np.array(tok_a), 'mask': np.array(mask_a), 'mask_labels': np.array(m_labs_a),
                  'pad_mask': np.array(pad_mask_a),
                  'text2': np.array(tok_b), 'mask2': np.array(mask_b), 'mask_labels2': np.array(m_labs_b),
                  'pad_mask2': np.array(pad_mask_b)}
        return sample

    def create_random_sentencepair(self, target_seq_length, rng):
        """
        fetches a random sentencepair corresponding to rng state similar to
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L248-L294
        """
        tokens_a, tokens_b = self.get_sentence(target_seq_length * 2.5, rng, sentence_num=0, split=True)
        return tokens_a, tokens_b

### ADDED BY STEPHANE ###
class bert_combined_sentences_dataset(bert_dataset):
    """
    Dataset containing sentencepairs for BERT training. Each index corresponds to a randomly generated sentence pair.
    Arguments:
        ds (Dataset or array-like): data corpus to use for training
        max_seq_len (int): maximum sequence length to use for a sentence pair
        mask_lm_prob (float): proportion of tokens to mask for masked LM
        max_preds_per_seq (int): Maximum number of masked tokens per sentence pair. Default: math.ceil(max_seq_len*mask_lm_prob/10)*10
        dataset_size (int): number of random sentencepairs in the dataset. Default: len(ds)*(len(ds)-1)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(False, *args, **kwargs)
        self.corrupt_per_sentence = 0.05

    def __getitem__(self, idx):
        # get rng state corresponding to index (allows deterministic random pair)
        rng = random.Random(idx)
        # get seq length
        target_seq_length = self.max_seq_len * 2
        # get sentence pair and label
        (a, b), (c_a, c_b), (ids_a, ids_b) = self.create_random_sentencepair(target_seq_length, rng)
        while (len(a) < 1) or (len(b) < 1):
            (a, b), (c_a, c_b), (ids_a, ids_b) = self.create_random_sentencepair(target_seq_length, rng)
        # truncate sentences to max_seq_len
        a = self.truncate_seq(a, self.max_seq_len, rng, pop_from="front")
        b = self.truncate_seq(b, self.max_seq_len, rng, pop_from="back")

        # Mask and pad sentence pair
        # A #
        tok_a, mask_a, m_labs_a, pad_mask_a = self.create_masked_lm_predictions(a, None, self.mask_lm_prob,
                                                                                self.max_preds_per_seq,
                                                                                self.vocab_words, rng,
                                                                                do_not_mask_tokens=ids_a)
        # B #
        tok_b, mask_b, m_labs_b, pad_mask_b = self.create_masked_lm_predictions(b, None, self.mask_lm_prob,
                                                                                self.max_preds_per_seq,
                                                                                self.vocab_words, rng,
                                                                                do_not_mask_tokens=ids_b)
        sample = {'sent_label': (c_a, c_b),
                  'text': np.array(tok_a), 'mask': np.array(mask_a), 'mask_labels': np.array(m_labs_a),
                  'pad_mask': np.array(pad_mask_a),
                  'text2': np.array(tok_b), 'mask2': np.array(mask_b), 'mask_labels2': np.array(m_labs_b),
                  'pad_mask2': np.array(pad_mask_b)}

        return sample

    def create_random_sentencepair(self, target_seq_length, rng):
        """
        fetches a random sentencepair corresponding to rng state similar to
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L248-L294
        """
        tokens_a, tokens_b = self.get_sentence(target_seq_length * 2.5, rng, sentence_num=0, split=True)

        tokens = [tokens_a, tokens_b]
        corrupted = [0, 0]
        ids = [[], []]

        for i in range(2):
            x = rng.random()
            num_to_corrupt = max(2, int(round(len(tokens) * self.corrupt_per_sentence)))
            if x < 0.2:
                tokens[i], corrupted[i], ids[i] = self.corrupt_permute(tokens[i], rng, num_to_corrupt)
            elif x < 0.4:
                tokens[i], corrupted[i], ids[i] = self.corrupt_replace(tokens[i], rng, num_to_corrupt)
            elif x < 0.6:
                tokens[i], corrupted[i], ids[i] = self.corrupt_insert(tokens[i], rng, num_to_corrupt)
            elif x < 0.8:
                tokens[i], corrupted[i], ids[i] = self.corrupt_delete(tokens[i], rng, num_to_corrupt)

        return tokens, corrupted, ids
