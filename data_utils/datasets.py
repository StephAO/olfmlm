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
from collections import deque
from operator import itemgetter
from bisect import bisect_right
from functools import reduce
import json
import csv
import math
import random
import scipy.stats as ss
import unicodedata

from torch.utils import data
import pandas as pd
import pickle
from math import ceil
import numpy as np
import os

import nltk
nltk.download('punkt')
from nltk import tokenize

from sentence_encoders.data_utils.lazy_loader import lazy_array_loader, exists_lazy, make_lazy
from sentence_encoders.data_utils.tokenization import Tokenization

def clean_tokens(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    clean_tokens = []
    for text in tokens:
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        clean_tokens.append("".join(output))
    return clean_tokens

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
    """
    def __init__(self, ds, max_seq_len=512, mask_lm_prob=.15, max_preds_per_seq=None, short_seq_prob=0.01,
                 presplit_sentences=False, max_dataset_size=None, **kwargs):
        self.avg_len = []
        self.ds = ds
        self.ds_len = min(len(self.ds), max_dataset_size) if max_dataset_size else len(self.ds)
        self.tokenizer = self.ds.GetTokenizer()
        self.vocab_words = list(self.tokenizer.text_token_vocab.values())
        self.ds.SetTokenizer(None)
        self.max_seq_len = max_seq_len
        self.mask_lm_prob = mask_lm_prob
        if max_preds_per_seq is None:
            max_preds_per_seq = math.ceil(max_seq_len*mask_lm_prob / 10)*10
        self.max_preds_per_seq = max_preds_per_seq
        self.presplit_sentences = presplit_sentences
        self.corrupt_per_sentence = 0.10
        self.epoch = 0
        self.num_tokens_seen = 0
        self.idf_path = os.path.dirname(__file__) + "/../idf.p"
        with open(self.idf_path, "rb") as f:
            self.idfs = pickle.load(f)
        self._all_tf = []
        self.task_list = ["mlm", "nsp", "psp", "sd", "so", "rg", "fs", "tc", "sc", "sbo", "wlen", "cap", "tf", "tf_idf", "tgs"]
        self.task_dict = dict(zip(self.task_list, range(1, len(self.task_list) + 1)))
	
    def __len__(self):
        return self.ds_len

    def set_args(self, modes):
        print("setting up args, modes:", modes)
        self.modes = modes
        self.split_percent = 1.0
        self.corruption_rate = 0.
        self.num_sent_per_seq = 1
        self.num_seq_returned = 1
        self.trigram_shuffle_rate = 0
        # Assert that at most 1 sentence distance loss exists
        self.sentence_tasks = ["nsp", "psp", "sc", "sd", "so"]
        assert [x in self.sentence_tasks for x in self.modes].count(True) <= 1
        # Masked Language Data (Default)
        self.mask_lm_prob = 0.15
        self.task_id = self.task_dict[self.modes[-1]]
        # Semantic Similarity
        if "nsp" in self.modes: # Next Sentence Prediction Data
            self.split_percent = 0.5
            self.num_sent_per_seq = 2
        if "psp" in self.modes:  # Prev Sentence Prediction Data
            self.split_percent = 2. / 3.
            self.num_sent_per_seq = 2
        if "sd" in self.modes: # Sentence Distance Data
            self.split_percent = 1. / 3.
            self.num_sent_per_seq = 2
        if "so" in self.modes: # Sentence Re-ordering Data
            self.split_percent = 1.0
            self.num_sent_per_seq = 2
        if "rg" in self.modes or "fs" in self.modes: # Referential Game Data
            self.num_seq_returned = 2
        if "sc" in self.modes or "tc" in self.modes: # Sequence Consistency
            self.corruption_rate = 0.50
        if "tgs" in self.modes:
            self.trigram_shuffle_rate = 0.05


    def __getitem__(self, idx):
        # get rng state corresponding to index (allows deterministic random pair)
        if idx >= self.ds_len:
            raise StopIteration
        rng = random.Random(idx)
        self.idx = idx
        # get sentence pair and label
        sentence_labels = None
        tokens, token_labels = [], {}
        
        while (sentence_labels is None) or any([len(x) < 1 for x in tokens]):
            tokens, sentence_labels, token_labels, corrupted_ids = self.create_random_sentencepair(rng)

        # join sentence pair, mask, and pad
        tokens, token_types, token_labels, mask, tgs_mask, mask_labels, pad_mask, num_tokens = \
            self.create_masked_lm_predictions(tokens, token_labels, self.mask_lm_prob, self.max_preds_per_seq,
                                              self.vocab_words, rng, do_not_mask_tokens=corrupted_ids)

        
        aux_labels = {k: np.array(list(map(int, v))) if not isinstance(v, np.ndarray) else v 
                      for k, v in sentence_labels.items() if k in self.modes}
        aux_labels.update({k: np.array(v) for k, v in token_labels.items() if k in self.modes})
        sample = {'aux_labels': aux_labels, 'n': len(tokens),'num_tokens': num_tokens}

        for i in range(len(tokens)):
            sample.update({'text_' + str(i): np.array(tokens[i]), 'types_' + str(i): np.array(token_types[i]),
                           'task_' + str(i): np.full_like(tokens[i], self.task_id),
                           'mask_' + str(i): np.array(mask[i]), 'mask_labels_' + str(i): np.array(mask_labels[i]),
                           'tgs_mask_' + str(i): np.array(tgs_mask[i]), 'pad_mask_' + str(i): np.array(pad_mask[i])})

        return sample

    def swap_order(self, tokens, token_labels, idx=0):
        # Swap tokens and token labels at index with those at index + 1
        tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
        for k in token_labels:
            token_labels[k][idx], token_labels[k][idx + 1] = token_labels[k][idx + 1], token_labels[k][idx]
        return tokens, token_labels

    def create_random_sentencepair(self, rng):
        """
        fetches a random sentencepair based on requirements of modes
        """
        sentence_labels = {k: [] for k in  ["nsp", "psp", "sd", "so", "sc"]}
        token_labels = {k: [] for k in self.modes if k in ["cap", "wlen", "tf", "tf_idf"]}
        # either split or corrupt not both
        split = rng.random()
        # Single sequence
        if self.num_sent_per_seq * self.num_seq_returned == 1:
            tokens, token_labels = self.get_sentence(self.max_seq_len, 1, rng)

        # Contiguous multiple sequences
        elif split <= self.split_percent:
            target_seq_len = self.max_seq_len * 2 if "rg" in self.modes or "fs" in self.modes else self.max_seq_len
            tokens, token_labels = self.get_sentence(target_seq_len, self.num_sent_per_seq * self.num_seq_returned, rng)
            
            for i in range(self.num_seq_returned):
                sentence_labels["nsp"].append(True)
                sentence_labels["sd"].append(0)
                if "psp" in self.modes or "so" in self.modes:
                    if rng.random() < 0.5:
                        # Next sentence
                        sentence_labels["psp"].append(0)
                        sentence_labels["so"].append(0)
                    else:
                        # Previous sentence
                        # Swap sequences
                        sentence_labels["psp"].append(1)
                        sentence_labels["so"].append(1)
                        tokens, token_labels = self.swap_order(tokens, token_labels, idx=i*2)

        # Same document, non-contiguous multiple sequences
        elif self.split_percent * 2 < 1 and split <= self.split_percent * 2:
            tokens, token_labels = self.get_sentence(self.max_seq_len * 1.5, self.num_sent_per_seq,
                                                                  rng, non_contiguous=True)
            for i in range(self.num_seq_returned):
                sentence_labels["sd"].append(1)
                if "psp" in self.modes:
                    if rng.random() < 0.5:
                        sentence_labels["psp"].append(3) # Same document after
                    else:
                        sentence_labels["psp"].append(4) # Same document before
                        # Swap sequences
                        tokens, token_labels = self.swap_order(tokens, token_labels, idx=i*2)
        # Multiple sequences from different documents
        else:
            tokens = []
            for i in range(self.num_seq_returned):
                for j in range(self.num_sent_per_seq):
                    tok, tok_labels = self.get_sentence(self.max_seq_len // self.num_sent_per_seq, 1, rng, diff_doc=True)
                    tokens += tok
                    [token_labels[k].extend(v) for k, v in tok_labels.items()]

                sentence_labels["nsp"].append(False)
                sentence_labels["sd"].append(2)
                sentence_labels["psp"].append(2)

        # Apply corruption
        corrupted = bool(rng.random() < self.corruption_rate)
        ids = []
        token_labels["tc"] = []
        if corrupted:
            sentence_labels["sc"] += [True] * self.num_seq_returned
            for i in range(len(tokens)):
                cor_ids, cor_lab = self.corrupt_seq(tokens[i], rng)
                ids.append(cor_ids)
                token_labels["tc"].append(np.zeros_like(tokens[i]))
            for i in range(len(tokens)):
                token_labels["tc"][i][ids[i]] = 1. #float(cor_lab)
                token_labels["tc"][i] = token_labels["tc"][i].tolist()
        else:
            sentence_labels["sc"] += [False] * self.num_seq_returned
            for i in range(len(tokens)):
                token_labels["tc"].append(np.zeros_like(tokens[i]).tolist())

        return tokens, sentence_labels, token_labels, ids

    def sentence_tokenize(self, sent):
        """tokenize sentence and get token types if tokens=True"""
        return self.tokenizer.EncodeAsIds(sent).tokenization

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

    def truncate_sequence(self, tokens, token_types, token_labels, tl_idx, rng):
        """
        Truncate sequence pair
        """
        max_num_tokens = self.max_seq_len - 2
        while True:
            if len(tokens) <= max_num_tokens:
                break
            idx = 0 if rng.random() < 0.5 else len(tokens) - 1
            tokens.pop(idx)
            token_types.pop(idx)
            for k, v in token_labels.items():
                v[tl_idx].pop(idx)

    def pad_seq(self, tokens,  token_types, token_labels, tl_idx):
        """helper function to pad sequence pair"""
        self.avg_len.append(len(tokens))
        num_pad = max(0, self.max_seq_len - len(tokens))
        pad_mask = [0] * len(tokens) + [1] * num_pad
        tokens += [self.tokenizer.get_command('pad').Id] * num_pad
        token_types += [token_types[-1]] * num_pad
        for k in token_labels:
            token_labels[k][tl_idx] += [0.] * num_pad
        return pad_mask

    def mask_token(self, idx, tokens, types, vocab_words, rng):
        """
        helper function to mask `idx` token from `tokens` according to
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

    def get_caps(self, s, t):
        # Returns a list of labels with 1 for each capitablized token in the sentence, 0 else
        si = 0
        ti = 0
        tii = 0
        caps = [0] * len(t)
        while si < len(s) and ti < len(t):
            if t[ti][tii] == s[si].lower():
                if s[si].isupper():
                    caps[ti] = 1
                si += 1
                tii += 1
            elif t[ti][tii] == "#" or ord(t[ti][tii]) >= 128:
                tii += 1
            elif s[si] in [" ", "#"] or ord(s[si]) >= 128:
                si += 1
            elif t[ti] == "[UNK]":
                tii = 0
                ti += 1
                if ti >= len(t):
                    break
            else:
                while s[si].lower() != t[ti][tii]:
                    si += 1
                    if si >= len(s):
                        print(s, si)
                        print(t, ti, tii)
                        break
                        
            if tii == len(t[ti]):
                tii = 0
                ti += 1
        return caps

    def run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def count_doc_tfs(self, doc):
        # Count term frequency for given document
        doc_tokens = self.tokenizer.text_tokenizer.tokenize(" ".join(doc))
        self.doc_tf = {}
        for tok in doc_tokens:
            self.doc_tf[tok] = self.doc_tf.get(tok, 0) + 1

    def scale(self, vector):
        # Scale a list of floats (vector) to floats between 0 and 10
        if len(vector) == 0:
            return vector
        scale = 10
        scaling = lambda x: ((scale) * (x - _min) / (_max - _min + 1e-8))
        _min = min(vector)
        _max = max(vector)
        return [scaling(w) for w in vector]

    def get_word_labels(self, sent, doc):
        labels = {}
        tokens = self.tokenizer.text_tokenizer.tokenize(sent)
        ids = self.tokenizer.text_tokenizer.convert_tokens_to_ids(tokens) 
        sent = self.run_strip_accents(sent)
        if "cap" in self.modes:
            labels["cap"] = self.get_caps(sent, tokens)
        if "wlen" in self.modes:
            labels["wlen"] = [len(w.strip("#")) for w in tokens]
        if "tf" in self.modes or "tf_idf" in self.modes:
            tf = [self.doc_tf[tok] for tok in tokens]
            idf = [self.idfs[tok] for tok in ids]
            tf_idf = [tf[i] * idf[i] for i in range(len(tokens))]
            if "tf" in self.modes:
                labels["tf"] = self.scale(tf)
            elif "tf_idf" in self.modes:
                labels["tf_idf"] = self.scale(tf_idf)

        return labels

    def get_discrete_normal_distributions(self, x, n):
        """Returns distribution over the range of x with 'n' normal distributions equally spread across the range"""
        if x == 1:
            return np.ones(1)
        num_normal_dist = n - 1
        dist = np.zeros(x)
        for d in range(num_normal_dist):
            mu, sigma = float(d + 1) * (x - 1) / n, (x - 1) / (3. * n)
            dist += ss.norm.pdf(list(range(x)), mu, sigma)
        return dist / np.sum(dist)


    def get_sentence(self, target_seq_length, num_sents, rng, non_contiguous=False, diff_doc=False):
        """
        Returns a list of sentences (List[List[tokens]) and any token level labels (List[Dict[str, List[Any]])
        :param target_seq_length: Ideal length for the full string of sentences (counted by number of tokens)
        :param num_sents: Mininum number of sentences required
        :param rng: RNG
        :param non_contiguous: Whether the sentences should be contiguous or non-contiguous from the idexed document
        :param diff_doc: Whether it should use the indexed document or a different random document
        :return:
        """
        num_sent_required = num_sents + 1 if non_contiguous else num_sents
        sentences = deque()
        sent_token_labels = deque()

        idx = self.idx
        while diff_doc and idx == self.idx:
            idx = rng.randint(0, self.ds_len - 1)
        doc = self.sentence_split(self.get_doc(idx))

        # If corpora is filtered properly, this should never occur
        while len(doc) < num_sents:
            print(idx, doc, "YIKES")
            doc = self.sentence_split(self.get_doc(rng.randint(0, self.ds_len - 1)))

        self.count_doc_tfs(doc)

        end_idx = rng.randint(0, len(doc) - 1)
        start_idx = end_idx - 1
        total_length = 0
        while total_length < target_seq_length or len(sentences) < num_sent_required:
            # Add next sentence to sequence
            if end_idx < len(doc):
                sentence = doc[end_idx]
                tl = self.get_word_labels(sentence, doc)
                sentence = self.sentence_tokenize(sentence)
                sentences.append(sentence)
                sent_token_labels.append(tl)
                end_idx += 1
            # Add previous sentence to sequence
            elif start_idx >= 0:
                sentence = doc[start_idx]
                tl = self.get_word_labels(sentence, doc)
                sentence = self.sentence_tokenize(sentence)
                sentences.insert(0, sentence)
                sent_token_labels.insert(0, tl)
                start_idx -= 1
            # No more sentences to add
            else:
                break

            if len(sentence) == 0:
                print(doc)
            total_length += len(sentence)
        
        assert len(sentences) >= num_sent_required

        # Get split indices (i.e. subdivide the set of sentences)
        num_sent = len(sentences)
        split_idxs = [0, num_sent]
        if num_sent_required > 1:
            split_dist = self.get_discrete_normal_distributions(num_sent - 1, num_sent_required)
            split_idxs = np.random.choice(list(range(1, num_sent)), num_sent_required - 1, replace=False, p=split_dist)
            split_idxs = [0] + list(split_idxs) + [num_sent]
            split_idxs = sorted(split_idxs)

        # Combine grouped sentences
        tokens = []
        token_labels = {k: [] for k in self.modes if k in ["cap", "wlen", "tf", "tf_idf"]}
        for i in range(len(split_idxs) - 1):
            if non_contiguous and i == 1:
                continue
            seq_tokens, seq_tl = [], {k: [] for k in self.modes if k in ["cap", "wlen", "tf", "tf_idf"]}

            for sent_idx in range(split_idxs[i], split_idxs[i+1]):
                seq_tokens += sentences[sent_idx]
                for k in seq_tl:
                    seq_tl[k] += sent_token_labels[sent_idx][k]

            tokens.append(seq_tokens)
            for k in seq_tl:
                token_labels[k].append(seq_tl[k])

        return tokens, token_labels

    def concat_sentences(self, tokens, token_types, token_labels):
        """Concatenate tokens and associated lists. E.g. [[1,2,3],[4,5,6]] -> [[1,2,3,4,5,6]]"""
        assert len(tokens) >= 2
        tokens = [reduce(lambda a, b: a + [self.tokenizer.get_command('sep').Id] + b, tokens)]
        token_types = [reduce(lambda a, b: a + [a[0]] + b, token_types)]
        token_labels = {k: [reduce(lambda a, b: a + [0.] + b, v)] for k, v in token_labels.items()}
        return tokens, token_types, token_labels

    def shuffle_trigrams(self, tokens, token_types, token_labels, i, rng):
        """
        Shuffle trigrams in spot with a rate of self.trigram_shuffle_rate.
        Returns mask (which tokens to look for loss). Labels (which shuffle permutation used) is added to token labels.
        Labels and masks are defined on the 3rd token of the sequence:
        e.g. starting with [1,2,3,4,5] with one shuffle to create [1,2,5,4,3]
             would have labels: [0,0,0,0,5] and mask [0,0,0,0,1]
        """
        if self.trigram_shuffle_rate == 0:
            return []

        ngram = 3
        # 6 permutations (ngram = 3)
        classes = {0: [2, 1, 0], 1: [0, 2, 1], 2: [1, 0, 2], 3: [1, 2, 0], 4: [2, 0, 1], 5: [0, 1, 2]}
        # 2 permutations (ngram = 2)
        #classes = {0: [1, 0], 1: [0, 1]}
        labels = []
        mask = []
        idx = 0
        valid_seq_len = 0
        while idx < len(tokens[i]):
            if tokens[i][idx] in [self.tokenizer.get_command('ENC').Id, self.tokenizer.get_command('sep').Id,
                                  self.tokenizer.get_command('pad').Id, self.tokenizer.get_command('MASK').Id]:
                valid_seq_len = 0
            else:
                valid_seq_len += 1

            if valid_seq_len >= ngram and rng.random() < self.trigram_shuffle_rate:
                valid_seq_len = 0
                # Shuffle
                label = rng.randint(0,5)
                perm = classes[label]
                tokens[i][idx - (ngram - 1) : idx + 1] = [tokens[i][idx - p] for p in perm]
                token_types[i][idx - (ngram - 1) : idx + 1] = [token_types[i][idx - p] for p in perm]
                for k in token_labels:
                    token_labels[k][i][idx - (ngram - 1): idx + 1] = [token_labels[k][i][idx - p] for p in perm]
                labels.append(label)
                mask.append(1)
            else:
                labels.append(0)
                mask.append(0)

            idx += 1

        assert len(labels) == len(mask) == len(tokens[i])
        if "tgs" not in token_labels:
            token_labels["tgs"] = [None] * len(tokens)
        token_labels["tgs"][i] = labels

        return mask

    def create_masked_lm_predictions(self, tokens, token_labels, mask_lm_prob, max_preds_per_seq,
                                     vocab_words, rng, do_not_mask_tokens=[]):
        """
        Mask sequence pair for BERT training. Includes necessary concatenation, padding, and trigram shuffling.
        """
        token_types = [[self.tokenizer.get_type('str' + str(i % 2)).Id] * len(tokens[i]) for i in range(len(tokens))]
        # Concatenate sequences if requested
        if len(tokens) == 4:
            half = int(len(tokens) / 2)
            t1, tt1, tl1 = self.concat_sentences(tokens[:half], token_types[:half], {k: v[:half] for k, v in token_labels.items()})
            t2, tt2, tl2 = self.concat_sentences(tokens[half:], token_types[half:], {k: v[half:] for k, v in token_labels.items()})
            tokens, token_types, token_labels = t1 + t2, tt1 + tt2, {k: tl1[k] + tl2[k] for k in token_labels}
        elif len(tokens) == 2 and self.num_seq_returned == 1:
            tokens, token_types, token_labels = self.concat_sentences(tokens, token_types, token_labels)

        mask = [[] for _ in range(len(tokens))]
        tgs_mask = [[] for _ in range(len(tokens))]
        mask_labels = [[] for _ in range(len(tokens))]
        pad_mask = [[] for _ in range(len(tokens))]
        num_tokens = 0
        for i in range(len(tokens)):
            # Truncate sequence if too long
            self.truncate_sequence(tokens[i], token_types[i], token_labels, i, rng)
            # Add start and end tokens ('CLS' and 'SEP' respectively)
            tokens[i] = [self.tokenizer.get_command('ENC').Id] + tokens[i] + [self.tokenizer.get_command('sep').Id]
            token_types[i] = [token_types[i][0]] + token_types[i] + [token_types[i][0]]
            for k in token_labels.keys():
                token_labels[k][i] = [0.] + token_labels[k][i] + [0.]
            cand_indices = list(range(len(tokens[i])))
            num_to_predict = min(max_preds_per_seq, max(1, int(round(len(tokens[i]) * mask_lm_prob))))
            rng.shuffle(cand_indices)
            num_tokens += len(tokens[i])
            # Pad sequence if too short
            pad_mask[i] = self.pad_seq(tokens[i], token_types[i], token_labels, i)
            # Mask tokens
            mask[i] = [0] * len(tokens[i])
            mask_labels[i] = [-1] * len(tokens[i])
            num_masked, ci = 0, 0
            while num_masked < num_to_predict:
                idx = cand_indices[ci]
                ci += 1
                if idx in do_not_mask_tokens or tokens[i][idx] in [self.tokenizer.get_command('ENC').Id,
                                                                   self.tokenizer.get_command('sep').Id,
                                                                   self.tokenizer.get_command('pad').Id]:
                    continue
                mask[i][idx] = 1
                label = self.mask_token(idx, tokens[i], token_types[i], vocab_words, rng)
                mask_labels[i][idx] = label
                num_masked += 1

            # StructBERT Tri-grams
            tgs_mask[i] = self.shuffle_trigrams(tokens, token_types, token_labels, i, rng)

        return tokens, token_types, token_labels, mask, tgs_mask, mask_labels, pad_mask, num_tokens

    def corrupt_replace(self, tokens, rng, num_to_corrupt):
        indices = []
        cand_indices = [idx for idx in range(len(tokens))]
        rng.shuffle(cand_indices)

        for idx in sorted(cand_indices[:num_to_corrupt]):
            tokens[idx] = rng.choice(self.vocab_words)
            indices += [idx]

        return indices

    def corrupt_permute(self, tokens, rng, num_to_corrupt):
        if len(tokens) < 2:
            return []

        indices = []
        cand_indices = [idx for idx in range(len(tokens))]
        rng.shuffle(cand_indices)

        for idx in sorted(cand_indices[:num_to_corrupt]):
            if idx + 1 >= len(tokens):
                continue
            tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
            indices += [idx, idx + 1]

        return indices

    def corrupt_insert(self, tokens, rng, num_to_corrupt):
        indices = []
        cand_indices = [idx for idx in range(len(tokens))]
        rng.shuffle(cand_indices)

        for idx in sorted(cand_indices[:num_to_corrupt]):
            tokens.insert(idx, rng.choice(self.vocab_words))
            indices += [idx]
            # indices += [idx - 1, idx, idx + 1]

        return indices

    def corrupt_delete(self, tokens, rng, num_to_corrupt):
        cand_indices = [idx for idx in range(len(tokens))]
        rng.shuffle(cand_indices)
        indices = []

        for i, idx in enumerate(sorted(cand_indices[:num_to_corrupt], reverse=True)):
            del tokens[idx]
            # adjust_idx = idx - (num_to_corrupt - 1 - i)
            # indices += [adjust_idx - 1, adjust_idx]

        return indices

    def corrupt_seq(self, tokens, rng):
        x = rng.random()
        num_to_corrupt = max(2, int(round(len(tokens) * self.corrupt_per_sentence)))
        if x < (1. / 3.):
            ids = self.corrupt_permute(tokens, rng, num_to_corrupt)
            label = 1.
        elif x < (2. / 3.):
            ids = self.corrupt_replace(tokens, rng, num_to_corrupt)
            label = 2.
        else:
            ids = self.corrupt_insert(tokens, rng, num_to_corrupt)
            label = 3.
        return ids, label

