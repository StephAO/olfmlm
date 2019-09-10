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

from sentence_encoders.data_utils.lazy_loader import lazy_array_loader, exists_lazy, make_lazy
from sentence_encoders.data_utils.tokenization import Tokenization

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
    def __init__(self, ds, max_seq_len=512, mask_lm_prob=.15, max_preds_per_seq=None, short_seq_prob=0.01, dataset_size=None, presplit_sentences=False, **kwargs):
        self.avg_len = []
        self.ds = ds
        self.ds_len = len(self.ds)
        self.tokenizer = self.ds.GetTokenizer()
        self.vocab_words = list(self.tokenizer.text_token_vocab.values())
        self.ds.SetTokenizer(None)
        self.max_seq_len = max_seq_len
        self.mask_lm_prob = mask_lm_prob
        if max_preds_per_seq is None:
            max_preds_per_seq = math.ceil(max_seq_len*mask_lm_prob / 10)*10
        self.max_preds_per_seq = max_preds_per_seq
        self.dataset_size = dataset_size
        if self.dataset_size is None:
            self.dataset_size = self.ds_len * (self.ds_len-1)
        self.presplit_sentences = presplit_sentences
        self.corrupt_per_sentence = 0.05
        self.target_seq_length = self.max_seq_len
        self.epoch = 0

    def __len__(self):
        return self.dataset_size

    def set_args(self, modes, epoch, num_iters):
        print("setting up args, modes:", modes)
        self.modes = modes
        self.epoch = epoch
        self.num_iters = num_iters
        self.split_percent = 0.0
        self.corruption_rate = 0.0
        self.num_sent_per_seq = 1
        self.target_seq_length = self.max_seq_len
        self.concat = False
        self.mask_lm_prob = 0.0
        if "mlm" in self.modes:
            self.mask_lm_prob = 0.15
            self.task_id = 0
            self.concat = True
        if "nsp" in self.modes:
            self.split_percent = 0.5
            self.num_sent_per_seq = 2
            self.target_seq_length = int(self.max_seq_len / 2)
            self.concat = True
            self.task_id = 1
        if "rg" in self.modes:
            self.num_sent_per_seq = 2
            self.task_id = 2
        if "corrupt" in self.modes:
            self.corruption_rate = 0.05
            self.task_id = 3

    def __getitem__(self, idx):
        # get rng state corresponding to index (allows deterministic random pair)
        rng = random.Random(idx + (self.epoch - 1) * self.num_iters)
        # get sentence pair and label
        sentence_label = None
        tokens = []
        while (sentence_label is None) or any([len(x) < 1 for x in tokens]):
            tokens, token_types, sentence_label, do_not_mask_ids = self.create_random_sentencepair(rng)
        # truncate sentence pair to max_seq_len
        self.truncate_sequences(tokens, token_types, self.max_seq_len, rng)
        # join sentence pair, mask, and pad
        output, mask, mask_labels, pad_mask = self.create_masked_lm_predictions(tokens, token_types, self.mask_lm_prob,
                                                                                self.max_preds_per_seq,
                                                                                self.vocab_words, rng, self.concat,
                                                                                do_not_mask_ids)
        sample = {'sent_label': int(sentence_label), 'n': len(output)}
        for i in range(len(output)):
            tokens, token_types = output[i]
            sample.update({'text_' + str(i): np.array(tokens), 'types_' + str(i): np.array(token_types),
                           'task_' + str(i): np.full_like(tokens, self.task_id),
                           'mask_' + str(i): np.array(mask[i]), 'mask_labels_' + str(i): np.array(mask_labels[i]),
                           'pad_mask_' + str(i): np.array(pad_mask[i])})
        return sample

    def create_random_sentencepair(self, rng):
        """
        fetches a random sentencepair corresponding to rng state similar to
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L248-L294
        """
        sentence_label = False
        # either split or corrupt not both
        assert self.split_percent * self.corruption_rate == 0
        split = bool(rng.random() < self.split_percent)
        if split: # Split sequence
            sentence_label = True
            tokens = []
            token_types = []
            for i in range(self.num_sent_per_seq):
                tok, tok_types = self.get_sentence(self.target_seq_length, rng, sentence_num=i)
                tokens.append(tok)
                token_types.append(tok_types)
        else: # Contiguous sequence
            tokens, token_types = self.get_sentence(self.target_seq_length * 2.5, rng, sentence_num=0,
                                                    split=self.target_seq_length)

        corrupted = bool(rng.random() < self.corruption_rate)
        ids = []
        if corrupted:
            sentence_label = True
            for i in range(len(tokens)):
                ids.append(self.corrupt_seq(tokens[i], token_types[i], rng))

        return tokens, token_types, sentence_label, ids

    def sentence_tokenize(self, sent, sentence_num=0, beginning=False, ending=False):
        """tokenize sentence and get token types if tokens=True"""
        tokens = self.tokenizer.EncodeAsIds(sent).tokenization
        token_types = None
        str_type = 'str' + str(sentence_num)
        token_types = [self.tokenizer.get_type(str_type).Id]*len(tokens)
        return tokens, token_types

    def sentence_split(self, document, min_length):
        """split document into sentences"""
        if len(self.tokenizer.EncodeAsIds(document).tokenization) < min_length:
            return None
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

    def truncate_sequences(self, tokens, token_types, max_seq_len, rng):
        """
        Truncate sequence pair according to original BERT implementation:
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L391
        """
        max_num_tokens = max_seq_len - 1 - len(tokens)
        while True:
            lengths = [len(s) for s in tokens]
            total_length = sum(lengths)
            if total_length <= max_num_tokens:
                break
            trunc_idx = np.argmax(lengths)

            assert len(tokens[trunc_idx]) >= 1
            if rng.random() < 0.5:
                tokens[trunc_idx].pop(0)
                token_types[trunc_idx].pop(0)
            else:
                tokens[trunc_idx].pop()
                token_types[trunc_idx].pop()

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
        self.avg_len.append(len(seq))
        if len(self.avg_len) == 100000:
            print("Tokens (min, mean, max):", np.min(self.avg_len), np.mean(self.avg_len), np.max(self.avg_len))
        num_pad = max(0, self.max_seq_len - len(seq))
        pad_mask = [0] * len(seq) + [1] * num_pad
        seq += [self.tokenizer.get_command('pad').Id] * num_pad
        return seq, pad_mask

    def get_sentence(self, target_seq_length, rng, sentence_num=0, split=False):

        tokens = []
        token_types = []
        split_points = []

        doc = None
        doc_idx = rng.randint(0, self.ds_len - 1)

        while not tokens:
            while doc is None:
                doc = self.sentence_split(self.get_doc(doc_idx), target_seq_length)
                if not doc:
                    doc_idx = (doc_idx + 1) % self.ds_len

            end_idx = rng.randint(0, len(doc) - 1)
            start_idx = end_idx - 1
            while len(tokens) < target_seq_length:
                if split and len(tokens) > 0:
                    split_points += [len(tokens)]
                if end_idx < len(doc):
                    sentence = doc[end_idx]
                    sentence, sentence_types = self.sentence_tokenize(sentence, sentence_num, end_idx == 0,
                                                                      end_idx == len(doc))
                    if len(sentence) + len(tokens) > target_seq_length:
                        break
                    tokens = tokens + sentence
                    token_types = token_types + sentence_types
                    end_idx += 1
                elif start_idx >= 0:
                    sentence = doc[start_idx]
                    sentence, sentence_types = self.sentence_tokenize(sentence, sentence_num, start_idx == 0,
                                                                      start_idx == len(doc))
                    if len(sentence) + len(tokens) > target_seq_length:
                        break
                    tokens = sentence + tokens
                    token_types = sentence_types + token_types
                    start_idx -= 1
                else:
                    break

            if split:
                first_split = -1
                second_split = -1
                for i in range(len(split_points)):
                    if i + 1 >= len(split_points):
                        break
                    if first_split == -1 and split_points[i + 1] > split:
                        first_split = split_points[i]
                    elif first_split != -1 and split_points[i + 1] - first_split > split:
                        second_split = split_points[i]
                        break

                # Not able to split document in a way that works for training, try a different doc
                if -1 in [first_split, second_split] or \
                                        len(tokens[:first_split]) * len(tokens[first_split:second_split]) == 0:
                    tokens = []
                    token_types = []
                    split_points = []
                    doc = None
                    doc_idx = (doc_idx + 1) % self.ds_len
                    continue

                tokens = [tokens[:first_split], tokens[first_split:second_split]]
                token_types_a = [self.tokenizer.get_type('str' + str(0)).Id]*len(tokens[0])
                token_types_b = [self.tokenizer.get_type('str' + str(1)).Id]*len(tokens[1])
                token_types = [token_types_a, token_types_b]

        return tokens, token_types

    def create_masked_lm_predictions(self, tokens, token_types, mask_lm_prob, max_preds_per_seq, vocab_words, rng,
                                     concat=False, do_not_mask_tokens=[]):
        """
        Mask sequence pair for BERT training according to:
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L338
        """
        cand_indices = []
        if concat:
            masked_tokens = [self.tokenizer.get_command('ENC').Id]
            masked_tt = [token_types[0][0]]
            cum_len = 1
            cd = []
            for i in range(len(tokens)):
                masked_tokens += tokens[i] + [self.tokenizer.get_command('sep').Id]
                masked_tt += token_types[i] + [token_types[i][0]]
                cd += [idx + cum_len for idx in range(len(tokens[i]))]
                cum_len += len(tokens[i]) + 1
            masked_tt, _ = self.pad_seq(masked_tt)
            tokens = [masked_tokens]
            token_types = [masked_tt]
            cand_indices = [cd]
        else:
            for i in range(len(tokens)):
                tokens[i] = [self.tokenizer.get_command('ENC').Id] + tokens[i] + [self.tokenizer.get_command('sep').Id]
                token_types[i] = [token_types[i][0]] + token_types[i] + [token_types[i][0]]
                token_types[i], _ = self.pad_seq(token_types[i])
                cand_indices += [[idx + 1 for idx in range(len(tokens[i]))]]
        output = [[] for _ in range(len(tokens))]
        output_tokens = [[] for _ in range(len(tokens))]
        mask = [[] for _ in range(len(tokens))]
        mask_labels = [[] for _ in range(len(tokens))]
        pad_mask = [[] for _ in range(len(tokens))]
        for i in range(len(tokens)):
            rng.shuffle(cand_indices[i])

            output_tokens[i], pad_mask[i] = self.pad_seq(list(tokens[i]))
            num_to_predict = min(max_preds_per_seq, max(1, int(round(len(tokens[i]) * mask_lm_prob))))

            mask[i] = [0] * len(output_tokens[i])
            mask_labels[i] = [-1] * len(output_tokens[i])

            for idx in sorted(cand_indices[i][:num_to_predict]):
                while idx in do_not_mask_tokens:
                    idx = (idx + 1) % len(mask[i])
                mask[i][idx] = 1
                label = self.mask_token(idx, output_tokens[i], token_types[i], vocab_words, rng)
                mask_labels[i][idx] = label

            output[i] = (output_tokens[i], token_types[i])

        return output, mask, mask_labels, pad_mask

    def corrupt_replace(self, tokens, token_types, rng, num_to_corrupt):
        indices = []
        cand_indices = [idx for idx in range(len(tokens))]
        rng.shuffle(cand_indices)

        for idx in sorted(cand_indices[:num_to_corrupt]):
            tokens[idx] = rng.choice(self.vocab_words)
            indices += [idx]
            # indices += [idx - 1, idx, idx + 1]

        return indices

    def corrupt_permute(self, tokens, token_types, rng, num_to_corrupt):
        if len(tokens) < 2:
            return []

        indices = []
        cand_indices = [idx for idx in range(len(tokens))]
        rng.shuffle(cand_indices)

        for idx in sorted(cand_indices[:num_to_corrupt]):
            if idx + 1 >= len(tokens):
                continue
            tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
            token_types[idx], token_types[idx + 1] = token_types[idx + 1], token_types[idx]
            indices += [idx, idx + 1]
            # indices += [idx - 1, idx, idx + 1, idx + 2]

        return indices

    def corrupt_insert(self, tokens, token_types, rng, num_to_corrupt):
        indices = []
        cand_indices = [idx for idx in range(len(tokens))]
        rng.shuffle(cand_indices)

        for idx in sorted(cand_indices[:num_to_corrupt]):
            tokens.inser(idx, rng.choice(self.vocab_words))
            token_types.insert(idx, token_types[idx])
            indices += [idx]
            # indices += [idx - 1, idx, idx + 1]

        return indices

    def corrupt_delete(self, tokens, token_types, rng, num_to_corrupt):
        cand_indices = [idx for idx in range(len(tokens))]
        rng.shuffle(cand_indices)
        indices = []

        for i, idx in enumerate(sorted(cand_indices[:num_to_corrupt], reverse=True)):
            del tokens[idx]
            del token_types[idx]
            # adjust_idx = idx - (num_to_corrupt - 1 - i)
            # indices += [adjust_idx - 1, adjust_idx]

        return indices

    def corrupt_seq(self, tokens, token_types, rng):
        x = rng.random()
        num_to_corrupt = max(2, int(round(len(tokens) * self.corrupt_per_sentence)))
        if x < (1. / 3.):
            ids = self.corrupt_permute(tokens, token_types, rng, num_to_corrupt)
        elif x < (2. / 3.):
            ids = self.corrupt_replace(tokens, token_types, rng, num_to_corrupt)
        else:
            ids = self.corrupt_insert(tokens, token_types, rng, num_to_corrupt)
        return ids

