"""
Script to calculate the inverse document frequency (idf) used in tf-idf labels of a dataset.
"""

from olfmlm import data_utils
import numpy as np
from math import ceil, log
from multiprocessing import Pool
import pickle
import sys
import time
import tqdm

data_set_args = {
    'path': ['bert_corpus'],#, 'cnn_dailymail', 'gutenberg'],
    'seq_length': 128,
    'lazy': True,
    'delim': ',',
    'text_key': 'text',
    'label_key': 'label',
    'non_binary_cols': None,
    'split': [1.],
    'loose': False,
    'tokenizer_type': 'BertWordPieceTokenizer',
    'tokenizer_model_path': 'tokenizer.model',
    'vocab_size': 30522,
    'model_type': 'bert-base-uncased',
    'cache_dir': 'cache_dir',
    'max_preds_per_seq': 80,
    'presplit_sentences': True,
}


def sentence_tokenize(tokenizer, sent):
    """tokenize sentence and get token types if tokens=True"""
    tokens = tokenizer.EncodeAsIds(sent).tokenization
    return tokens

def get_doc(ds, idx):
    """gets text of document corresponding to idx"""
    rtn = ds[idx]
    if isinstance(rtn, dict):
        rtn = rtn['text']
    return rtn

def worker_init():
    global ds_
    global tokenizer_
    ds_, tokenizer_ = data_utils.make_dataset(**data_set_args)
    ds_.SetTokenizer(None)

def work(self_idx):
    start_idx = self_idx * bin_size
    end_idx = min((self_idx + 1) * bin_size, len(ds))
    word_in_num_docs = {}
    for i in range(int(start_idx), int(end_idx)):
        doc = get_doc(ds_, i)
        tokens = set(sentence_tokenize(tokenizer_, doc))
        for tok in tokens:
            word_in_num_docs[tok] = word_in_num_docs.get(tok, 0) + 1
    print("Finished with bin", self_idx, flush=True)

    return word_in_num_docs

ds, tokenizer = data_utils.make_dataset(**data_set_args)

num_workers = 32
num_subsets = 10000
bin_size = ceil(len(ds) / (num_subsets))

print("Total size:", len(ds))

start_time = time.time()

with Pool(num_workers, initializer=worker_init) as p:
  result = list(p.imap(work, range(num_subsets)))

print("Took: ", time.time() - start_time, flush=True)

idfs = {}
for i in range(num_subsets):
    idf = result[i]
    for k, v in idf.items():
        idfs[k] = idfs.get(k, 0) + v

idfs = {k: log(float(len(ds)) / float(idfs[k])) for k in idfs.keys()}

print("Writing idfs to file", flush=True)
with open("idf.p", "wb") as f:
    pickle.dump(idfs, f)
print("Finished writing idfs to file", flush=True)

print("len idf:", len(idfs))

for test_tok in ["a", "is", "the", "fruit", "apple", "pear", "red", "pink", "crimson"]:
    print(test_tok, ":", idfs[tokenizer.TokenToId(test_tok)])
