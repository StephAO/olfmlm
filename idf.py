from sentence_encoders import data_utils
import numpy as np
from math import ceil, log
from multiprocessing import Pool
from klepto.archives import file_archive, dir_archive
import sys
import time
import tqdm

data_set_args = {
    'path': ['wikipedia'],
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


def sentence_tokenize(tokenizer, sent, sentence_num=0, beginning=False, ending=False):
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
    #print("Starting bin", self_idx, flush=True)
    tf_arch = file_archive("archives/tf_archive_" + str(self_idx))
    start_idx = self_idx * bin_size
    end_idx = min((self_idx + 1) * bin_size, len(ds))
    #print("I am worker {} and I am a taking indices {} to {}".format(self_idx, start_idx, end_idx), flush=True)
    word_in_num_docs = {}
    for i in range(int(start_idx), int(end_idx)):
        doc = get_doc(ds_, i)
        tf_arch[i] = {}
        tokens = sentence_tokenize(tokenizer_, doc)
        for tok in tokens:
            if not tok in tf_arch[i]:
                word_in_num_docs[tok] = word_in_num_docs.get(tok, 0) + 1
                tf_arch[i][tok] = 1
            else:
                tf_arch[i][tok] += 1
        tf_arch[i] = {k: float(v) / len(tokens) for k, v in tf_arch[i].items()}
    #print(len(tf_arch), flush=True)
    tf_arch.dump()
    print("Finished with bin", self_idx, flush=True)

    return word_in_num_docs

#subset = int(sys.argv[1])
#num_subset = 4
ds, tokenizer = data_utils.make_dataset(**data_set_args)
#range_size = ceil(len(ds) / num_subset)
#file_range_s = range_size * subset
#file_range_e = min(range_size * (subset + 1), len(ds))
#print("I am process {} and I am reading files {} to {} from a total of {} files".format(subset, file_range_s, file_range_e, len(ds), flush=True))
num_workers = 32
num_subsets = 10000
bin_size = ceil(len(ds) / (num_subsets))
#subsubset_s, subsubset_e = [(0, 2500), (2500, 5000), (5000, 7500), (7500, 10000)][subset]

start_time = time.time()

with Pool(num_workers, initializer=worker_init) as p:
  result = list(p.imap(work, range(num_subsets))) #subsubset_s, subsubset_e)))

print("Took: ", time.time() - start_time, flush=True)

idfs = {}
for i in range(num_subsets):
    idf = result[i]
    for k, v in idf.items():
        idfs[k] = idfs.get(k, 0) + v

idfs = {k: log(float(len(ds)) / float(idfs[k])) for k in idfs.keys()}

print("Writing idfs to file", flush=True)
file_archive("archives/idf_archive", idfs).dump()
print("Finished writing idfs to file", flush=True)

print("len idf:", len(idfs))

for test_tok in ["a", "is", "the", "fruit", "apple", "pear", "red", "pink", "crimson"]:
    print(test_tok, ":", idfs[tokenizer.TokenToId(test_tok)])
