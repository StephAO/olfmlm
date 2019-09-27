from sentence_encoders import data_utils
from math import ceil, log
from multiprocessing import Pool
import time

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
    str_type = 'str' + str(sentence_num)
    token_types = [tokenizer.get_type(str_type).Id] * len(tokens)
    return tokens, token_types

def get_doc(ds, idx):
    """gets text of document corresponding to idx"""
    rtn = ds[idx]
    if isinstance(rtn, dict):
        rtn = rtn['text']
    return rtn

def work(self_idx):
    start_idx = self_idx * bin_size
    end_idx = min((self_idx + 1) * bin_size, len(ds))
    print("I am worker {} and I am a taking indices {} to {}".format(self_idx, start_idx, end_idx))
    word_in_num_docs = {}
    for i in range(start_idx, end_idx):
        doc = get_doc(ds, i)
        tokens, _ = sentence_tokenize(tokenizer, doc)
        tokens = set(tokens)
        for tok in tokens:
            word_in_num_docs[tok] = word_in_num_docs.get(tok, 0) + 1
    results[self_idx] = word_in_num_docs

ds, tokenizer = data_utils.make_dataset(**data_set_args)
ds.SetTokenizer(None)
ds = ds[:10000]

num_workers = 1
threads = [None] * num_workers
results = [None] * num_workers
bin_size = ceil(len(ds) / num_workers)

start_time = time.time()
pool = Pool(num_workers)
pool.map_async(work, range(num_workers))

    # do some other stuff

print("Took: ", time.time() - start_time)

# idfs = {k: log(float(len(ds)) / float(word_in_num_docs[k])) for k in word_in_num_docs.keys()}


for test_tok in ["a", "is", "the", "apple", "pear", "red", "pink", "magenta", "crimson"]:
    print(test_tok, ":", idfs[tokenizer.TokenToId(test_tok)])



