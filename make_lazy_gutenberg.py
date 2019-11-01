from blingfire import text_to_sentences
import os
import pickle as pkl
import numpy as np
import re
import unicodedata
from tqdm import tqdm

transl_table = dict( [ (ord(x), ord(y)) for x,y in zip( u"‘’´“”–-",  u"'''\"\"--") ] ) 
#max_doc_length = 512

def convert_into_sentences(text_file):
    paragraphs= []
    stack = []
    for chunk in text_file:
        if not chunk.strip():
            if stack:
                sents = text_to_sentences(
                    " ".join(stack).strip().replace('\n', ' ')).split('\n')
                paragraphs.append(sents)
                stack = []
            continue
        stack.append(chunk.strip())

    if stack:
        sents = text_to_sentences(
            " ".join(stack).strip().replace('\n', ' ')).split('\n')
        paragraphs.append(sents)
    return paragraphs

data_type = "data"
readpath = "/scratch/gobi1/stephaneao/Gutenberg/txt/"
writepath = "/scratch/gobi1/stephaneao/train_data/"
lazypath = os.path.join(writepath, "Gutenberg.lazy/")
if not os.path.exists(lazypath):
    os.makedirs(lazypath)
datapath = os.path.join(lazypath, data_type)
lenpath = os.path.join(lazypath, data_type+'.len.pkl')

doc_separator = "\n".encode('utf-8')
word_total = 0

input_files = [os.path.join(readpath, fn) for fn in os.listdir(readpath)]

with open(datapath, 'wb') as data_file:
    str_lens = []
    for input_file in tqdm(input_files):
        with open(input_file, 'r') as in_f:
            paragraphs = convert_into_sentences(in_f)
            doc_word_count = 0
            str_cnt = 0
            for p in paragraphs:
                p_len = sum([len(s) for s in p])
                words = [len(s.split(' ')) for s in p]
                if p_len < 100 or sum(words) < 10 or np.mean(words) < 5:
                    #print(p)
                    continue
                if sum([len(re.sub(r'[^\w\s.,!?:;\"\'“”‘’]', '', s)) for s in p]) < 0.9 * p_len:
                    continue
                for s in p:
                    if not re.search('[a-zA-Z\n]', s):
                        continue
                    if s.lower() == "@highlight":
                        continue
                    if s[:7].lower() == "chapter":
                        continue
                    if isinstance(s, dict):
                        s = s['text']
                    s += "\n"
                    s = s.translate(transl_table)
                    encoded = unicodedata.normalize('NFKD', s).encode('utf-8')# clean(s)
                    data_file.write(encoded)
                    str_cnt += len(encoded)
                    word_total += len(s.split(' '))
                
                data_file.write(doc_separator)
                str_cnt += len(doc_separator) 
                str_lens.append(str_cnt)
                str_cnt = 0
        
pkl.dump(str_lens, open(lenpath, 'wb'))
print("Total tokens: {}, total docs: {}".format(word_total, len(str_lens)))
