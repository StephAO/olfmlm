from blingfire import text_to_sentences
import os
import pickle as pkl
import re
import unicodedata
from tqdm import tqdm

transl_table = dict( [ (ord(x), ord(y)) for x,y in zip( u"‘’´“”–-",  u"'''\"\"--") ] ) 
max_doc_length = 256

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
readpath = "/scratch/gobi1/datasets/NLP-Corpus/CNN_dailymail/"
writepath = "/scratch/gobi1/stephaneao/"
lazypath = os.path.join(writepath, "cnn_dailymail.lazy/")
if not os.path.exists(lazypath):
    os.makedirs(lazypath)
inputpath_1 = os.path.join(readpath, "cnn/stories/")
inputpath_2 = os.path.join(readpath, "dailymail/stories/")
datapath = os.path.join(lazypath, data_type)
lenpath = os.path.join(lazypath, data_type+'.len.pkl')

doc_separator = "\n".encode('utf-8')
word_total = 0

input_files = [os.path.join(inputpath_1, fn) for fn in os.listdir(inputpath_1)] + \
              [os.path.join(inputpath_2, fn) for fn in os.listdir(inputpath_2)]  

with open(datapath, 'wb') as data_file:
    str_lens = []
    for input_file in tqdm(input_files):
        with open(input_file, 'r') as in_f:
            paragraphs = convert_into_sentences(in_f)
            doc_word_count = 0
            str_cnt = 0
            for p in paragraphs:
                #if sum([len(re.sub('\s+', ' ', re.sub("[^a-zA-Z]", " ", s)).strip().split(' ')) for s in p]) < 10:
                #    continue
                if sum([len(re.sub(r'[^\w\s.,!?:;\"\'“”‘’]', '', s)) for s in p]) < 0.9 * sum([len(s) for s in p]):
                    continue
                for s in p:
                    if not re.search('[a-zA-Z\n]', s):
                        continue
                    if s.lower() == "@highlight":
                        continue
                    if isinstance(s, dict):
                        s = s['text']
                    s += "\n"
                    s = s.translate(transl_table)
                    encoded = unicodedata.normalize('NFKD', s).encode('utf-8')# clean(s)
                    data_file.write(encoded)
                    str_cnt += len(encoded)
                    word_total += len(s.split(' '))
                    doc_word_count += len(s.split(' '))
                
                if doc_word_count >= max_doc_length:
                    data_file.write(doc_separator)
                    str_cnt += len(doc_separator) 
                    str_lens.append(str_cnt)
                    str_cnt = 0
                    doc_word_count = 0
            
            if str_cnt != 0:
                data_file.write(doc_separator)
                str_cnt += len(doc_separator)
                str_lens.append(str_cnt)

        
pkl.dump(str_lens, open(lenpath, 'wb'))
print("Total tokens: ", word_total)
