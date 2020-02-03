from sentence_encoders import data_utils
from multiprocessing import Pool
from blingfire import text_to_sentences
from torch.utils import data
import os
import pickle as pkl
import queue
import re
import unicodedata
import unidecode
from tqdm import tqdm
import torch
from textwrap import shorten

doc_separator = "\n".encode('utf-8')

def sentence_tokenize(tokenizer, sent):
    """tokenize sentence and get token types if tokens=True"""
    tokens = tokenizer.EncodeAsIds(sent).tokenization
    return tokens

def get_doc_len(s, tokenizer):
    toks = 0
    words = len(s.split(' '))
    if tokenizer:
        num_toks = len(sentence_tokenize(tokenizer, s))
        toks += num_toks
        doc_len = num_toks
    else:
        doc_len = words

    return doc_len, toks

def process_document(document, max_doc_length, tokenizer=None):
    str_lens = []
    writes = []
    
    if type(document) == str:
        document = document.split("\n")

    tok_total, word_total, sentence_total, document_total = 0, 0, 0, 0

    string_document = ' '.join(document)
    string_document = re.sub('\s+', ' ', string_document)
    # Remove special characters
    string_document = re.sub(r'[^\w\s.,!?:;\"\'“”‘’]', '', string_document)
    # Filter documents where special characters makes up > 10% of the document
    if float(len(string_document)) < 0.9 * len(' '.join(document)):
        return [], [], 0, 0, 0, 0
    # Filter documents containing less than 10 words
    if len(string_document.split(' ')) < 10: 
        return [], [], 0, 0, 0, 0
    # Filter documents containing less than 100 characters
    if len(string_document) < 100:
        return [], [], 0, 0, 0, 0
    # Filter documents containing a single sentence
    if len(document) < 2:
        return [], [], 0, 0, 0, 0

    num_toks = 0
    doc_len = 0
    doc_bytes = b''
    num_sents = 0
    str_cnt = 0
    for i, s in enumerate(document):
        # Filter sentences that have no letters
        if not re.search('[a-zA-Z\n]', s):
            continue
        # Specific sentence filter for CNN_dailymail which has this frequent tag
        if s.lower() == "@highlight":
            continue
        # Specific sentence filter to remove chapter headings from books
        if s.lower()[:7] == "chapter":
            continue
        if isinstance(s, dict):
            s = s['text']
        # Ensure exactly one terminal newline char
        s = s.strip("\n") + "\n"
        # Translate some weird utf-8 characters to their more regular counterparts
        s = s.translate(DatasetWriter.transl_table)
        # Remove the rest of the weird utf-8 characters
        #s = ''.join([chr(c) for c in s.encode('utf-8') if c < 128]) # [9,10,13] + list(range(32,127))])
        #s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', s)
        s = unidecode.unidecode(s)
        encoded = unicodedata.normalize('NFKD', s).encode('utf-8')  # clean(s)
        doc_bytes += encoded
        str_cnt += len(encoded)
        num_sents += 1

        dl, nt = get_doc_len(s, tokenizer)
        doc_len += dl
        num_toks += nt

        # Split if we've reached the max doc length and there's at least 2 sentences left
        if doc_len >= max_doc_length and num_sents >= 2 and len(document) - i > 3:
            # Update stats
            doc_str = doc_bytes.decode('utf-8')
            num_words = len(doc_str.split(' '))
            tok_total += num_toks
            word_total += num_words
            sentence_total += num_sents
            document_total += 1
            num_sents = 0
            doc_len = 0
            num_toks = 0

            # Append write data
            writes += [doc_bytes + doc_separator]
            str_lens.append(str_cnt + 1)  # + 1 for doc separator
            str_cnt = 0
            doc_bytes = b''

    if str_cnt != 0 and num_sents >= 2:
        # Update stats
        doc_str = doc_bytes.decode('utf-8')
        num_words = len(doc_str.split(' '))
        tok_total += num_toks
        word_total += num_words
        sentence_total += num_sents
        document_total += 1

        # Append write data
        writes += [doc_bytes + doc_separator]
        str_lens.append(str_cnt + 1)  # + 1 for doc separator
    
    return writes, str_lens, tok_total, word_total, sentence_total, document_total

class DatasetWriter:

    transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”––-æ", u"'''\"\"---e")])

    def __init__(self, name, read_path, path_ext=None, max_doc_length=1024, preamble_len=100, from_text_files=False, split_on_newlines=False):
        """
        :param name [string]: Name of the dataset
        :param read_path Union[string, List[string]]: If using text files, the base read path to the files, else a list of datasets
        :param path_ext List[string]: List of extensions to the read path if there are multiple subdirectories to get files from
        :param from_text_files [Bool]: Whether to use text files to read data or existing datasets
        :param split_on_newlines [Bool]: If True, paragraphs are independent documents, if False, tiles are independent documents
        """
        self.read_path = read_path
        self.path_ext = path_ext
        self.max_doc_length = max_doc_length
        self.split_on_newlines = split_on_newlines
        self.preamble_len = preamble_len
        self.from_text_files = from_text_files
        self.doc_separator = "\n".encode('utf-8')
        self.name = name
        self.base_path = "/h/stephaneao/"
        self.lazy_path = os.path.join(self.base_path, self.name + ".lazy/")
        if not os.path.exists(self.lazy_path):
            os.makedirs(self.lazy_path)

        data_type = "data"
        self.write_path = os.path.join(self.lazy_path, data_type)
        self.len_path = os.path.join(self.lazy_path, data_type + '.len.pkl')

        self.tokenizer = None

    def __enter__(self):
        self.write_file = open(self.write_path, 'wb')
        return self

    def __exit__(self, type_, val, tb):
        self.write_file.close()

    def create(self):
        self.str_lens = []
        self.init_dataset_stats()
        doc_iter = self.dataset_iterator(self.read_path) if not self.from_text_files else \
                   self.text_file_iterator(self.read_path, self.path_ext)
        for doc_info in doc_iter:
            if len(doc_info) == 1:
               doc_info = doc_info[0]
            writes, str_lens, toks, words, sents, documents = doc_info 
                
            self.write_document(writes, str_lens)
            self.update_stats(toks, words, sents, documents)
        
        pkl.dump(self.str_lens, open(self.len_path, 'wb'))
        self.print_stats()

    def init_dataset_stats(self):
        self.tok_total = 0
        self.word_total = 0
        self.sentence_total = 0
        self.document_total = 0
        # self.short_q = queue.PriorityQueue(maxsize=5)
        # self.shortest_len = self.max_doc_length
        
    def print_stats(self):
        if type(self.tok_total) == torch.Tensor:
            self.tok_total = self.tok_total.item()
        if type(self.word_total) == torch.Tensor:
            self.word_total = self.word_total.item()
        if type(self.sentence_total) == torch.Tensor:
            self.sentence_total = self.sentence_total.item()
        if type(self.document_total) == torch.Tensor:
            self.document_total = self.document_total.item()
        
        stat_str = ""
        stat_str += "Total number of tokens: {}\n".format(self.tok_total)
        stat_str += "Total number of words: {}\n".format(self.word_total)
        stat_str += "Total number of sentences: {}\n".format(self.sentence_total)
        stat_str += "Total number of documents: {}\n".format(self.document_total)
        stat_str += "Average number of tokens per document: {:.2f}\n".format(float(self.tok_total) / self.document_total)
        stat_str += "Average number of words per document: {:.2f}\n".format(float(self.word_total) / self.document_total)
        stat_str += "Average number of sentences per document: {:.2f}\n".format(float(self.sentence_total) / self.document_total)

        print(stat_str)

    def write_document(self, writes, str_lens):
        assert len(writes) == len(str_lens)
        #if type(writes[0]) == tuple:
        #    writes = [w[0] for w in writes]
        #if type(str_lens[0]) == torch.Tensor:
        #    str_lens = [s.item() for s in str_lens]
        for i in range(len(writes)):
            self.write_file.write(writes[i])
            self.str_lens.append(str_lens[i])

    def update_stats(self, toks, words, sents, documents):
        """Updates stats of dataset. Expects 4 ints."""
        self.tok_total += toks
        self.word_total += words
        self.sentence_total += sents
        self.document_total += documents

    def dataset_iterator(self, paths):
        data_set_args = {
            'path': paths, # ['wikipedia', 'cnn_dailymail', 'gutenberg'],
            'seq_length': 512,
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
        ds, tokenizer = data_utils.make_dataset(**data_set_args)
        self.tokenizer = tokenizer
        ds.SetTokenizer(None)
        print("Starting length:", len(ds))

        fd = FilterDataset(ds, tokenizer, self.max_doc_length)
        #sampler = torch.utils.data.SequentialSampler(fd)
        #batch_sampler = torch.utils.data.BatchSampler(sampler, 1, False)

        data_loader = torch.utils.data.dataloader.DataLoader(fd,
                                                             collate_fn=lambda x: x,
                                                             num_workers=30,
                                                             pin_memory=True)

        dl_iter = iter(data_loader)
        for i in tqdm(range(len(ds))):
            try:
                doc_info = next(dl_iter)
            except (TypeError) as e:
                print("Caught {}".format(e))
                continue
            if len(doc_info[0]) == 0:
                continue
            yield doc_info

    def text_file_iterator(self, base_read_path, read_paths_exts=None):
        """
        Generator to retrieve inputs from a directory of text files
        :param base_read_path String: Base path to read files - full path if there is only one
        :param read_paths_exts List[String]: If there are multiple directories, list of the extensions beyond the base
        :return: yields documents
        """
        if read_paths_exts:
            read_paths = [os.path.join(base_read_path, rpe) for rpe in read_paths_exts]
        else:
            read_paths = [base_read_path]

        input_files = []
        for p in read_paths:
            input_files += [os.path.join(p, fn) for fn in os.listdir(p)]

        for input_file in tqdm(input_files):
            with open(input_file, 'r') as in_f:
                paragraphs = self.convert_into_sentences(in_f)
                doc = []
                for p in paragraphs:
                    # Each paragraph is it's own document
                    if self.split_on_newlines:
                        yield process_document(p, self.max_doc_length)
                    doc += p
                # Each file is it's own document
                if not self.split_on_newlines:
                    yield process_document(doc, self.max_doc_length)

    def convert_into_sentences(self, text_file):
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


class FilterDataset(data.Dataset):
    """
    Abstract bert dataset.
    Arguments:
        ds (Dataset or array-like): data corpus to use for training
        max_seq_len (int): maximum sequence length to use for a sentence pair
        mask_lm_prob (float): proportion of tokens to mask for masked LM
        max_preds_per_seq (int): Maximum number of masked tokens per sentence pair. Default: math.ceil(max_seq_len*mask_lm_prob/10)*10
        dataset_size (int): number of random sentencepairs in the dataset. Default: len(ds)*(len(ds)-1)

    """
    def __init__(self, ds, tokenizer, max_len):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_doc_length = max_len
        self.doc_separator = "\n".encode('utf-8')

    def __len__(self):
        return len(self.ds)

    def get_doc(self, idx):
        """gets text of document corresponding to idx"""
        rtn = self.ds[idx]
        if isinstance(rtn, dict):
            rtn = rtn['text']
        return rtn

    def __getitem__(self, idx):
        document = self.get_doc(idx)
        return process_document(document, self.max_doc_length, self.tokenizer)


if __name__ == "__main__":
    #base_read_path = "/scratch/gobi1/datasets/NLP-Corpus/CNN_dailymail/"
    #read_path_extension = ["cnn/stories/", "dailymail/stories/"]
    base_read_path = ['bookcorpus', 'wikipedia'] #"/h/stephaneao/bookcorpus_clean"  
    read_path_extension = None #["books_large_p1_clean.txt", "books_large_p2_clean.txt"]
    with DatasetWriter("bert_corpus", base_read_path, read_path_extension, from_text_files=False, split_on_newlines=True) as dw:
        dw.create()
