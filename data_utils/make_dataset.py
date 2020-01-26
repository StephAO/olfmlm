from sentence_encoders import data_utils
from multiprocessing import Pool
from blingfire import text_to_sentences
import os
import pickle as pkl
import queue
import re
import unicodedata
from tqdm import tqdm
from textwrap import shorten

class DatasetWriter:

    transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])

    def __init__(self, name, read_path, path_ext=None, max_doc_length=512, preamble_len=100, from_text_files=False, split_on_newlines=False):
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
        # self.read_files = []
        # for file in self.read_paths:
        #     self.read_files.append(open(file, 'r'))

    def __exit__(self, type_, val, tb):
        self.write_file.close()
        # [rf.close() for rf in self.read_files]
        
    def create(self):
        self.str_lens = []
        self.init_dataset_stats()
        doc_iter = self.dataset_iterator(self.read_path) if not self.from_text_files else \
                  self.text_file_iterator(self.read_path, self.path_ext)
        for doc in doc_iter:
            self.process_document(doc)
        
        pkl.dump(self.str_lens, open(self.len_path, 'wb'))
        self.print_stats()

    def init_dataset_stats(self):
        self.tok_total = 0
        self.word_total = 0
        self.sentence_total = 0
        self.document_total = 0
        self.short_q = queue.PriorityQueue(maxsize=5)
        self.shortest_len = self.max_doc_length
        
    def print_stats(self):
        stat_str = ""
        stat_str += "Total number of tokens: {}\n".format(self.tok_total)
        stat_str += "Total number of words: {}\n".format(self.word_total)
        stat_str += "Total number of sentences: {}\n".format(self.sentence_total)
        stat_str += "Total number of documents: {}\n".format(self.document_total)
        stat_str += "Average number of tokens per document: {:.2f}\n".format(float(self.tok_total) / self.document_total)
        stat_str += "Average number of words per document: {:.2f}\n".format(float(self.word_total) / self.document_total)
        stat_str += "Average number of sentences per document: {:.2f}\n".format(float(self.sentence_total) / self.document_total)
        stat_str += "The shortest 5 sentences were:\n{}\n{}\n{}\n{}\n{}\n".format(self.short_q.get(),
                                                                                  self.short_q.get(),
                                                                                  self.short_q.get(),
                                                                                  self.short_q.get(),
                                                                                  self.short_q.get())
        print(stat_str)

    def sentence_tokenize(self, tokenizer, sent):
        """tokenize sentence and get token types if tokens=True"""
        tokens = tokenizer.EncodeAsIds(sent).tokenization
        return tokens

    def get_doc(self, ds, idx):
        """gets text of document corresponding to idx"""
        rtn = ds[idx]
        if isinstance(rtn, dict):
            rtn = rtn['text']
        return rtn

    def get_doc_len(self, s):
        if self.tokenizer and False:
            num_toks = len(self.sentence_tokenize(self.tokenizer, s))
            self.tok_total += num_toks
            doc_len = num_toks
        else:
            doc_len = len(s.split(' '))

        return doc_len

    def process_document(self, document):
        """
        Filters document (either whole or sentences of the document), and writes remaining text to a lazy loading dataset
        :param document List[str]: document to filter and process
        :return: None
        """
        string_document = ' '.join(document)
        string_document = re.sub('\s+', ' ', string_document)
        # Remove special characters
        string_document = re.sub(r'[^\w\s.,!?:;\"\'“”‘’]', '', string_document)
        # if sum([len(re.sub('\s+', ' ', re.sub("[^a-zA-Z]", " ", s)).strip().split(' ')) for s in p]) < 10:
        #    continue

        # Filter documents where special characters makes up > 10% of the document
        if float(len(string_document)) < 0.9 * len(string_document):
            return None
        # Filter documents containing less than 10 words
        if len(string_document.split(' ')) < 10:
            return None
        # Filter documents conatining less than 100 characters
        if len(string_document) < 100:
            return None
        # Filter documents containing a single sentence
        if len(document) < 2:
            return None

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
            s += "\n"
            s = s.translate(DatasetWriter.transl_table)
            encoded = unicodedata.normalize('NFKD', s).encode('utf-8')  # clean(s)
            doc_bytes += encoded
            str_cnt += len(encoded)
            num_sents += 1

            doc_len += self.get_doc_len(s)

            # Split if we've reached the max doc length and there's at least 2 sentences left
            if doc_len >= self.max_doc_length and num_sents >= 2 and len(document) - i > 3:
                # Update stats
                doc_str = doc_bytes.decode('utf-8')
                self.document_total += 1
                num_words = len(doc_str.split(' '))
                self.word_total += num_words
                self.sentence_total += num_sents
                num_sents = 0
                doc_len = 0
                
                # Write end of document
                self.write_file.write(doc_bytes)
                self.write_file.write(self.doc_separator)
                str_cnt += len(self.doc_separator)
                self.str_lens.append(str_cnt)
                str_cnt = 0
                doc_bytes = b''
                
                # Add preamble from previous document
                #short_s = shorten(s[::-1], width=self.preamble_len, placeholder=' ...')[::-1]
                #short_enc = unicodedata.normalize('NFKD', short_s).encode('utf-8')  # clean(s)
                #self.write_file.write(short_enc)
                #str_cnt = len(short_enc)
                #doc_len += self.update_stats(short_s)

        if str_cnt != 0 and num_sents >= 2:
            # Update stats
            doc_str = doc_bytes.decode('utf-8')
            self.check_shortest(doc_str)
            self.document_total += 1
            num_words = len(doc_str.split(' '))
            self.word_total += num_words
            self.sentence_total += num_sents
            num_sents = 0
            doc_len = 0
            
            # Write end of document
            self.write_file.write(doc_bytes)
            self.write_file.write(self.doc_separator)
            str_cnt += len(self.doc_separator)
            self.str_lens.append(str_cnt)
            str_cnt = 0
            doc_bytes = b''


    def check_shortest(self, doc):
        dl = len(doc)
        if dl < self.shortest_len:
            if self.short_q.full():
                self.short_q.get()
                self.short_q.put((-dl, doc))
                self.shortest_len = dl
            else:
                self.short_q.put((-dl, doc))

    def dataset_iterator(self, paths):
        data_set_args = {
            'path': paths, # ['wikipedia', 'cnn_dailymail', 'gutenberg'],
            'seq_length': 512,
            'shuffle': False,
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
        print(type(ds))
        print("Starting length:", len(ds))
        self.tokenizer = tokenizer
        ds.SetTokenizer(None)

        for i in tqdm(range(len(ds))):
            #if i % 10000 == 0:
            #    print("Reached document {} out of {}".format(i, len(ds)), flush=True)
            doc = self.get_doc(ds, i)

            yield doc

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
                        yield p
                    doc += p
                # Each file is it's own document
                if not self.split_on_newlines:
                    yield doc

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


        # def worker_init():
        #     global ds_
        #     global tokenizer_
        #     ds_, tokenizer_ = data_utils.make_dataset(**data_set_args)
        #     ds_.SetTokenizer(None)
        #
        # def work(self_idx):
        #     start_idx = self_idx * bin_size
        #     end_idx = min((self_idx + 1) * bin_size, len(ds))
        #     word_in_num_docs = {}
        #     for i in range(int(start_idx), int(end_idx)):
        #         doc = get_doc(ds_, i)
        #         tokens = set(sentence_tokenize(tokenizer_, doc))
        #         for tok in tokens:
        #             word_in_num_docs[tok] = word_in_num_docs.get(tok, 0) + 1
        #     print("Finished with bin", self_idx, flush=True)
        #
        #     return word_in_num_docs

if __name__ == "__main__":
    #base_read_path = "/scratch/gobi1/datasets/NLP-Corpus/CNN_dailymail/"
    #read_path_extension = ["cnn/stories/", "dailymail/stories/"]
    base_read_path = ['wikipedia'] #"/h/stephaneao/bookcorpus_clean"
    read_path_extension = None #["books_large_p1_clean.txt", "books_large_p2_clean.txt"]
    with DatasetWriter("bert_corpus", base_read_path, read_path_extension, from_text_files=False, split_on_newlines=True) as dw:
        dw.create()
