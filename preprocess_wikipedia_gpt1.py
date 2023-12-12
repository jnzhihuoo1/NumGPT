from pathlib import Path
from transformers import OpenAIGPTTokenizer, OpenAIGPTForSequenceClassification
from fcgpt_tokenizer import MyOpenAIGPTTokenizer
import json
import collections
import logging
import os
import shelve
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
from multiprocessing import Pool

import numpy as np
from random import random, randrange, randint, shuffle, choice
from tqdm import tqdm, trange
import json
import os
import time
class DocumentDatabase:
    def __init__(self, reduce_memory=False, working_dir_name='./gpt_exp/temp/'):
        if reduce_memory:
            self.temp_dir = None
            self.working_dir = Path(working_dir_name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        #print("--enter--")
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        #print("--exit--")
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

def create_instances_from_document(doc_database, doc_idx, block_size, tokenizer, current_chunk=[]):
    document = doc_database[doc_idx]
    max_num_tokens = block_size
    instances = []
    for i in range(len(document)):
        segment = document[i]
        current_chunk.extend(segment)
    current_length = len(current_chunk)
    current_idx = 0
    for i in range(0, current_length-block_size+1, block_size): # Truncate in block of block_size
        instances.append({
            "token_ids":tokenizer.convert_tokens_to_ids(current_chunk[i:i+block_size])                                                        
        })
        current_idx = i+block_size
    if current_length > current_idx:
        current_chunk = current_chunk[current_idx:]
    else:
        current_chunk = []
    return instances, current_chunk



def create_training_file(docs, output_dir, block_size, tokenizer):
    os.makedirs(output_dir, exist_ok = True)
    epoch_filename = output_dir / "preprocessed_instances.json"
    num_instances = 0
    current_chunk = []
    with epoch_filename.open('w') as epoch_file:
        for doc_idx in trange(len(docs), desc="Document"):
            doc_instances, current_chunk = create_instances_from_document(docs, doc_idx, block_size, tokenizer, current_chunk)
            doc_instances = [json.dumps(instance) for instance in doc_instances]
            for instance in doc_instances:
                epoch_file.write(instance + '\n')
                num_instances += 1
    metrics_filename = output_dir / "preprocessed_metrics.json"
    with metrics_filename.open('w') as metrics_file:
        metrics = {
            "num_training_examples": num_instances,
            "block_size": block_size
        }
        metrics_file.write(json.dumps(metrics))
    print("Training files statistics:", metrics)
    return epoch_filename, metrics_filename


'''
Sample command:
python -u preprocess_wikipedia_gpt1.py | tee ./logs/preprocess_wikipedia_gpt1.txt
'''
### User parameters
train_data_file_name = "./corpus_raw/corpus.xml"
tokenizer_file_name = './openai-gpt' 
block_size = 512
output_dir_name = "./gpt_exp/preprocessed_wikipedia_output/"
log_step = 100000
working_dir_name = './gpt_exp/temp/'  ## TEMP FILES

### Main
args = {
    "train_data_file_name":train_data_file_name,
    "tokenizer_file_name":tokenizer_file_name,
    "block_size":block_size,
    "output_dir_name":output_dir_name,
    "log_step":log_step,
    "working_dir_name":working_dir_name
}
print("Running args:", args)
start_time = time.time()
train_data_file = Path(train_data_file_name)
tokenizer = MyOpenAIGPTTokenizer.from_pretrained(tokenizer_file_name)
vocab_list = list(tokenizer.encoder.keys())
doc_num = 0
with DocumentDatabase(reduce_memory=True, working_dir_name=working_dir_name) as docs:
    with train_data_file.open() as f:
        doc = []
        for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
            line = line.strip()
            if line == "":
                docs.add_document(doc)
                doc = []
                doc_num += 1
                if doc_num % log_step == 0:
                    print('loaded {} docs!'.format(doc_num))
                    #break
            else:
                tokens = tokenizer.tokenize(line)
                doc.append(tokens)
        if doc:
            docs.add_document(doc)  # If the last doc didn't end on a newline, make sure it still gets added
    print("Loaded time:", time.time() - start_time)
    start_time = time.time()
    output_dir = Path(output_dir_name)
    epoch_filename, metrics_filename = create_training_file(docs, output_dir, block_size, tokenizer)
    print("Dataset filename:", epoch_filename)
    print("Metrics filename:", metrics_filename)
    print("Create training file time:", time.time() - start_time)