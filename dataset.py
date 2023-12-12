import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
#from utils import MyVocabulary, tokenize, tokenize_with_pos, transform_number_arr
import pandas as pd
# now let's give the trained model an addition exam
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import sample
from tqdm import tqdm
from utils import *
class ProbingTask_GPTDataset(Dataset):

    def __init__(self, dataset_name, vocab, input_type=1, block_size=15, debug=False):
        self.samples = pd.read_csv(dataset_name).to_dict("records")
        self.block_size = block_size
        self.vocab = vocab
        self.data_size = len(self.samples)
        self.vocab_size = len(vocab)
        self.input_type = input_type
        if self.debug:
            self.data_size = min(self.data_size, 1000)
        print('data size: %d, vocab size: %d.' % (self.data_size, self.vocab_size))
    
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample["question"]
        answer = sample["answer"]
        label = sample["label"]
        #full_sentence = "{} {}".format(sentence1, sentence2)
        full_sentence = "{} {} {} {} {}".format(self.vocab.cls_token, question, self.vocab.sep_token, answer, self.vocab.eos_token)
        encoding = self.vocab(full_sentence, return_tensors='pt',
                             padding="max_length", truncation=True, max_length=self.block_size)
        #encoding = self.vocab(full_sentence, return_tensors='pt')
        if self.input_type == 1:
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            #token_type_ids = encoding["token_type_ids"].squeeze(0)
            y = torch.tensor(label, dtype=torch.long)
            return input_ids, attention_mask, y

        
class MWPAS_GPTDataset(Dataset):

    def __init__(self, dataset_name, vocab, input_type=1, block_size=15, debug=False):
        self.dataset_name = dataset_name
        self.samples = pd.read_csv(dataset_name).to_dict("records")
        self.block_size = block_size
        self.vocab = vocab
        self.data_size = len(self.samples)
        self.vocab_size = len(vocab)
        self.input_type = input_type
        self.debug = debug
        if self.debug:
            self.data_size = min(self.data_size, 10)
        print('data size: %d, vocab size: %d.' % (self.data_size, self.vocab_size))
    
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample["question"]
        answer = str(sample["answer"])
        label = sample["label"]
        #full_sentence = "{} {}".format(sentence1, sentence2)
        full_sentence = "{} {} {} {} {}".format(self.vocab.cls_token, question, self.vocab.sep_token, answer, self.vocab.eos_token)
        encoding = self.vocab(full_sentence, return_tensors='pt',
                             padding="max_length", truncation=True, max_length=self.block_size)
        #encoding = self.vocab(full_sentence, return_tensors='pt')
        if self.input_type == 1:
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            #token_type_ids = encoding["token_type_ids"].squeeze(0)
            y = torch.tensor(label, dtype=torch.long)
            return input_ids, attention_mask, y
class MWPAS_minGPTDataset(Dataset):

    def __init__(self, dataset_name, vocab, input_type=1, block_size=15, debug=False):
        self.dataset_name = dataset_name
        self.samples = pd.read_csv(dataset_name).to_dict("records")
        self.block_size = block_size
        self.vocab = vocab
        self.data_size = len(self.samples)
        self.vocab_size = len(vocab)
        self.input_type = input_type
        self.debug = debug
        if self.debug:
            self.data_size = min(self.data_size, 1000)
        print('data size: %d, vocab size: %d.' % (self.data_size, self.vocab_size))
    
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample["question"]
        answer = sample["answer"]
        label = sample["label"]
        #full_sentence = "{} {}".format(sentence1, sentence2)
        full_sentence = "{} {} {} {} {}".format(self.vocab.cls_token, question, self.vocab.sep_token, answer, self.vocab.eos_token)
        encoding = self.vocab(full_sentence, return_tensors='pt',
                             padding="max_length", truncation=True, max_length=self.block_size)
        #encoding = self.vocab(full_sentence, return_tensors='pt')
        if self.input_type == 1:
            input_ids = encoding['input_ids'].squeeze(0)
            #attention_mask = encoding['attention_mask'].squeeze(0)
            #token_type_ids = encoding["token_type_ids"].squeeze(0)
            y = torch.tensor(label, dtype=torch.long)
            return input_ids, y        
class AgeNumberComparison_GPTDataset(Dataset):

    def __init__(self, dataset_name, vocab, input_type=1, block_size=15, debug=False):
        self.dataset_name = dataset_name
        self.samples = pd.read_csv(dataset_name).to_dict("records")
        self.block_size = block_size
        self.vocab = vocab
        self.data_size = len(self.samples)
        self.vocab_size = len(vocab)
        self.input_type = input_type
        self.debug = debug
        if self.debug:
            self.data_size = min(self.data_size, 1000)
        print('data size: %d, vocab size: %d.' % (self.data_size, self.vocab_size))
    
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample["question"]
        label = sample["label"]
        #full_sentence = "{} {}".format(sentence1, sentence2)
        full_sentence = "{} {} {}".format(self.vocab.cls_token, question, self.vocab.eos_token)
        encoding = self.vocab(full_sentence, return_tensors='pt',
                             padding="max_length", truncation=True, max_length=self.block_size)
        #encoding = self.vocab(full_sentence, return_tensors='pt')
        if self.input_type == 1:
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            #token_type_ids = encoding["token_type_ids"].squeeze(0)
            y = torch.tensor(label, dtype=torch.long)
            return input_ids, attention_mask, y
class AgeNumberComparison_minGPTDataset(Dataset):

    def __init__(self, dataset_name, vocab, input_type=1, block_size=15, debug=False):
        self.dataset_name = dataset_name
        self.samples = pd.read_csv(dataset_name).to_dict("records")
        self.block_size = block_size
        self.vocab = vocab
        self.data_size = len(self.samples)
        self.vocab_size = len(vocab)
        self.input_type = input_type
        self.debug = debug
        if self.debug:
            self.data_size = min(self.data_size, 1000)
        print('data size: %d, vocab size: %d.' % (self.data_size, self.vocab_size))
    
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample["question"]
        label = sample["label"]
        #full_sentence = "{} {}".format(sentence1, sentence2)
        full_sentence = "{} {} {}".format(self.vocab.cls_token, question, self.vocab.eos_token)
        encoding = self.vocab(full_sentence, return_tensors='pt',
                             padding="max_length", truncation=True, max_length=self.block_size)
        #encoding = self.vocab(full_sentence, return_tensors='pt')
        if self.input_type == 1:
            input_ids = encoding['input_ids'].squeeze(0)
            #attention_mask = encoding['attention_mask'].squeeze(0)
            #token_type_ids = encoding["token_type_ids"].squeeze(0)
            y = torch.tensor(label, dtype=torch.long)
            return input_ids, y        
class ProbingTaskDataset(Dataset):

    def __init__(self, dataset_name, vocab, input_type=1, block_size=15):
        self.samples = pd.read_csv(dataset_name).to_dict("records")
        self.block_size = block_size
        self.vocab = vocab
        self.data_size = len(self.samples)
        self.vocab_size = len(vocab)
        self.input_type = input_type
        print('data size: %d, vocab size: %d.' % (self.data_size, self.vocab_size))
    
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample["question"]
        answer = sample["answer"]
        label = sample["label"]
        if self.input_type in [1,2,3]:
            token_list = tokenize(question)
        else:
            question = question + " " + str(answer)
            token_list, pos_list = tokenize_with_pos(question)
        if self.input_type == 1:
            answer_token_list = list(str(answer))
            answer_token_list.reverse()
            token_list = token_list + answer_token_list
            ## No Paddings
            #token_ids = self.vocab.convert_to_ids(token_list)
            ## With Paddings
            token_ids = self.vocab.convert_to_ids_with_padding(token_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            #number = torch.tensor(float(answer), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, y
        elif self.input_type == 2:
            ## No Paddings
            #token_ids = self.vocab.convert_to_ids(token_list)
            ## With Paddings
            token_ids = self.vocab.convert_to_ids_with_padding(token_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            number = torch.tensor(float(answer), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, number, y
        elif self.input_type == 3:
            answer_token_list = list(str(answer))
            answer_token_list.reverse()
            token_list = token_list + answer_token_list
            ## No Paddings
            #token_ids = self.vocab.convert_to_ids(token_list)
            ## With Paddings
            token_ids = self.vocab.convert_to_ids_with_padding(token_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            number = torch.tensor(float(answer), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, number, y
        elif self.input_type == 4:
            token_ids, numeral_list, selector_idx = self.vocab.convert_to_ids_with_numeral_and_padding(token_list, pos_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            number = torch.tensor(numeral_list, dtype=torch.float)
            selector = torch.tensor(selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, number, selector, y
        elif self.input_type == 5:
            token_ids, numeral_list, selector_idx = self.vocab.convert_to_ids_with_numeral_and_padding(token_list, pos_list, self.block_size)
            sign_list, fraction_list, exp_list = transform_number_arr(numeral_list)
            x = torch.tensor(token_ids, dtype=torch.long)
            sign = torch.tensor(sign_list, dtype=torch.long)
            fraction = torch.tensor(fraction_list, dtype=torch.float)
            exp = torch.tensor(exp_list, dtype=torch.float)
            selector = torch.tensor(selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, sign, fraction, exp, selector, y

class NumberComparisonDataset(Dataset):

    def __init__(self, dataset_name, vocab, input_type=1, block_size=15):
        self.samples = pd.read_csv(dataset_name).to_dict("records")
        self.block_size = block_size
        self.vocab = vocab
        self.data_size = len(self.samples)
        self.vocab_size = len(vocab)
        self.input_type = input_type
        print('data size: %d, vocab size: %d.' % (self.data_size, self.vocab_size))
    
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        A = sample["A"]
        B = sample["B"]
        label = sample["Label"]
        
        if self.input_type == 1:
            
            A_token_list = list(str(A))
            A_token_list.reverse()
            B_token_list = list(str(B))
            B_token_list.reverse()
            pad_token_list = [self.vocab.PAD_token]
            token_list = A_token_list + pad_token_list + B_token_list
            ## No Paddings
            #token_ids = self.vocab.convert_to_ids(token_list)
            ## With Paddings
            token_ids = self.vocab.convert_to_ids_with_padding(token_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            #number = torch.tensor(float(answer), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, y
        elif self.input_type == 4:
            token_list = [A, self.vocab.PAD_token, B]
            pos_list = ["NUM","SPACE","NUM"]
            token_ids, numeral_list, selector_idx = self.vocab.convert_to_ids_with_numeral_and_padding(token_list, pos_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            number = torch.tensor(numeral_list, dtype=torch.float)
            selector = torch.tensor(selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, number, selector, y
        elif self.input_type == 5:
            token_list = [A, self.vocab.PAD_token, B]
            pos_list = ["NUM","SPACE","NUM"]
            token_ids, numeral_list, selector_idx = self.vocab.convert_to_ids_with_numeral_and_padding(token_list, pos_list, self.block_size)
            sign_list, fraction_list, exp_list = transform_number_arr(numeral_list)
            x = torch.tensor(token_ids, dtype=torch.long)
            sign = torch.tensor(sign_list, dtype=torch.long)
            fraction = torch.tensor(fraction_list, dtype=torch.float)
            exp = torch.tensor(exp_list, dtype=torch.float)
            selector = torch.tensor(selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, sign, fraction, exp, selector, y


class AdditionClassificationDataset(Dataset):

    def __init__(self, dataset_name, vocab, input_type=1, block_size=20):
        self.samples = pd.read_csv(dataset_name).to_dict("records")
        self.block_size = block_size
        self.vocab = vocab
        self.data_size = len(self.samples)
        self.vocab_size = len(vocab)
        self.input_type = input_type
        print('data size: %d, vocab size: %d.' % (self.data_size, self.vocab_size))
    
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        A = sample["A"]
        B = sample["B"]
        C = sample["C"]
        label = sample["Label"]
        
        if self.input_type == 1:
            
            A_token_list = list(str(A))
            A_token_list.reverse()
            B_token_list = list(str(B))
            B_token_list.reverse()
            C_token_list = list(str(C))
            C_token_list.reverse()
            pad_token_list = [self.vocab.PAD_token]
            token_list = A_token_list + pad_token_list + B_token_list + pad_token_list + C_token_list
            ## No Paddings
            #token_ids = self.vocab.convert_to_ids(token_list)
            ## With Paddings
            token_ids = self.vocab.convert_to_ids_with_padding(token_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            #number = torch.tensor(float(answer), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, y
        elif self.input_type == 4:
            token_list = [A, self.vocab.PAD_token, B, self.vocab.PAD_token, C]
            pos_list = ["NUM","SPACE","NUM","SPACE","NUM"]
            token_ids, numeral_list, selector_idx = self.vocab.convert_to_ids_with_numeral_and_padding(token_list, pos_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            number = torch.tensor(numeral_list, dtype=torch.float)
            selector = torch.tensor(selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, number, selector, y
        elif self.input_type == 5:
            token_list = [A, self.vocab.PAD_token, B, self.vocab.PAD_token, C]
            pos_list = ["NUM","SPACE","NUM","SPACE","NUM"]
            token_ids, numeral_list, selector_idx = self.vocab.convert_to_ids_with_numeral_and_padding(token_list, pos_list, self.block_size)
            sign_list, fraction_list, exp_list = transform_number_arr(numeral_list)
            x = torch.tensor(token_ids, dtype=torch.long)
            sign = torch.tensor(sign_list, dtype=torch.long)
            fraction = torch.tensor(fraction_list, dtype=torch.float)
            exp = torch.tensor(exp_list, dtype=torch.float)
            selector = torch.tensor(selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, sign, fraction, exp, selector, y

class MultiplicationClassificationDataset(Dataset):

    def __init__(self, dataset_name, vocab, input_type=1, block_size=20):
        self.samples = pd.read_csv(dataset_name).to_dict("records")
        self.block_size = block_size
        self.vocab = vocab
        self.data_size = len(self.samples)
        self.vocab_size = len(vocab)
        self.input_type = input_type
        print('data size: %d, vocab size: %d.' % (self.data_size, self.vocab_size))
    
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        A = sample["A"]
        B = sample["B"]
        C = sample["C"]
        label = sample["Label"]
        
        if self.input_type == 1:
            
            A_token_list = list(str(A))
            A_token_list.reverse()
            B_token_list = list(str(B))
            B_token_list.reverse()
            C_token_list = list(str(C))
            C_token_list.reverse()
            pad_token_list = [self.vocab.PAD_token]
            token_list = A_token_list + pad_token_list + B_token_list + pad_token_list + C_token_list
            ## No Paddings
            #token_ids = self.vocab.convert_to_ids(token_list)
            ## With Paddings
            token_ids = self.vocab.convert_to_ids_with_padding(token_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            #number = torch.tensor(float(answer), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, y
        elif self.input_type == 4:
            token_list = [A, self.vocab.PAD_token, B, self.vocab.PAD_token, C]
            pos_list = ["NUM","SPACE","NUM","SPACE","NUM"]
            token_ids, numeral_list, selector_idx = self.vocab.convert_to_ids_with_numeral_and_padding(token_list, pos_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            number = torch.tensor(numeral_list, dtype=torch.float)
            selector = torch.tensor(selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, number, selector, y
        elif self.input_type == 5:
            token_list = [A, self.vocab.PAD_token, B, self.vocab.PAD_token, C]
            pos_list = ["NUM","SPACE","NUM","SPACE","NUM"]
            token_ids, numeral_list, selector_idx = self.vocab.convert_to_ids_with_numeral_and_padding(token_list, pos_list, self.block_size)
            sign_list, fraction_list, exp_list = transform_number_arr(numeral_list)
            x = torch.tensor(token_ids, dtype=torch.long)
            sign = torch.tensor(sign_list, dtype=torch.long)
            fraction = torch.tensor(fraction_list, dtype=torch.float)
            exp = torch.tensor(exp_list, dtype=torch.float)
            selector = torch.tensor(selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, sign, fraction, exp, selector, y

class Numeracy600KDataset(Dataset):

    def __init__(self, dataset_name, vocab, input_type=1, block_size=15):
        self.samples = read_json(dataset_name)
        self.block_size = block_size
        self.vocab = vocab
        self.data_size = len(self.samples)
        self.vocab_size = len(vocab)
        self.input_type = input_type
        print('data size: %d, vocab size: %d.' % (self.data_size, self.vocab_size))
    
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        token_ids_list = sample["token_ids_list"]
        label = sample["magnitude"]
        if self.input_type == 1:

            ## No Paddings
            #token_ids = self.vocab.convert_to_ids(token_list)
            ## With Paddings
            pad_token_ids = self.vocab.pad_token_ids(token_ids_list, self.block_size)
            x = torch.tensor(pad_token_ids, dtype=torch.long)
            #number = torch.tensor(float(answer), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, y
        elif self.input_type == 2:
            ## No Paddings
            #token_ids = self.vocab.convert_to_ids(token_list)
            ## With Paddings
            token_ids = self.vocab.convert_to_ids_with_padding(token_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            number = torch.tensor(float(answer), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, number, y
        elif self.input_type == 3:
            answer_token_list = list(str(answer))
            answer_token_list.reverse()
            token_list = token_list + answer_token_list
            ## No Paddings
            #token_ids = self.vocab.convert_to_ids(token_list)
            ## With Paddings
            token_ids = self.vocab.convert_to_ids_with_padding(token_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            number = torch.tensor(float(answer), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, number, y
        elif self.input_type == 4:
            token_ids, numeral_list, selector_idx = self.vocab.convert_to_ids_with_numeral_and_padding(token_list, pos_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            number = torch.tensor(numeral_list, dtype=torch.float)
            selector = torch.tensor(selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, number, selector, y
        elif self.input_type == 5:
            numeral_list = sample["ne_numeral_list"]
            selector_idx = sample["ne_selector_idx"]
            token_ids_list = sample["ne_token_ids_list"]
            label = sample["magnitude"]
            pad_token_ids = self.vocab.pad_token_ids(token_ids_list, self.block_size)
            pad_numeral_list = self.vocab.pad_array(numeral_list, self.block_size)
            pad_selector_idx = self.vocab.pad_array(selector_idx, self.block_size)
            sign_list, fraction_list, exp_list = transform_number_arr(pad_numeral_list)
            x = torch.tensor(pad_token_ids, dtype=torch.long)
            sign = torch.tensor(sign_list, dtype=torch.long)
            fraction = torch.tensor(fraction_list, dtype=torch.float)
            exp = torch.tensor(exp_list, dtype=torch.float)
            selector = torch.tensor(pad_selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, sign, fraction, exp, selector, y
        '''
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
        '''

class Numeracy600KFactcheckDataset(Dataset):

    def __init__(self, dataset_name, vocab, input_type=1, block_size=15):
        self.samples = read_json(dataset_name)
        self.block_size = block_size
        self.vocab = vocab
        self.data_size = len(self.samples)
        self.vocab_size = len(vocab)
        self.input_type = input_type
        print('data size: %d, vocab size: %d.' % (self.data_size, self.vocab_size))
    
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.input_type == 1:
            token_ids_list = sample["token_ids_list"]
            label = sample["label"]
            ## No Paddings
            #token_ids = self.vocab.convert_to_ids(token_list)
            ## With Paddings
            pad_token_ids = self.vocab.pad_token_ids(token_ids_list, self.block_size)
            x = torch.tensor(pad_token_ids, dtype=torch.long)
            #number = torch.tensor(float(answer), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, y
        elif self.input_type == 4:
            numeral_list = sample["ne_numeral_list"]
            selector_idx = sample["ne_selector_idx"]
            token_ids_list = sample["ne_token_ids_list"]
            label = sample["label"]
            pad_token_ids = self.vocab.pad_token_ids(token_ids_list, self.block_size)
            pad_numeral_list = self.vocab.pad_array(numeral_list, self.block_size)
            pad_selector_idx = self.vocab.pad_array(selector_idx, self.block_size)
            x = torch.tensor(pad_token_ids, dtype=torch.long)
            number = torch.tensor(pad_numeral_list, dtype=torch.float)
            selector = torch.tensor(pad_selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, number, selector, y
        elif self.input_type == 5:
            numeral_list = sample["ne_numeral_list"]
            selector_idx = sample["ne_selector_idx"]
            token_ids_list = sample["ne_token_ids_list"]
            label = sample["label"]
            pad_token_ids = self.vocab.pad_token_ids(token_ids_list, self.block_size)
            pad_numeral_list = self.vocab.pad_array(numeral_list, self.block_size)
            pad_selector_idx = self.vocab.pad_array(selector_idx, self.block_size)
            sign_list, fraction_list, exp_list = transform_number_arr(pad_numeral_list)
            x = torch.tensor(pad_token_ids, dtype=torch.long)
            sign = torch.tensor(sign_list, dtype=torch.long)
            fraction = torch.tensor(fraction_list, dtype=torch.float)
            exp = torch.tensor(exp_list, dtype=torch.float)
            selector = torch.tensor(pad_selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, sign, fraction, exp, selector, y

class MultiNLIDataset(Dataset):

    def __init__(self, dataset_name, vocab, input_type=1, block_size=15):
        self.dataset_name = dataset_name
        self.samples = read_json(dataset_name)
        self.block_size = block_size
        self.vocab = vocab
        self.data_size = len(self.samples)
        self.vocab_size = len(vocab)
        self.input_type = input_type
        print('data size: %d, vocab size: %d.' % (self.data_size, self.vocab_size))
    
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        token_ids_list = sample["token_ids_list"]
        label = sample["label"]
        if self.input_type == 1:

            ## No Paddings
            #token_ids = self.vocab.convert_to_ids(token_list)
            ## With Paddings
            pad_token_ids = self.vocab.pad_token_ids(token_ids_list, self.block_size)
            x = torch.tensor(pad_token_ids, dtype=torch.long)
            #number = torch.tensor(float(answer), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, y
        elif self.input_type == 2:
            ## No Paddings
            #token_ids = self.vocab.convert_to_ids(token_list)
            ## With Paddings
            token_ids = self.vocab.convert_to_ids_with_padding(token_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            number = torch.tensor(float(answer), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, number, y
        elif self.input_type == 3:
            answer_token_list = list(str(answer))
            answer_token_list.reverse()
            token_list = token_list + answer_token_list
            ## No Paddings
            #token_ids = self.vocab.convert_to_ids(token_list)
            ## With Paddings
            token_ids = self.vocab.convert_to_ids_with_padding(token_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            number = torch.tensor(float(answer), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, number, y
        elif self.input_type == 4:
            token_ids, numeral_list, selector_idx = self.vocab.convert_to_ids_with_numeral_and_padding(token_list, pos_list, self.block_size)
            x = torch.tensor(token_ids, dtype=torch.long)
            number = torch.tensor(numeral_list, dtype=torch.float)
            selector = torch.tensor(selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, number, selector, y
        elif self.input_type == 5:
            numeral_list = sample["ne_numeral_list"]
            selector_idx = sample["ne_selector_idx"]
            token_ids_list = sample["ne_token_ids_list"]
            label = sample["label"]
            pad_token_ids = self.vocab.pad_token_ids(token_ids_list, self.block_size)
            pad_numeral_list = self.vocab.pad_array(numeral_list, self.block_size)
            pad_selector_idx = self.vocab.pad_array(selector_idx, self.block_size)
            sign_list, fraction_list, exp_list = transform_number_arr(pad_numeral_list)
            x = torch.tensor(pad_token_ids, dtype=torch.long)
            sign = torch.tensor(sign_list, dtype=torch.long)
            fraction = torch.tensor(fraction_list, dtype=torch.float)
            exp = torch.tensor(exp_list, dtype=torch.float)
            selector = torch.tensor(pad_selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, sign, fraction, exp, selector, y
        '''
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
        '''


def give_exam(model,trainer, dataset, batch_size=32, max_batches=-1):
    model.eval()
    results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        y_pred, _ = model(x)
        y_real_pred = torch.argmax(y_pred, dim=1).cpu()
        correct = (y_real_pred == y).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
        
    print("final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))
def give_exam_revised(model,trainer, dataset, batch_size=32, max_batches=-1):
    model.eval()
    results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    datasetName = dataset.dataset_name
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        y_pred, _ = model(x)
        y_real_pred = torch.argmax(y_pred, dim=1).cpu()
        if "quant" in datasetName.lower() or "newsnli" in datasetName.lower():
            y_real_pred[y_real_pred>=1] = 1
        elif "awpnli" in datasetName.lower():
            y_real_pred[y_real_pred>=1] = 2
        correct = (y_real_pred == y).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
        
    print("(revised) final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))   
def give_exam_revised_mirco_marco(model,trainer, dataset, batch_size=32, max_batches=-1):
    model.eval()
    results = []
    pred_results = []
    gt_results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    datasetName = dataset.dataset_name
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        y_pred, _ = model(x)
        y_real_pred = torch.argmax(y_pred, dim=1).cpu()
        if "quant" in datasetName.lower() or "newsnli" in datasetName.lower():
            y_real_pred[y_real_pred>=1] = 1
        elif "awpnli" in datasetName.lower():
            y_real_pred[y_real_pred>=1] = 2
        correct = (y_real_pred == y).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            pred_results.append(int(y_real_pred[i]))
            gt_results.append(int(y[i].cpu()))
    print("(revised) final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))   
    micro_f1 = f1_score(gt_results, pred_results, average='micro')
    macro_f1 = f1_score(gt_results, pred_results, average='macro')
    print("micro_f1:", micro_f1, "macro_f1", macro_f1)  
    
def give_exam_type_2(model,trainer, dataset, batch_size=32, max_batches=-1):
    model.eval()
    results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for b, (x, number, y) in enumerate(loader):
        x = x.to(trainer.device)
        number = number.to(trainer.device)
        y_pred, _ = model(x, number)
        y_real_pred = torch.argmax(y_pred, dim=1).cpu()
        correct = (y_real_pred == y).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
        
    print("final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))
    
def give_exam_type_3(model,trainer, dataset, batch_size=32, max_batches=-1):
    model.eval()
    results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for b, (x, number, selector, y) in enumerate(loader):
        x = x.to(trainer.device)
        number = number.to(trainer.device)
        selector = selector.to(trainer.device)
        y_pred, _ = model(x, number, selector)
        y_real_pred = torch.argmax(y_pred, dim=1).cpu()
        correct = (y_real_pred == y).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
        
    print("final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))

def dump_instance_to_str(dataset, x, number, selector, y):
    token_list = dataset.vocab.convert_to_token_list(x.tolist())
    numeral_list = number.tolist()
    selector_list = selector.tolist()
    label = y.tolist()
    combined_token_list = []
    for j in range(len(selector_list)):
        selector = selector_list[j]
        if selector:
            current_token = str(int(numeral_list[j]))
        else:
            current_token = token_list[j]
        combined_token_list.append(current_token)
    #print(combined_token_list)
    combined_str = " ".join(combined_token_list)
    #label = label[0]
    combined_str = "Input: "+ combined_str + "  Label: " + str(label)
    return combined_str
def give_exam_type_3_verbose(model,trainer, dataset, batch_size=32, max_batches=-1, split="Train"):
    model.eval()
    results = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for b, (x, number, selector, y) in enumerate(loader):
        x = x.to(trainer.device)
        number = number.to(trainer.device)
        selector = selector.to(trainer.device)
        y_pred, _ = model(x, number, selector)
        y_real_pred = torch.argmax(y_pred, dim=1).cpu()
        correct = (y_real_pred == y).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            correctness = int(correct[i])
            #if not correctness:
            #    print("Wrong - ", dump_instance_to_str(dataset, x[i], number[i], selector[i], y[i]))
    print("final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))
def give_exam_type_4_verbose(model,trainer, dataset, batch_size=32, max_batches=-1, split="Train"):
    model.eval()
    results = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for b, (x, sign, fraction, exp, selector, y) in enumerate(loader):
        x = x.to(trainer.device)
        sign = sign.to(trainer.device)
        fraction = fraction.to(trainer.device)
        exp = exp.to(trainer.device)
        selector = selector.to(trainer.device)
        y_pred, _ = model(x, sign, fraction, exp, selector)
        y_real_pred = torch.argmax(y_pred, dim=1).cpu()
        correct = (y_real_pred == y).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            correctness = int(correct[i])
            #if not correctness:
            #    print("Wrong - ", dump_instance_to_str(dataset, x[i], number[i], selector[i], y[i]))
    print("final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))
def give_exam_type_8_verbose(model,trainer, dataset, batch_size=32, max_batches=-1, split="Train"):
    model.eval()
    results = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for b, (x, fraction, exp, selector, y) in enumerate(loader):
        x = x.to(trainer.device)
        fraction = fraction.to(trainer.device)
        exp = exp.to(trainer.device)
        selector = selector.to(trainer.device)
        y_pred, _ = model(x, fraction, exp, selector)
        y_real_pred = torch.argmax(y_pred, dim=1).cpu()
        correct = (y_real_pred == y).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            correctness = int(correct[i])
            #if not correctness:
            #    print("Wrong - ", dump_instance_to_str(dataset, x[i], number[i], selector[i], y[i]))
    print("final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))
def give_exam_type_9_verbose(model,trainer, dataset, batch_size=32, max_batches=-1, split="Train"):
    model.eval()
    results = []
    pred_results = []
    gt_results = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for b, (x, fraction, exp, selector, y) in enumerate(loader):
        x = x.to(trainer.device)
        fraction = fraction.to(trainer.device)
        exp = exp.to(trainer.device)
        selector = selector.to(trainer.device)
        y_pred, _ = model(x, fraction, exp, selector)
        y_real_pred = torch.argmax(y_pred, dim=1).cpu()
        correct = (y_real_pred == y).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            correctness = int(correct[i])
            pred_results.append(int(y_real_pred[i]))
            gt_results.append(int(y[i].cpu()))
            #if not correctness:
            #    print("Wrong - ", dump_instance_to_str(dataset, x[i], number[i], selector[i], y[i]))
    print("final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))
    micro_f1 = f1_score(gt_results, pred_results, average='micro')
    macro_f1 = f1_score(gt_results, pred_results, average='macro')
    print("micro_f1:", micro_f1, "macro_f1", macro_f1)
from sklearn.metrics import f1_score    
def micro_marco_f1_measure_gpt1(model, trainer, dataset, batch_size=8, max_batches=-1):
    model.eval()
    pred_results = []
    gt_results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for b, (input_ids, attention_mask, y) in tqdm(enumerate(loader), total=len(loader)):
        input_ids = input_ids.to(trainer.device)
        attention_mask = attention_mask.to(trainer.device)
        outputs = model(input_ids, attention_mask=attention_mask)
        y_pred = outputs.logits
        y_real_pred = torch.argmax(y_pred, dim=1).cpu()
        for i in range(input_ids.size(0)):
            pred_results.append(int(y_real_pred[i]))
            gt_results.append(int(y[i].cpu()))
    micro_f1 = f1_score(gt_results, pred_results, average='micro')
    macro_f1 = f1_score(gt_results, pred_results, average='macro')
    print("micro_f1:", micro_f1, "macro_f1", macro_f1)
def micro_marco_f1_measure(model, trainer, dataset, batch_size=128, max_batches=-1):
    model.eval()
    pred_results = []
    gt_results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        y_pred, _ = model(x)
        y_real_pred = torch.argmax(y_pred, dim=1).cpu()
        for i in range(x.size(0)):
            pred_results.append(int(y_real_pred[i]))
            gt_results.append(int(y[i].cpu()))
    micro_f1 = f1_score(gt_results, pred_results, average='micro')
    macro_f1 = f1_score(gt_results, pred_results, average='macro')
    print("micro_f1:", micro_f1, "macro_f1", macro_f1)

def micro_marco_f1_measure_type_4(model, trainer, dataset, batch_size=128, max_batches=-1):
    model.eval()
    pred_results = []
    gt_results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for b, (x, number, selector, y) in enumerate(loader):
        x = x.to(trainer.device)
        number = number.to(trainer.device)
        selector = selector.to(trainer.device)
        y_pred, _ = model(x, number, selector)
        y_real_pred = torch.argmax(y_pred, dim=1).cpu()
        for i in range(x.size(0)):
            pred_results.append(int(y_real_pred[i]))
            gt_results.append(int(y[i].cpu()))
    micro_f1 = f1_score(gt_results, pred_results, average='micro')
    macro_f1 = f1_score(gt_results, pred_results, average='macro')
    print("micro_f1:", micro_f1, "macro_f1", macro_f1)

def micro_marco_f1_measure_type_5(model, trainer, dataset, batch_size=128, max_batches=-1):
    model.eval()
    pred_results = []
    gt_results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for b, (x, sign, fraction, exp, selector, y) in enumerate(loader):
        x = x.to(trainer.device)
        sign = sign.to(trainer.device)
        fraction = fraction.to(trainer.device)
        exp = exp.to(trainer.device)
        selector = selector.to(trainer.device)
        y_pred, _ = model(x, sign, fraction, exp, selector)
        y_real_pred = torch.argmax(y_pred, dim=1).cpu()
        for i in range(x.size(0)):
            pred_results.append(int(y_real_pred[i]))
            gt_results.append(int(y[i].cpu()))
    micro_f1 = f1_score(gt_results, pred_results, average='micro')
    macro_f1 = f1_score(gt_results, pred_results, average='macro')
    print("micro_f1:", micro_f1, "macro_f1", macro_f1)