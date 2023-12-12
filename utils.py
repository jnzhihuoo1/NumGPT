## Myvocabulary definition
import json
import spacy
from spacy.symbols import ORTH
import math
CLS_token = "[cls]"
SEP_token = "[sep]"
EOS_token = "[eos]"

 
sp = spacy.load('en_core_web_sm')
def tokenize(raw_data):
    sentence = sp(raw_data)
    token_list = [word.text for word in sentence]
    return token_list

def tokenize_with_pos(raw_data):
    sentence = sp(raw_data)
    token_list = [word.text for word in sentence]
    pos_list = [word.pos_ for word in sentence]
    return token_list, pos_list

def read_json(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
    return data
def save_json(file_name, data):
    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)

def is_number(s):
    try:
        x = float(s)
        if math.isnan(x) or math.isinf(x):
            return False
        return True
    except ValueError:
        pass
 
    return False
def pad_list(token_ids_list, fixed_length, pad_token):
    encoded = token_ids_list[:fixed_length]
    while len(encoded) < fixed_length:
        encoded.append(pad_token)
    return encoded
def transform_tokens_list(tokens_list, block_size):
    selector = []
    numeral = []
    for token in tokens_list:
        if is_number(token):
            selector.append(1)
            numeral.append(float(token))
        else:
            selector.append(0)
            numeral.append(0)
    selector = pad_list(selector, block_size, 0)
    numeral = pad_list(numeral, block_size, 0)
    return selector, numeral


def transform_number(number):
    sign = 1
    fraction = 0
    exp = 0
    if number < 0:
        sign = 0
        number = -number
    # number_str = "%.10e"%(number)
    number_str = "%.18e"%(number)
    arr = number_str.split("e")
    fraction = float(arr[0])
    exp = float(arr[1])
    return sign, fraction, exp
def transform_number_arr(number_arr):
    sign_list = []
    fraction_list = []
    exp_list = []
    for number in number_arr:
        sign, fraction, exp = transform_number(number)
        sign_list.append(sign)
        fraction_list.append(fraction)
        exp_list.append(exp)
    return sign_list, fraction_list, exp_list
def transform_number_2(number):
    sign = 1
    fraction = 0
    exp = 0
    if number < 0:
        sign = -1
        number = -number
    # number_str = "%.10e"%(number)
    number_str = "%.18e"%(number)
    arr = number_str.split("e")
    fraction = float(arr[0]) * sign
    exp = float(arr[1])
    return 1, fraction, exp
def transform_number_arr_2(number_arr):
    sign_list = []
    fraction_list = []
    exp_list = []
    for number in number_arr:
        sign, fraction, exp = transform_number_2(number)
        sign_list.append(sign)
        fraction_list.append(fraction)
        exp_list.append(exp)
    return sign_list, fraction_list, exp_list
def construct_exp_vocab(ne_exp_range_min=-8, ne_exp_range_max=12):
    vocab = {}
    vocab_list = ["-inf","+inf"]
    for i in range(ne_exp_range_min, ne_exp_range_max+1):
        vocab_list.append(str(i))
    for index, item in enumerate(vocab_list):
        vocab[item] = index
    return vocab
    
def transform_number_3(number):
    sign = 1
    fraction = 0
    exp = 0
    if number < 0:
        sign = -1
        number = -number
    # number_str = "%.10e"%(number)
    number_str = "%.18e"%(number)
    arr = number_str.split("e")
    fraction = float(arr[0]) * sign
    exp = int(arr[1])
    return fraction, exp
def transform_number_arr_3(number_arr, exp_vocab, exp_range_min, exp_range_max):
    fraction_list = []
    exp_list = []
    for number in number_arr:
        fraction, exp = transform_number_3(number)
        if exp < exp_range_min:
            new_exp = "-inf"
        elif exp > exp_range_max:
            new_exp = "+inf"
        elif str(exp) in exp_vocab:
            new_exp = str(exp)
        else:
            new_exp = str(0)
        exp = exp_vocab[new_exp]
        fraction_list.append(fraction)
        exp_list.append(exp)
    return fraction_list, exp_list
def transform_number_4(number):
    sign = 1
    fraction = 0
    exp = 0
    if number < 0:
        sign = -1
        number = -number
    # number_str = "%.10e"%(number)
    number_str = "%.18e"%(number)
    arr = number_str.split("e")
    fraction = float(arr[0]) * sign
    exp = float(arr[1])
    return fraction, exp
def transform_number_arr_4(number_arr):
    fraction_list = []
    exp_list = []
    for number in number_arr:
        fraction, exp = transform_number_4(number)
        fraction_list.append(fraction)
        exp_list.append(exp)
    return fraction_list, exp_list
def transform_number_5(number):
    sign = 1
    fraction = 0
    exp = 0
    if number < 0:
        sign = -1
        number = -number
    # number_str = "%.10e"%(number)
    number_str = "%.18e"%(number)
    arr = number_str.split("e")
    fraction = float(arr[0]) * sign
    exp = int(arr[1])
    return fraction, exp
def transform_number_arr_5(number_arr, exp_vocab, exp_range_min, exp_range_max):
    fraction_list = []
    exp_list = []
    real_exp_list = []
    for number in number_arr:
        fraction, exp = transform_number_5(number)
        real_exp_list.append(exp)
        if exp < exp_range_min:
            new_exp = "-inf"
        elif exp > exp_range_max:
            new_exp = "+inf"
        elif str(exp) in exp_vocab:
            new_exp = str(exp)
        else:
            new_exp = str(0)
        exp = exp_vocab[new_exp]
        fraction_list.append(fraction)
        exp_list.append(exp)
    return fraction_list, exp_list, real_exp_list
def transform_number_6(number, significance=6):
    sign = 1
    output_sign = 1
    fraction = 0
    exp = 0
    if number < 0:
        sign = -1
        output_sign = 0
        number = -number
    # number_str = "%.10e"%(number)
    number_str = "%.18e"%(number)
    arr = number_str.split("e")
    fraction = float(arr[0]) 
    fraction_sign = fraction * sign
    fraction_str = str(fraction)
    fraction_digit = []
    counter = 0
    while len(fraction_digit) < significance:
        if counter < len(fraction_str):
            current_digit = fraction_str[counter]
            counter = counter + 1
            try:
                current_digit = int(current_digit)
            except ValueError:
                continue
            fraction_digit.append(current_digit)
        else:
            fraction_digit.append(0)
            counter = counter + 1
    exp = int(arr[1])
    return fraction_sign, exp, output_sign, fraction_digit
def transform_number_arr_6(number_arr, exp_vocab, exp_range_min, exp_range_max, fraction_significance=6):
    fraction_list = []
    exp_list = []
    sign_list = []
    fraction_digit_list = []
    for number in number_arr:
        fraction, exp, sign, fraction_digit = transform_number_6(number, fraction_significance)
        if exp < exp_range_min:
            new_exp = "-inf"
        elif exp > exp_range_max:
            new_exp = "+inf"
        elif str(exp) in exp_vocab:
            new_exp = str(exp)
        else:
            new_exp = str(0)
        exp = exp_vocab[new_exp]
        fraction_list.append(fraction)
        exp_list.append(exp)
        sign_list.append(sign)
        fraction_digit_list.append(fraction_digit)
    return fraction_list, exp_list, sign_list, fraction_digit_list


def get_trainable_parameter_size(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params


#from utils import transform_number_arr
from mingpt.utils import top_k_logits
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
def crop_tensor(x, block_size):
    return x if x.size(1) <= block_size else x[:, -block_size:]

class FCGPTSampler:
    def __init__(self,  model, tokenizer, block_size=512, sci_mode=False):
        self.model = model
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.sci_mode = sci_mode
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
    @torch.no_grad()
    def generate(self, context, steps=20, temperature=1.0, sample=False, top_k=None):
        sci_mode = self.sci_mode
        input_ids, numeral_list, selector_idx = self.tokenizer.tokenize_transform_with_full_numeral(context)
        sign_list, fraction_list, exp_list = transform_number_arr(numeral_list)            
        
        
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        sign = torch.tensor(sign_list, dtype=torch.long).unsqueeze(0).to(self.device)
        fraction = torch.tensor(fraction_list, dtype=torch.float).unsqueeze(0).to(self.device)
        exp = torch.tensor(exp_list, dtype=torch.float).unsqueeze(0).to(self.device)
        selector = torch.tensor(selector_idx, dtype=torch.long).unsqueeze(0).to(self.device)
        model = self.model
        block_size = self.block_size
        model.eval()
        for k in range(steps):
            x_cond = crop_tensor(x, block_size)
            sign_cond = crop_tensor(sign, block_size)
            fraction_cond = crop_tensor(fraction, block_size)
            exp_cond = crop_tensor(exp, block_size)
            selector_cond = crop_tensor(selector, block_size)
            (token_logits, numeral_sign_logits, numeral_fraction_logits, numeral_exp_logits, selector_logits), loss = model(x_cond, sign_cond, fraction_cond, exp_cond, selector_cond)
            # pluck the logits at the final step and scale by temperature
            token_logits = token_logits[:, -1, :] / temperature
            numeral_sign_logits = numeral_sign_logits[:, -1, :]
            numeral_fraction_logits = numeral_fraction_logits[:,-1,:]
            numeral_exp_logits = numeral_exp_logits[:,-1,:]
            selector_logits = selector_logits[:,-1,:]
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                token_logits = top_k_logits(token_logits, top_k)
            # apply softmax to convert to probabilities
            token_probs = F.softmax(token_logits, dim=-1)
            selector_probs = F.softmax(selector_logits, dim=-1)
            numeral_sign_probs = F.softmax(numeral_sign_logits, dim=-1)
            
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(token_probs, num_samples=1)
                iselector = torch.multinomial(selector_probs, num_samples=1)
                inumeral_sign = torch.multinomial(numeral_sign_probs, num_samples=1)
            else:
                _, ix = torch.topk(token_probs, k=1, dim=-1)
                _, iselector = torch.topk(selector_probs, k=1, dim=-1)
                _, inumeral_sign = torch.topk(numeral_sign_probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1).detach()
            sign = torch.cat((sign, inumeral_sign), dim=1).detach()
            selector = torch.cat((selector, iselector), dim=1).detach()
            fraction = torch.cat((fraction, numeral_fraction_logits), dim=1).detach()
            exp = torch.cat((exp, numeral_exp_logits), dim=1).detach()
            #print(x, sign, selector, fraction, exp)
            #print("x.shape: {}, sign.shape: {}, selector.shape: {}, fraction.shape: {}, exp.shape: {}".format(
            #x.shape, sign.shape, selector.shape, fraction.shape, exp.shape))
        x = x.squeeze(0)
        sign = sign.squeeze(0)
        selector = selector.squeeze(0)
        fraction = fraction.squeeze(0)
        exp = exp.squeeze(0)
        text = self.tokenizer.decode_with_numeral(x, sign,  fraction, exp, selector,sci_mode)
        #text = self.tokenizer.decode(x)
        return text
    
class FCGPTV3_2_Sampler:
    def __init__(self,  model, tokenizer, block_size=512, sci_mode=False):
        self.model = model
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.sci_mode = sci_mode
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
    @torch.no_grad()
    def generate(self, context, steps=20, temperature=1.0, sample=False, top_k=None):
        sci_mode = self.sci_mode
        input_ids, numeral_list, selector_idx = self.tokenizer.tokenize_transform_with_full_numeral(context)
        sign_list, fraction_list, exp_list = transform_number_arr(numeral_list)            
        
        
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        sign = torch.tensor(sign_list, dtype=torch.long).unsqueeze(0).to(self.device)
        fraction = torch.tensor(fraction_list, dtype=torch.float).unsqueeze(0).to(self.device)
        exp = torch.tensor(exp_list, dtype=torch.float).unsqueeze(0).to(self.device)
        selector = torch.tensor(selector_idx, dtype=torch.long).unsqueeze(0).to(self.device)
        model = self.model
        block_size = self.block_size
        model.eval()
        raw_model = model.module if hasattr(self.model, "module") else model
        fraction_prototype = raw_model.ne_emb.fraction_emb.prototype.to(self.device)
        exp_prototype = raw_model.ne_emb.exp_emb.prototype.to(self.device)
        for k in range(steps):
            x_cond = crop_tensor(x, block_size)
            sign_cond = crop_tensor(sign, block_size)
            fraction_cond = crop_tensor(fraction, block_size)
            exp_cond = crop_tensor(exp, block_size)
            selector_cond = crop_tensor(selector, block_size)
            (token_logits, numeral_sign_logits, numeral_fraction_logits, numeral_exp_logits, selector_logits), loss = model(x_cond, sign_cond, fraction_cond, exp_cond, selector_cond)
            # pluck the logits at the final step and scale by temperature
            token_logits = token_logits[:, -1, :] / temperature
            numeral_sign_logits = numeral_sign_logits[:, -1, :] 
            numeral_fraction_logits = torch.sum(numeral_fraction_logits[:,-1,:] * fraction_prototype.unsqueeze(0), dim=1).unsqueeze(1)
            numeral_exp_logits = torch.sum(numeral_exp_logits[:,-1,:] * exp_prototype.unsqueeze(0), dim=1).unsqueeze(1)
            #print(fraction_prototype.shape, numeral_fraction_logits.shape)
            #print(exp_prototype.shape, numeral_exp_logits.shape)
            selector_logits = selector_logits[:,-1,:]
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                token_logits = top_k_logits(token_logits, top_k)
            # apply softmax to convert to probabilities
            token_probs = F.softmax(token_logits, dim=-1)
            selector_probs = F.softmax(selector_logits, dim=-1)
            numeral_sign_probs = F.softmax(numeral_sign_logits, dim=-1)
            
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(token_probs, num_samples=1)
                iselector = torch.multinomial(selector_probs, num_samples=1)
                inumeral_sign = torch.multinomial(numeral_sign_probs, num_samples=1)
            else:
                _, ix = torch.topk(token_probs, k=1, dim=-1)
                _, iselector = torch.topk(selector_probs, k=1, dim=-1)
                _, inumeral_sign = torch.topk(numeral_sign_probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1).detach()
            sign = torch.cat((sign, inumeral_sign), dim=1).detach()
            selector = torch.cat((selector, iselector), dim=1).detach()
            fraction = torch.cat((fraction, numeral_fraction_logits), dim=1).detach()
            exp = torch.cat((exp, numeral_exp_logits), dim=1).detach()
            #print(x, sign, selector, fraction, exp)
            #print("x.shape: {}, sign.shape: {}, selector.shape: {}, fraction.shape: {}, exp.shape: {}".format(
            #x.shape, sign.shape, selector.shape, fraction.shape, exp.shape))
        x = x.squeeze(0)
        sign = sign.squeeze(0)
        selector = selector.squeeze(0)
        fraction = fraction.squeeze(0)
        exp = exp.squeeze(0)
        text = self.tokenizer.decode_with_numeral(x, sign, fraction, exp, selector,sci_mode)
        #text = self.tokenizer.decode(x)
        return text
    
def convert_fraction_exp_to_str(fraction, exp):
    return "{}e{}".format(fraction, exp)
def convert_fraction_exp_to_number(fraction, exp):
    return str(fraction*10**exp)
def convert_fraction_exp_to_integer(fraction, exp):
    return str(round(fraction, 5)*10**exp)
class FCGPTV5_Sampler:
    def __init__(self,  model, tokenizer, block_size=512, sci_mode=False, extra_config=None):
        self.model = model
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.sci_mode = sci_mode
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        if not extra_config is None:
            self.ne_exp_range_min = extra_config["ne_exp_range_min"]
            self.ne_exp_range_max = extra_config["ne_exp_range_max"]
            self.ne_exp_vocab = extra_config["ne_exp_vocab"]
            self.ne_exp_decoder = extra_config["ne_exp_decoder"]
    def decode_numeral(self, fraction_list, exp_list):
        sci_mode = self.sci_mode
        numeral_list = []
        for i in range(len(exp_list)):
            fraction = fraction_list[i].item()
            exp = exp_list[i].item()
            real_exp = self.ne_exp_decoder[str(exp)]
            if sci_mode == True:
                numeral = convert_fraction_exp_to_str(fraction, real_exp)
            else:
                if real_exp == "+inf":
                    numeral = str("+inf")
                elif real_exp == "-inf":
                    numeral = str("0")
                else:
                    numeral = convert_fraction_exp_to_number(fraction, int(real_exp))
            numeral_list.append(numeral)
        return numeral_list
    @torch.no_grad()
    def generate(self, context, steps=20, temperature=1.0, sample=False, top_k=None):
        sci_mode = self.sci_mode
        extra_config = self.sci_mode
        input_ids, numeral_list, selector_idx = self.tokenizer.tokenize_transform_with_full_numeral(context)
        fraction_list, exp_list = transform_number_arr_3(numeral_list, self.ne_exp_vocab, self.ne_exp_range_min, self.ne_exp_range_max)           
        
        
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        fraction = torch.tensor(fraction_list, dtype=torch.float).unsqueeze(0).to(self.device)
        exp = torch.tensor(exp_list, dtype=torch.long).unsqueeze(0).to(self.device)
        selector = torch.tensor(selector_idx, dtype=torch.long).unsqueeze(0).to(self.device)
        model = self.model
        block_size = self.block_size
        model.eval()
        raw_model = model.module if hasattr(self.model, "module") else model
        fraction_prototype = raw_model.ne_emb.fraction_emb.prototype.to(self.device)
        for k in range(steps):
            x_cond = crop_tensor(x, block_size)
            fraction_cond = crop_tensor(fraction, block_size)
            exp_cond = crop_tensor(exp, block_size)
            selector_cond = crop_tensor(selector, block_size)
            (token_logits, numeral_fraction_logits, numeral_exp_logits, selector_logits), loss = model(x_cond,  fraction_cond, exp_cond, selector_cond)
            # pluck the logits at the final step and scale by temperature
            token_logits = token_logits[:, -1, :] / temperature
            #torch.set_printoptions(profile="full")
            #if k == 0:
            #    print("numeral fraction logits", numeral_fraction_logits[:,-1,:])
            #    print("fraction prototype", fraction_prototype.unsqueeze(0))
            numeral_last_fraction_logits = numeral_fraction_logits[:,-1,:]
            #numeral_last_fraction_logits = numeral_last_fraction_logits / (torch.sum(numeral_last_fraction_logits,dim=1) + 1e-9)
            numeral_fraction_logits = torch.sum(numeral_last_fraction_logits * fraction_prototype.unsqueeze(0), dim=1).unsqueeze(1)
            
            #if k == 0:
            #    print("new numeral_fraction_logits", numeral_fraction_logits)
            #torch.set_printoptions(profile="default")
            numeral_exp_logits = numeral_exp_logits[:, -1, :]
            #print(fraction_prototype.shape, numeral_fraction_logits.shape)
            #print(exp_prototype.shape, numeral_exp_logits.shape)
            selector_logits = selector_logits[:,-1,:]
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                token_logits = top_k_logits(token_logits, top_k)
            # apply softmax to convert to probabilities
            token_probs = F.softmax(token_logits, dim=-1)
            selector_probs = F.softmax(selector_logits, dim=-1)
            numeral_exp_probs = F.softmax(numeral_exp_logits, dim=-1)
            
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(token_probs, num_samples=1)
                iselector = torch.multinomial(selector_probs, num_samples=1)
                inumeral_exp = torch.multinomial(numeral_exp_probs, num_samples=1)
            else:
                _, ix = torch.topk(token_probs, k=1, dim=-1)
                _, iselector = torch.topk(selector_probs, k=1, dim=-1)
                _, inumeral_exp = torch.topk(numeral_exp_probs, k=1, dim=-1)
                
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1).detach()
            selector = torch.cat((selector, iselector), dim=1).detach()
            fraction = torch.cat((fraction, numeral_fraction_logits), dim=1).detach()
            exp = torch.cat((exp, inumeral_exp), dim=1).detach()
            #print(x, sign, selector, fraction, exp)
            #print("x.shape: {}, sign.shape: {}, selector.shape: {}, fraction.shape: {}, exp.shape: {}".format(
            #x.shape, sign.shape, selector.shape, fraction.shape, exp.shape))
        x = x.squeeze(0)
        selector = selector.squeeze(0)
        fraction = fraction.squeeze(0)
        exp = exp.squeeze(0)
        numeral_list = self.decode_numeral(fraction, exp)
        text = self.tokenizer.decode_with_numeral_2(x, numeral_list, selector)
        #text = self.tokenizer.decode(x)
        return text
import numpy as np
class FCGPTV6_Sampler:
    def __init__(self,  model, tokenizer, block_size=512, sci_mode=False, extra_config=None):
        self.model = model
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.sci_mode = sci_mode
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        if not extra_config is None:
            self.ne_exp_range_min = extra_config["ne_exp_range_min"]
            self.ne_exp_range_max = extra_config["ne_exp_range_max"]
            self.ne_exp_vocab = extra_config["ne_exp_vocab"]
            self.ne_exp_decoder = extra_config["ne_exp_decoder"]
    
    def decode_numeral(self, fraction_list, exp_list):
        sci_mode = self.sci_mode
        numeral_list = []
        for i in range(len(exp_list)):
            fraction = fraction_list[i].item()
            exp = exp_list[i].item()
            real_exp = self.ne_exp_decoder[str(exp)]
            if sci_mode == True:
                numeral = convert_fraction_exp_to_str(fraction, real_exp)
            else:
                if real_exp == "+inf":
                    numeral = str("+inf")
                elif real_exp == "-inf":
                    numeral = str("0")
                else:
                    numeral = convert_fraction_exp_to_number(fraction, int(real_exp))
            numeral_list.append(numeral)
        return numeral_list
    @torch.no_grad()
    def generate(self, context, steps=20, temperature=1.0, sample=False, top_k=None, crop_head=False):
        sci_mode = self.sci_mode
        extra_config = self.sci_mode
        input_ids, numeral_list, selector_idx = self.tokenizer.tokenize_transform_with_full_numeral_revised(context)
        fraction_list, exp_list = transform_number_arr_3(numeral_list, self.ne_exp_vocab, self.ne_exp_range_min, self.ne_exp_range_max)           
        
        input_length = len(input_ids)
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        fraction = torch.tensor(fraction_list, dtype=torch.float).unsqueeze(0).to(self.device)
        exp = torch.tensor(exp_list, dtype=torch.long).unsqueeze(0).to(self.device)
        selector = torch.tensor(selector_idx, dtype=torch.long).unsqueeze(0).to(self.device)

        model = self.model
        block_size = self.block_size
        model.eval()
        raw_model = model.module if hasattr(self.model, "module") else model
        for k in range(steps):
            x_cond = crop_tensor(x, block_size)
            fraction_cond = crop_tensor(fraction, block_size)
            exp_cond = crop_tensor(exp, block_size)
            selector_cond = crop_tensor(selector, block_size)
            (token_logits, numeral_fraction_logits, numeral_exp_logits, selector_logits), loss = model(x_cond,  fraction_cond, exp_cond, selector_cond)
            # pluck the logits at the final step and scale by temperature
            token_logits = token_logits[:, -1, :] / temperature
            torch.set_printoptions(profile="full")
            numeral_fraction_logits = numeral_fraction_logits[:,-1,:]
            numeral_exp_logits = numeral_exp_logits[:, -1, :]
            #print(fraction_prototype.shape, numeral_fraction_logits.shape)
            #print(exp_prototype.shape, numeral_exp_logits.shape)
            selector_logits = selector_logits[:,-1,:]
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                token_logits = top_k_logits(token_logits, top_k)
            # apply softmax to convert to probabilities
            token_probs = F.softmax(token_logits, dim=-1)
            selector_probs = F.softmax(selector_logits, dim=-1)
            numeral_exp_probs = F.softmax(numeral_exp_logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(token_probs, num_samples=1)
                iselector = torch.multinomial(selector_probs, num_samples=1)
                inumeral_exp = torch.multinomial(numeral_exp_probs, num_samples=1)
            else:
                _, ix = torch.topk(token_probs, k=1, dim=-1)
                _, iselector = torch.topk(selector_probs, k=1, dim=-1)
                _, inumeral_exp = torch.topk(numeral_exp_probs, k=1, dim=-1)
                
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1).detach()
            selector = torch.cat((selector, iselector), dim=1).detach()
            fraction = torch.cat((fraction, numeral_fraction_logits), dim=1).detach()
            exp = torch.cat((exp, inumeral_exp), dim=1).detach()
            #print(x, sign, selector, fraction, exp)
            #print("x.shape: {}, sign.shape: {}, selector.shape: {}, fraction.shape: {}, exp.shape: {}".format(
            #x.shape, sign.shape, selector.shape, fraction.shape, exp.shape))
        x = x.squeeze(0)
        selector = selector.squeeze(0)
        fraction = fraction.squeeze(0)
        exp = exp.squeeze(0)
        if crop_head:
            x = x[input_length:]
            selector = selector[input_length:]
            fraction = fraction[input_length:]
            exp = exp[input_length:]
        numeral_list = self.decode_numeral(fraction, exp)
        text = self.tokenizer.decode_with_numeral_2(x, numeral_list, selector)
        #text = self.tokenizer.decode(x)
        return text
class FCGPTV6_Sampler_backup:
    def __init__(self,  model, tokenizer, block_size=512, sci_mode=False, extra_config=None):
        self.model = model
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.sci_mode = sci_mode
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        if not extra_config is None:
            self.ne_exp_range_min = extra_config["ne_exp_range_min"]
            self.ne_exp_range_max = extra_config["ne_exp_range_max"]
            self.ne_exp_vocab = extra_config["ne_exp_vocab"]
            self.ne_exp_decoder = extra_config["ne_exp_decoder"]
            self.ne_fraction_decoder = np.arange(0.0, 10.0, 0.1).tolist()
            
    def get_fraction_decoder_matrix(self, vector, prototype, sigma):
        ## decoder_prototype: [fraction_vocab_size]
        ## return [fraction_vocab_size, fraction_dimension]
        #vector = torch.Tensor(self.ne_fraction_decoder)
        #prototype = self.prototype
        prototype = prototype.to(vector.device)
        #print(vector.shape, prototype.shape)

        ## torch.Size([64, 13]) torch.Size([128])
        seq_size = vector.shape[0]
        dimension = prototype.shape[0]
        vector_expand = vector.unsqueeze(1).repeat(1, dimension)
        prototype_expand = prototype.unsqueeze(0).repeat(seq_size, 1)
        square_diff = (vector_expand - prototype_expand) ** 2 / (sigma ** 2)
        square_diff = torch.exp(-square_diff)
        square_diff = square_diff / torch.sqrt(torch.sum(square_diff**2, dim=1)).unsqueeze(1).repeat(1, dimension)
        return square_diff
    
    def decode_numeral(self, fraction_list, exp_list):
        sci_mode = self.sci_mode
        numeral_list = []
        for i in range(len(exp_list)):
            fraction = fraction_list[i].item()
            exp = exp_list[i].item()
            real_exp = self.ne_exp_decoder[str(exp)]
            if sci_mode == True:
                numeral = convert_fraction_exp_to_str(fraction, real_exp)
            else:
                if real_exp == "+inf":
                    numeral = str("+inf")
                elif real_exp == "-inf":
                    numeral = str("0")
                else:
                    numeral = convert_fraction_exp_to_number(fraction, int(real_exp))
            numeral_list.append(numeral)
        return numeral_list
    @torch.no_grad()
    def generate(self, context, steps=20, temperature=1.0, sample=False, top_k=None, crop_head=False):
        sci_mode = self.sci_mode
        extra_config = self.sci_mode
        input_ids, numeral_list, selector_idx = self.tokenizer.tokenize_transform_with_full_numeral(context)
        fraction_list, exp_list = transform_number_arr_3(numeral_list, self.ne_exp_vocab, self.ne_exp_range_min, self.ne_exp_range_max)           
        
        input_length = len(input_ids)
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        fraction = torch.tensor(fraction_list, dtype=torch.float).unsqueeze(0).to(self.device)
        exp = torch.tensor(exp_list, dtype=torch.long).unsqueeze(0).to(self.device)
        selector = torch.tensor(selector_idx, dtype=torch.long).unsqueeze(0).to(self.device)

        model = self.model
        block_size = self.block_size
        model.eval()
        raw_model = model.module if hasattr(self.model, "module") else model
        fraction_prototype = raw_model.ne_emb.fraction_emb.prototype
        fraction_sigma = raw_model.ne_emb.fraction_emb.sigma
        fraction_vocab = torch.tensor(self.ne_fraction_decoder).to(self.device)
        fraction_decoder_matrix = self.get_fraction_decoder_matrix(fraction_vocab, fraction_prototype, fraction_sigma).transpose(0,1)
        
        for k in range(steps):
            x_cond = crop_tensor(x, block_size)
            fraction_cond = crop_tensor(fraction, block_size)
            exp_cond = crop_tensor(exp, block_size)
            selector_cond = crop_tensor(selector, block_size)
            (token_logits, numeral_fraction_logits, numeral_exp_logits, selector_logits), loss = model(x_cond,  fraction_cond, exp_cond, selector_cond)
            # pluck the logits at the final step and scale by temperature
            token_logits = token_logits[:, -1, :] / temperature
            torch.set_printoptions(profile="full")
            numeral_last_fraction_logits = numeral_fraction_logits[:,-1,:]
            numeral_last_fraction_logits = numeral_last_fraction_logits / torch.sqrt(torch.sum(numeral_last_fraction_logits**2, dim=1)).unsqueeze(1).repeat(1, numeral_last_fraction_logits.shape[1])
            
            numeral_fraction_logits = torch.matmul(numeral_last_fraction_logits, fraction_decoder_matrix)
            numeral_exp_logits = numeral_exp_logits[:, -1, :]
            #print(fraction_prototype.shape, numeral_fraction_logits.shape)
            #print(exp_prototype.shape, numeral_exp_logits.shape)
            selector_logits = selector_logits[:,-1,:]
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                token_logits = top_k_logits(token_logits, top_k)
            # apply softmax to convert to probabilities
            token_probs = F.softmax(token_logits, dim=-1)
            selector_probs = F.softmax(selector_logits, dim=-1)
            numeral_exp_probs = F.softmax(numeral_exp_logits, dim=-1)
            numeral_fraction_probs = F.softmax(numeral_fraction_logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(token_probs, num_samples=1)
                iselector = torch.multinomial(selector_probs, num_samples=1)
                inumeral_exp = torch.multinomial(numeral_exp_probs, num_samples=1)
            else:
                _, ix = torch.topk(token_probs, k=1, dim=-1)
                _, iselector = torch.topk(selector_probs, k=1, dim=-1)
                _, inumeral_exp = torch.topk(numeral_exp_probs, k=1, dim=-1)
                _, inumeral_fraction = torch.topk(numeral_fraction_probs, k=1, dim=-1)
                
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1).detach()
            selector = torch.cat((selector, iselector), dim=1).detach()
            fraction = torch.cat((fraction, fraction_vocab[inumeral_fraction]), dim=1).detach()
            exp = torch.cat((exp, inumeral_exp), dim=1).detach()
            #print(x, sign, selector, fraction, exp)
            #print("x.shape: {}, sign.shape: {}, selector.shape: {}, fraction.shape: {}, exp.shape: {}".format(
            #x.shape, sign.shape, selector.shape, fraction.shape, exp.shape))
        x = x.squeeze(0)
        selector = selector.squeeze(0)
        fraction = fraction.squeeze(0)
        exp = exp.squeeze(0)
        if crop_head:
            x = x[input_length:]
            selector = selector[input_length:]
            fraction = fraction[input_length:]
            exp = exp[input_length:]
        numeral_list = self.decode_numeral(fraction, exp)
        text = self.tokenizer.decode_with_numeral_2(x, numeral_list, selector)
        #text = self.tokenizer.decode(x)
        return text
    

class FCGPTV7_Sampler:
    def __init__(self,  model, tokenizer, block_size=512, sci_mode=False):
        self.model = model
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.sci_mode = sci_mode
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
    def decode_numeral(self, fraction_list, exp_list):
        sci_mode = self.sci_mode
        numeral_list = []
        for i in range(len(exp_list)):
            fraction = fraction_list[i].item()
            exp = exp_list[i].item()
            if sci_mode == True:
                numeral = convert_fraction_exp_to_str(fraction, exp)
            else:
                numeral = convert_fraction_exp_to_number(fraction, exp)
            numeral_list.append(numeral)
        return numeral_list
    @torch.no_grad()
    def generate(self, context, steps=20, temperature=1.0, sample=False, top_k=None, crop_head=False):
        sci_mode = self.sci_mode
        input_ids, numeral_list, selector_idx = self.tokenizer.tokenize_transform_with_full_numeral(context)
        fraction_list, exp_list = transform_number_arr_4(numeral_list)            
        
        input_length = len(input_ids)
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        fraction = torch.tensor(fraction_list, dtype=torch.float).unsqueeze(0).to(self.device)
        exp = torch.tensor(exp_list, dtype=torch.float).unsqueeze(0).to(self.device)
        selector = torch.tensor(selector_idx, dtype=torch.long).unsqueeze(0).to(self.device)
        model = self.model
        block_size = self.block_size
        model.eval()
        raw_model = model.module if hasattr(self.model, "module") else model
        for k in range(steps):
            x_cond = crop_tensor(x, block_size)
            fraction_cond = crop_tensor(fraction, block_size)
            exp_cond = crop_tensor(exp, block_size)
            selector_cond = crop_tensor(selector, block_size)
            #print(x_cond.shape, fraction_cond.shape, exp_cond.shape)
            (token_logits, numeral_fraction_logits, numeral_exp_logits, selector_logits), loss = model(x_cond,  fraction_cond, exp_cond, selector_cond)
            # pluck the logits at the final step and scale by temperature
            token_logits = token_logits[:, -1, :] / temperature
            numeral_fraction_logits = numeral_fraction_logits[:,-1,:]
            numeral_exp_logits = numeral_exp_logits[:, -1, :]
            #print(fraction_prototype.shape, numeral_fraction_logits.shape)
            #print(exp_prototype.shape, numeral_exp_logits.shape)
            selector_logits = selector_logits[:,-1,:]
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                token_logits = top_k_logits(token_logits, top_k)
            # apply softmax to convert to probabilities
            token_probs = F.softmax(token_logits, dim=-1)
            selector_probs = F.softmax(selector_logits, dim=-1)
            
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(token_probs, num_samples=1)
                iselector = torch.multinomial(selector_probs, num_samples=1)
            else:
                _, ix = torch.topk(token_probs, k=1, dim=-1)
                _, iselector = torch.topk(selector_probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1).detach()
            selector = torch.cat((selector, iselector), dim=1).detach()
            fraction = torch.cat((fraction, numeral_fraction_logits), dim=1).detach()
            exp = torch.cat((exp, numeral_exp_logits), dim=1).detach()
            #print(x, sign, selector, fraction, exp)
            #print("x.shape: {}, sign.shape: {}, selector.shape: {}, fraction.shape: {}, exp.shape: {}".format(
            #x.shape, sign.shape, selector.shape, fraction.shape, exp.shape))
        x = x.squeeze(0)
        selector = selector.squeeze(0)
        fraction = fraction.squeeze(0)
        exp = exp.squeeze(0)
        if crop_head:
            x = x[input_length:]
            selector = selector[input_length:]
            fraction = fraction[input_length:]
            exp = exp[input_length:]
        numeral_list = self.decode_numeral(fraction, exp)
        text = self.tokenizer.decode_with_numeral_2(x, numeral_list, selector)
        #text = self.tokenizer.decode_with_numeral(x, fraction, exp, selector,sci_mode)
        #text = self.tokenizer.decode(x)
        return text
class FCGPTV9_Sampler:
    def __init__(self,  model, tokenizer, block_size=512, sci_mode=False, extra_config=None):
        self.model = model
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.sci_mode = sci_mode
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        if not extra_config is None:
            self.ne_exp_range_min = extra_config["ne_exp_range_min"]
            self.ne_exp_range_max = extra_config["ne_exp_range_max"]
            self.ne_exp_vocab = extra_config["ne_exp_vocab"]
            self.ne_exp_decoder = extra_config["ne_exp_decoder"]
    
    def decode_numeral(self, fraction_list, exp_list):
        sci_mode = self.sci_mode
        numeral_list = []
        for i in range(len(exp_list)):
            fraction = fraction_list[i].item()
            exp = exp_list[i].item()
            real_exp = self.ne_exp_decoder[str(exp)]
            if sci_mode == True:
                numeral = convert_fraction_exp_to_str(fraction, real_exp)
            else:
                if real_exp == "+inf":
                    numeral = str("+inf")
                elif real_exp == "-inf":
                    numeral = str("0")
                else:
                    numeral = convert_fraction_exp_to_integer(fraction, int(real_exp))
            numeral_list.append(numeral)
        return numeral_list
    def decode_fraction_from_sign_digit(self, sign, fraction_digit):
        if sign == 1:
            number_sign = 1
        else:
            number_sign = -1
        fraction_str = ""
        for i in range(len(fraction_digit)):
            if i == 0:
                fraction_str = fraction_str + str(fraction_digit[i]) + "."
            else:
                fraction_str = fraction_str + str(fraction_digit[i])
        try:
            current_number = float(fraction_str)
        except ValueError:
            print("WARNING: Decode number from fraction_digit is failed. The fraction_digit is {}.".format(fraction_digit))
            current_number = 0
        current_number = current_number * number_sign
        return current_number
    @torch.no_grad()
    def generate(self, context, steps=20, temperature=1.0, sample=False, top_k=None, crop_head=False):
        sci_mode = self.sci_mode
        extra_config = self.sci_mode
        input_ids, numeral_list, selector_idx = self.tokenizer.tokenize_transform_with_full_numeral_revised(context)
        fraction_list, exp_list = transform_number_arr_3(numeral_list, self.ne_exp_vocab, self.ne_exp_range_min, self.ne_exp_range_max)           
        
        input_length = len(input_ids)
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        fraction = torch.tensor(fraction_list, dtype=torch.float).unsqueeze(0).to(self.device)
        exp = torch.tensor(exp_list, dtype=torch.long).unsqueeze(0).to(self.device)
        selector = torch.tensor(selector_idx, dtype=torch.long).unsqueeze(0).to(self.device)

        model = self.model
        block_size = self.block_size
        model.eval()
        raw_model = model.module if hasattr(self.model, "module") else model
        for k in range(steps):
            x_cond = crop_tensor(x, block_size)
            fraction_cond = crop_tensor(fraction, block_size)
            exp_cond = crop_tensor(exp, block_size)
            selector_cond = crop_tensor(selector, block_size)
            (token_logits, numeral_sign_logits, numeral_fraction_logits, numeral_exp_logits, selector_logits), loss = model(x_cond,  fraction_cond, exp_cond, selector_cond)
            # pluck the logits at the final step and scale by temperature
            token_logits = token_logits[:, -1, :] / temperature
            torch.set_printoptions(profile="full")
            numeral_sign_logits = numeral_sign_logits[:,-1,:]
            numeral_fraction_logits = numeral_fraction_logits[:,-1,:,:]
            numeral_exp_logits = numeral_exp_logits[:, -1, :]
            #print(fraction_prototype.shape, numeral_fraction_logits.shape)
            #print(exp_prototype.shape, numeral_exp_logits.shape)
            selector_logits = selector_logits[:,-1,:]
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                token_logits = top_k_logits(token_logits, top_k)
            # apply softmax to convert to probabilities
            token_probs = F.softmax(token_logits, dim=-1)
            selector_probs = F.softmax(selector_logits, dim=-1)
            numeral_exp_probs = F.softmax(numeral_exp_logits, dim=-1)
            numeral_sign_probs = F.softmax(numeral_sign_logits, dim=-1)
            numeral_fraction_probs = F.softmax(numeral_fraction_logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(token_probs, num_samples=1)
                iselector = torch.multinomial(selector_probs, num_samples=1)
                inumeral_exp = torch.multinomial(numeral_exp_probs, num_samples=1)
            else:
                _, ix = torch.topk(token_probs, k=1, dim=-1)
                _, iselector = torch.topk(selector_probs, k=1, dim=-1)
                _, inumeral_exp = torch.topk(numeral_exp_probs, k=1, dim=-1)
                _, inumeral_sign = torch.topk(numeral_sign_probs, k=1, dim=-1)
                _, inumeral_fraction_digit = torch.topk(numeral_fraction_probs, k=1, dim=-1)
            numeral_fraction = self.decode_fraction_from_sign_digit(inumeral_sign[0].item(), inumeral_fraction_digit.squeeze(0).squeeze(1).tolist()) 
            numeral_fraction = torch.tensor([[numeral_fraction]], dtype=torch.float).to(self.device)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1).detach()
            selector = torch.cat((selector, iselector), dim=1).detach()
            fraction = torch.cat((fraction, numeral_fraction), dim=1).detach()
            exp = torch.cat((exp, inumeral_exp), dim=1).detach()
            #print(x, sign, selector, fraction, exp)
            #print("x.shape: {}, sign.shape: {}, selector.shape: {}, fraction.shape: {}, exp.shape: {}".format(
            #x.shape, sign.shape, selector.shape, fraction.shape, exp.shape))
        x = x.squeeze(0)
        selector = selector.squeeze(0)
        fraction = fraction.squeeze(0)
        exp = exp.squeeze(0)
        if crop_head:
            x = x[input_length:]
            selector = selector[input_length:]
            fraction = fraction[input_length:]
            exp = exp[input_length:]
        numeral_list = self.decode_numeral(fraction, exp)
        text = self.tokenizer.decode_with_numeral_2(x, numeral_list, selector)
        #text = self.tokenizer.decode(x)
        return text