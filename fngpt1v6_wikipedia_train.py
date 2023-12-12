from __future__ import absolute_import, division, print_function

from dataset import *
from mingpt.utils import set_seed
set_seed(42)
import torch
import argparse
from transformers import BertTokenizer
from transformers import OpenAIGPTTokenizer, OpenAIGPTForSequenceClassification
from fcgpt_tokenizer import MyOpenAIGPTTokenizer
import argparse
import csv
import logging
import os

import random
import sys
import json
import time

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import (DataLoader, RandomSampler,Dataset)
from collections import namedtuple
from pathlib import Path
import logging
from utils import transform_number_arr
# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

logger = logging.getLogger(__name__)
InputFeatures = namedtuple("InputFeatures", "input_ids numeral_list selector_idx")

def convert_fcgpt1_instance_to_features(example, block_size):
    input_ids = example["token_ids"]
    numeral_list = example["numeral_list"]
    selector_idx = example["selector_idx"]
    if len(input_ids) > block_size:
        logger.info('len(tokens): {}'.format(len(tokens)))
        logger.info('tokens: {}'.format(tokens))
        tokens = tokens[:block_size]
        numeral_list = numeral_list[:block_size]
        selector_idx = numeral_list[:block_size]
    input_array = np.zeros(block_size, dtype=np.int)
    input_array[:len(input_ids)] = input_ids
    numeral_array = np.zeros(block_size, dtype=np.float)
    numeral_array[:len(numeral_list)] = numeral_list
    selector_idx_array = np.zeros(block_size, dtype=np.float)
    selector_idx_array[:len(selector_idx)] = selector_idx
    features = InputFeatures(input_ids=input_array, numeral_list=numeral_array, selector_idx=selector_idx_array)
    return features

class FCGPTPregeneratedDataset(Dataset):
    def __init__(self, pregenerated_dir_name, working_dir_name, tokenizer, input_type, reduce_memory=True, debug=False, extra_config=None):
        pregenerated_dir = Path(pregenerated_dir_name)
        data_file = pregenerated_dir / "preprocessed_instances.json"
        metrics_file = pregenerated_dir / "preprocessed_metrics.json"
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        block_size = metrics['block_size']
        if not extra_config is None:
            self.ne_exp_range_min = extra_config["ne_exp_range_min"]
            self.ne_exp_range_max = extra_config["ne_exp_range_max"]
            self.ne_exp_vocab = extra_config["ne_exp_vocab"]
            
        logger.info('data_file: {}'.format(data_file))
        logger.info('metrics_file: {}'.format(metrics_file)) 
        
        if reduce_memory:
            self.working_dir = Path(working_dir_name)
            os.makedirs(self.working_dir, exist_ok=True)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, block_size))
            numeral_list = np.memmap(filename=self.working_dir/'numeral_list.memmap',
                                  mode='w+', dtype=np.float, shape=(num_samples, block_size))
            selector_idx = np.memmap(filename=self.working_dir/'selector_idx.memmap',
                                  mode='w+', dtype=np.float, shape=(num_samples, block_size))
        else:
            input_ids = np.zeros(shape=(num_samples, block_size), dtype=np.int32)
            numeral_list = np.zeros(shape=(num_samples, block_size), dtype=np.float)
            selector_idx = np.zeros(shape=(num_samples, block_size), dtype=np.float)
        logger.info("Loading training examples: {}".format(num_samples))
        if debug:
            max_count = 100
            count = 0
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_fcgpt1_instance_to_features(example, block_size)
                input_ids[i] = features.input_ids
                numeral_list[i] = features.numeral_list
                selector_idx[i] = features.selector_idx
                if debug:
                    count = count + 1
                    if max_count <= count:
                        break
        logger.info("Loading complete!")
        if debug:
            num_samples = max_count
        self.num_samples = num_samples
        self.block_size = block_size
        self.input_ids = input_ids
        self.numeral_list = numeral_list
        self.selector_idx = selector_idx
        self.vocab_size = tokenizer.vocab_size
        self.input_type = input_type
    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        if self.input_type == 8:
            token_ids = self.input_ids[item].astype(np.int64)
            numeral_list = self.numeral_list[item].astype(np.float)
            selector_idx = self.selector_idx[item].astype(np.float)
    
            token_ids_list = token_ids
            fraction_list, exp_list = transform_number_arr_3(numeral_list, self.ne_exp_vocab, self.ne_exp_range_min, self.ne_exp_range_max)
            x = torch.tensor(token_ids_list[:-1], dtype=torch.long)
            fraction = torch.tensor(fraction_list[:-1], dtype=torch.float)
            exp = torch.tensor(exp_list[:-1], dtype=torch.long)
            selector = torch.tensor(selector_idx[:-1], dtype=torch.float)
            target_x = torch.tensor(token_ids_list[1:], dtype=torch.long)
            target_fraction = torch.tensor(fraction_list[1:], dtype=torch.float)
            target_exp = torch.tensor(exp_list[1:], dtype=torch.long)
            target_selector = torch.tensor(selector_idx[1:], dtype=torch.long)
            return x, fraction, exp, selector, target_x, target_fraction, target_exp, target_selector
        #return (torch.tensor(self.input_ids[item].astype(np.int64)),
        #       torch.tensor(self.numeral_list[item].astype(np.float)),
        #       torch.tensor(self.selector_idx[item].astype(np.float)))

'''
Sample command: 
python -u fngpt1v6_wikipedia_train.py Wikipedia_GPT1_Pretrain 8 --debug | tee ./logs/log_fngpt1v6_wikipedia_pretrain.txt
'''
os.environ["CUDA_VISIBLE_DEVICES"]="1"
parser = argparse.ArgumentParser(description='General probing task training.')
parser.add_argument('task')
parser.add_argument('input_type', type=int)
parser.add_argument('--ne_exp_range_min', type=int, default=-8)
parser.add_argument('--ne_exp_range_max', type=int, default=12)
parser.add_argument('--debug', action="store_const", default=False, const=True,
                        help='If true, run preprocess in debug mode.')

args = parser.parse_args()
#print("VARS:", vars(args))
## Settings

debug = args.debug
max_epochs = 3
input_type = args.input_type
task = args.task
#task = "Numeracy600KFactcheck_comment"
output_class = 2
n_gpu = torch.cuda.device_count()
batch_size = 4 

if input_type == 8:
    print("Pretraining FN2GPT1V3 on {}, n_gpu: {}".format(task, n_gpu))
    ne_exp_config = {
        "ne_exp_range_min":args.ne_exp_range_min,
        "ne_exp_range_max":args.ne_exp_range_max,
        "ne_exp_vocab":construct_exp_vocab(args.ne_exp_range_min, args.ne_exp_range_max)
    }
    ne_exp_config["ne_exp_vocab_size"] = len(ne_exp_config["ne_exp_vocab"])
    ne_exp_decoder = {}
    for i, (k, v) in enumerate(ne_exp_config["ne_exp_vocab"].items()):
        ne_exp_decoder[str(v)] = k
    ne_exp_config["ne_exp_decoder"] = ne_exp_decoder
if task == "Wikipedia_GPT1_Pretrain":
    ckpt_path = "./models/FN2GPT1V6_PT_Wikipedia/"
    tokenizer_file_name = './openai-gpt' 
    block_size = 512
    pregenerated_dir_name = "./gpt_exp/preprocessed_wikipedia_train_fcgpt_output/"  ## Load from Pregenerated Output
    log_step = 10000
    working_dir_name = './gpt_exp/temp_wikipedia_fcgpt_load/'  ## TEMP FILES
    reduce_memory = True
    vocab = MyOpenAIGPTTokenizer.from_pretrained(tokenizer_file_name)
    pretrain_dataset = FCGPTPregeneratedDataset(pregenerated_dir_name, working_dir_name, vocab, input_type, reduce_memory=reduce_memory, debug=debug, extra_config=ne_exp_config)


if debug:
    max_epochs = 1
print("max_epochs:", max_epochs, "ckpt_path:", ckpt_path)
    

def load_partial_weight(model, model_path, token_embedding_patch=True, verbose=True):
    model_state = model.state_dict()
    pretrained_state = torch.load(model_path)
    if token_embedding_patch:
        if model_state['tokens_embed.weight'].shape[1] == pretrained_state['tokens_embed.weight'].shape[1]:
            token_embedding_patched_num = model_state['tokens_embed.weight'].shape[0] - pretrained_state['tokens_embed.weight'].shape[0]
            token_embedding_dim = pretrained_state['tokens_embed.weight'].shape[1]
            patched_emb = torch.randn(token_embedding_patched_num, token_embedding_dim) * 0.02
            pretrained_state['tokens_embed.weight'] = torch.cat((pretrained_state['tokens_embed.weight'], patched_emb), dim=0)
            print("Patch token embedding num: {}".format(token_embedding_patched_num), "Patched token embedding shape: ", pretrained_state['tokens_embed.weight'].shape)
    not_loaded_keys = [k for k,v in pretrained_state.items() if not (k in model_state and v.size() == model_state[k].size())]
    pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() } 
    if len(not_loaded_keys) > 0:
        print("Discard some pretrained weights:")
        print(not_loaded_keys)
    model_state.update(pretrained_state)
    model.load_state_dict(model_state)

## Floating number 
def get_numeral_embedding_config_4(dimension=128, exp_vocab_size=23):
    ## Assume total dimension is 128
    exp_dimension = int(8/32 * dimension)
    fraction_dimension = dimension - exp_dimension
    ne_config = {
        "ne_fraction":{
            "sigma": 5,
            "rangemin": -10,
            "rangemax": 10,
            "dimension": fraction_dimension
        },
        "ne_exp":{
            "vocab_size":exp_vocab_size,
            "dimension": exp_dimension,
        },
        "total_dimension":dimension
    }
    return ne_config


from mingpt.model import GPTNLMv10,GPTConfig
mconf = GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
                  n_layer=12, n_head=12, n_embd=768, head_type="Linear", 
                  ne_sigma=20, ne_rangemin=0, ne_rangemax=400, ne_config=get_numeral_embedding_config_4(int(64), ne_exp_config["ne_exp_vocab_size"]),
                  input_type=pretrain_dataset.input_type, output_class=pretrain_dataset.vocab_size)



## Input Type 8
if input_type == 8:
    model = GPTNLMv10(mconf)

print(model)
model.train()
#print("number of parameters: ", sum(p.numel() for p in model.parameters()))

from mingpt.trainer import Trainer,  GPTLMV7Trainer
import time
# initialize a trainer instance and kick off training
start_time = time.time()
tconf = TrainerConfig(max_epochs=max_epochs, batch_size=batch_size, learning_rate=6.25e-5,
                      lr_decay=False, warmup_tokens=batch_size*20, final_tokens=2*len(pretrain_dataset)*pretrain_dataset.block_size,
                      num_workers=4, ckpt_path=ckpt_path)
if input_type == 8:
    trainer = GPTLMV7Trainer(model, pretrain_dataset, None, tconf)
trainer.train()
print("Training time", time.time() - start_time)

