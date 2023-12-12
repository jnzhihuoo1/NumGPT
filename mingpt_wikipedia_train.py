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
InputFeatures = namedtuple("InputFeatures", "input_ids")

def convert_gpt1_instance_to_features(example, block_size):
    input_ids = example["token_ids"]
    if len(input_ids) > block_size:
        logger.info('len(tokens): {}'.format(len(tokens)))
        logger.info('tokens: {}'.format(tokens))
        tokens = tokens[:block_size]
    input_array = np.zeros(block_size, dtype=np.int)
    input_array[:len(input_ids)] = input_ids
    features = InputFeatures(input_ids=input_array)
    return features

class minGPTPregeneratedDataset(Dataset):
    def __init__(self, pregenerated_dir_name, working_dir_name, tokenizer, input_type, reduce_memory=True, debug=False):
        pregenerated_dir = Path(pregenerated_dir_name)
        data_file = pregenerated_dir / "preprocessed_instances.json"
        metrics_file = pregenerated_dir / "preprocessed_metrics.json"
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        block_size = metrics['block_size']
        logger.info('data_file: {}'.format(data_file))
        logger.info('metrics_file: {}'.format(metrics_file)) 
        
        if reduce_memory:
            self.working_dir = Path(working_dir_name)
            os.makedirs(self.working_dir, exist_ok=True)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, block_size))
 
        else:
            input_ids = np.zeros(shape=(num_samples, block_size), dtype=np.int32)
        logger.info("Loading training examples: {}".format(num_samples))
        if debug:
            max_count = 100
            count = 0
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_gpt1_instance_to_features(example, block_size)
                input_ids[i] = features.input_ids
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
        self.vocab_size = tokenizer.vocab_size
        self.input_type = input_type
    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        token_ids = self.input_ids[item].astype(np.int64)
        x = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_x = torch.tensor(token_ids[1:], dtype=torch.long)
        return x, target_x

'''
Sample command: 
python -u mingpt_wikipedia_train.py Wikipedia_minGPT_Pretrain 1 | tee ./logs/log_mingpt_Wiki2_pretrain.txt
python -u mingpt_wikipedia_train.py Wikipedia_minGPT_Pretrain 1 --debug | tee ./logs/log_mingpt_wikipedia_pretrain.txt
'''
os.environ["CUDA_VISIBLE_DEVICES"]="2"
parser = argparse.ArgumentParser(description='General probing task training.')
parser.add_argument('task')
parser.add_argument('input_type', type=int)
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
batch_size = 4 * n_gpu

if input_type == 1:
    print("Pretraining minGPT on {}, n_gpu: {}".format(task, n_gpu))
elif input_type == 5:
    print("Pretraining FCGPT on {}, n_gpu: {}".format(task, n_gpu))
## if task == "Wikipedia_minGPT_Pretrain":
    ckpt_path = "./models/minGPT_PT_Wikipedia/"
    tokenizer_file_name = './openai-gpt' 
    block_size = 512
    pregenerated_dir_name = "./gpt_exp/preprocessed_wikipedia_train_fcgpt_output/"  ## Load from Pregenerated Output
    log_step = 10000
    working_dir_name = './gpt_exp/temp_wikipedia_fcgpt_load/'  ## TEMP FILES
    reduce_memory = True
    vocab = MyOpenAIGPTTokenizer.from_pretrained(tokenizer_file_name)
    pretrain_dataset = minGPTPregeneratedDataset(pregenerated_dir_name, working_dir_name, vocab, input_type, reduce_memory=reduce_memory, debug=debug)



if debug:
    max_epochs = 1
print("max_epochs:", max_epochs, "ckpt_path:", ckpt_path)
    
#print("Train: ", train_dataset[0], "\nTest: ", test_dataset_list[1][0])
#from transformers import OpenAIGPTLMHeadModel, OpenAIGPTConfig
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
def get_numeral_embedding_config(dimension=128):
    ## Assume total dimension is 128
    sign_dimension = int(1/32 * dimension)
    exp_dimension = int(8/32 * dimension)
    fraction_dimension = dimension - sign_dimension - exp_dimension
    ne_config = {
        "ne_sign":{
            "vocab_size":2,
            "dimension": sign_dimension
        },
        "ne_fraction":{
            "sigma": 0.5,
            "rangemin": 0,
            "rangemax": 10,
            "dimension": fraction_dimension
        },
        "ne_exp":{
            "sigma": 0.5,
            "rangemin": -2,
            "rangemax": 5,
            "dimension": exp_dimension
        }
    }
    return ne_config

from mingpt.model import GPT, GPTConfig
mconf = GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
                  n_layer=12, n_head=12, n_embd=768, head_type="Linear", 
                  ne_sigma=20, ne_rangemin=0, ne_rangemax=400, ne_config=get_numeral_embedding_config(int(64)),
                  input_type=pretrain_dataset.input_type, output_class=pretrain_dataset.vocab_size)


## Input Type 4
if input_type == 1:
    model = GPT(mconf)
print(model)
model.train()
#print("number of parameters: ", sum(p.numel() for p in model.parameters()))

from mingpt.trainer import Trainer, MoreInfoV4Trainer, MoreInfoV6Trainer, TrainerConfig
import time
# initialize a trainer instance and kick off training
start_time = time.time()
tconf = TrainerConfig(max_epochs=max_epochs, batch_size=batch_size, learning_rate=6.25e-5,
                      lr_decay=False, warmup_tokens=batch_size*20, final_tokens=2*len(pretrain_dataset)*pretrain_dataset.block_size,
                      num_workers=4, ckpt_path=ckpt_path)
if input_type == 1:
    trainer = Trainer(model, pretrain_dataset, None, tconf)
trainer.train()
print("Training time", time.time() - start_time)

