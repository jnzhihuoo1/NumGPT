# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)
# make deterministic
from mingpt.utils import set_seed
set_seed(42)
import torch

from dataset import *
import argparse
from transformers import BertTokenizer
from transformers import OpenAIGPTTokenizer, OpenAIGPTForSequenceClassification
from fcgpt_tokenizer import MyOpenAIGPTTokenizer
      
'''
Sample command: 
python -u mingpt_cls_task_train.py MWPAS_GPT 1 | tee ./logs/log_mingpt_MWPAS_train.txt
python -u mingpt_cls_task_train.py ProbingTask_GPT 1 | tee ./logs/log_mingpt_ProbingTask_train.txt
python -u mingpt_cls_task_train.py GeneralNumberComparison_GPT 1 | tee ./logs/log_mingpt_GNC_train.txt

'''

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
batch_size = 32
task = args.task
#task = "Numeracy600KFactcheck_comment"
output_class = 2
if input_type == 1:
    print("Training GPT1 on {}".format(task))

elif task == "MWPAS_GPT":
    block_size = 128
    output_class = 2
    max_epochs = 50
    vocab = OpenAIGPTTokenizer.from_pretrained('./openai-gpt')
    vocab.add_special_tokens({'pad_token': '[PAD]'})
    vocab.add_special_tokens({'cls_token': '[CLS]'})
    vocab.add_special_tokens({'sep_token': '[SEP]'})
    vocab.add_special_tokens({'eos_token': '[EOS]'})
    file_name = "./data/mwpas_training_samples.csv"
    train_dataset = MWPAS_minGPTDataset(file_name, vocab, input_type=input_type, block_size=block_size, debug=debug)
    test_file_name = "./data/mwpas_test_samples.csv"
    test_dataset = MWPAS_minGPTDataset(test_file_name, vocab, input_type=input_type, block_size=block_size, debug=debug)  
elif task == "ProbingTask_GPT":
    block_size = 128
    output_class = 2
    max_epochs = 50
    vocab = OpenAIGPTTokenizer.from_pretrained('./openai-gpt')
    vocab.add_special_tokens({'pad_token': '[PAD]'})
    vocab.add_special_tokens({'cls_token': '[CLS]'})
    vocab.add_special_tokens({'sep_token': '[SEP]'})
    vocab.add_special_tokens({'eos_token': '[EOS]'})
    ckpt_path = "./models/minGPT1_FT_MME/"
    file_name = "./data/mme_training_samples.csv"
    train_dataset = MWPAS_minGPTDataset(file_name, vocab, input_type=input_type, block_size=block_size, debug=debug)
    test_file_name = "./data/mme_test_samples.csv"
    test_dataset = MWPAS_minGPTDataset(test_file_name, vocab, input_type=input_type, block_size=block_size, debug=debug)
elif task == "GeneralNumberComparison_GPT":
    block_size = 128
    output_class = 2
    max_epochs = 50
    vocab = MyOpenAIGPTTokenizer.from_pretrained('./openai-gpt')
    vocab.add_special_tokens({'pad_token': '[PAD]'})
    vocab.add_special_tokens({'cls_token': '[CLS]'})
    vocab.add_special_tokens({'sep_token': '[SEP]'})
    vocab.add_special_tokens({'eos_token': '[EOS]'})
    ckpt_path = "./models/minGPT1_FT_GNC/"
    file_name = "./data/gnc_training_samples.csv"
    train_dataset = AgeNumberComparison_minGPTDataset(file_name, vocab, input_type=input_type, block_size=block_size)
    test_file_name = "./data/gnc_test_samples.csv"
    test_dataset = AgeNumberComparison_minGPTDataset(test_file_name, vocab, input_type=input_type, block_size=block_size)  

if debug:
    max_epochs = 1
print("max_epochs:", max_epochs, "ckpt_path:", ckpt_path)
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

from mingpt.model import GPTClassification, GPTClassificationV3, GPTClassificationV4,GPTClassificationV5, GPTConfig
model_block_size = 512
mconf = GPTConfig(train_dataset.vocab_size, model_block_size,
                  n_layer=12, n_head=12, n_embd=768, head_type="Linear", 
                  ne_sigma=400, ne_rangemin=0, ne_rangemax=40000,
                  ne_config=get_numeral_embedding_config(int(64)),
                  input_type=train_dataset.input_type, output_class=output_class)


## Input Type 4
if input_type == 1:
    model = GPTClassification(mconf)
print(model)




def load_partial_weight_v1(model, model_path, token_embedding_patch=True, verbose=True):
    model_state = model.state_dict()
    pretrained_state = torch.load(model_path)
    token_embedding_key = 'tok_emb.weight'
    if token_embedding_patch:
        if model_state[token_embedding_key].shape[1] == pretrained_state[token_embedding_key].shape[1]:
            token_embedding_patched_num = model_state[token_embedding_key].shape[0] - pretrained_state[token_embedding_key].shape[0]
            token_embedding_dim = pretrained_state[token_embedding_key].shape[1]
            patched_emb = torch.randn(token_embedding_patched_num, token_embedding_dim) * 0.02
            pretrained_state[token_embedding_key] = torch.cat((pretrained_state[token_embedding_key], patched_emb.cuda()), dim=0)
            print("Patch token embedding num: {}".format(token_embedding_patched_num), "Patched token embedding shape: ", pretrained_state[token_embedding_key].shape)
    not_loaded_keys = [k for k,v in pretrained_state.items() if not (k in model_state and v.size() == model_state[k].size())]
    pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() } 
    if len(not_loaded_keys) > 0:
        print("Discard some pretrained weights:")
        print(not_loaded_keys)
    model_state.update(pretrained_state)
    model.load_state_dict(model_state)
    
#model_path = "./models/minGPT_PT_Wikipedia/0_epochs.pkl"
model_path = "./models/minGPT1_FT_GNC/49_epochs.pkl"
#model_path = "./models/BERT_BASE_UNCASED_FT_MultiNLI/0_epochs.pkl"
#load_partial_weight_v1(model, model_path, token_embedding_patch=False)


from mingpt.trainer import Trainer, TrainerConfig

import time
# initialize a trainer instance and kick off training
start_time = time.time()
#ckpt_path = None
tconf = TrainerConfig(max_epochs=max_epochs, batch_size=batch_size, learning_rate=6.25e-5,
                      lr_decay=False, warmup_tokens=batch_size*20, final_tokens=2*len(train_dataset)*train_dataset.block_size,
                      num_workers=4, ckpt_path=ckpt_path)

if input_type == 1:
    trainer = Trainer(model, train_dataset, None, tconf)

trainer.train()
print("Training time", time.time() - start_time)


print("Train:")
test_batch_size = 8
if input_type == 1:
    give_exam_revised(model, trainer, train_dataset, batch_size)



if task == "AgeNumberComparison_GPT" or task == "MWPAS_GPT" or task == "ProbingTask_GPT" or task == "GeneralNumberComparison_GPT":
    print("Test:")
    if input_type == 1:
        #give_gpt1_exam(model, trainer, train_dataset, test_batch_size)
        give_exam_revised(model,trainer, test_dataset, batch_size)
    