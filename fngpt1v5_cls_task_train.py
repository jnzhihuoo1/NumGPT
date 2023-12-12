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
from transformers import OpenAIGPTTokenizer, OpenAIGPTForSequenceClassification
from fcgpt_tokenizer import MyOpenAIGPTTokenizer

ENABLE_NEW_NUMBER_TRANSFORM=True
ENABLE_NEW_TOKENIZER=True

            
        
class MWPAS_FCGPTDataset(Dataset):

    def __init__(self, dataset_name, vocab, input_type=1, block_size=15, debug=False, extra_config=None):
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
        if not extra_config is None:
            self.ne_exp_range_min = extra_config["ne_exp_range_min"]
            self.ne_exp_range_max = extra_config["ne_exp_range_max"]
            self.ne_exp_vocab = extra_config["ne_exp_vocab"]
        print('data size: %d, vocab size: %d.' % (self.data_size, self.vocab_size))
    
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sentence1 = sample["question"]
        sentence2 = str(sample["answer"])
        label = sample["label"]
        if self.input_type == 5:
            #full_sentence = "{} {} {} {} {}".format(self.vocab.cls_token, sentence1, self.vocab.sep_token, sentence2, self.vocab.eos_token)
            if not ENABLE_NEW_TOKENIZER:
                input_ids, numeral_list, selector_idx = self.vocab.tokenize_transform_with_full_numeral_two_sentences(sentence1, sentence2, self.block_size)
            else:
                input_ids, numeral_list, selector_idx = self.vocab.tokenize_transform_with_full_numeral_two_sentences_revised(sentence1, sentence2, self.block_size)
            
            token_ids_list = input_ids
            #pad_token_ids = self.vocab.pad_token_ids(token_ids_list, self.block_size)
            #pad_numeral_list = self.vocab.pad_array(numeral_list, self.block_size)
            #pad_selector_idx = self.vocab.pad_array(selector_idx, self.block_size)
            if not ENABLE_NEW_NUMBER_TRANSFORM:
                sign_list, fraction_list, exp_list = transform_number_arr(numeral_list)
            else:
                sign_list, fraction_list, exp_list = transform_number_arr_2(numeral_list)
            x = torch.tensor(token_ids_list, dtype=torch.long)
            sign = torch.tensor(sign_list, dtype=torch.long)
            fraction = torch.tensor(fraction_list, dtype=torch.float)
            exp = torch.tensor(exp_list, dtype=torch.float)
            selector = torch.tensor(selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, sign, fraction, exp, selector, y
        elif self.input_type == 8:
            if not ENABLE_NEW_TOKENIZER:
                input_ids, numeral_list, selector_idx = self.vocab.tokenize_transform_with_full_numeral_two_sentences(sentence1, sentence2, self.block_size)
            else:
                input_ids, numeral_list, selector_idx = self.vocab.tokenize_transform_with_full_numeral_two_sentences_revised(sentence1, sentence2, self.block_size)
    
            token_ids_list = input_ids
            fraction_list, exp_list = transform_number_arr_3(numeral_list, self.ne_exp_vocab, self.ne_exp_range_min, self.ne_exp_range_max)
            x = torch.tensor(token_ids_list, dtype=torch.long)
            fraction = torch.tensor(fraction_list, dtype=torch.float)
            exp = torch.tensor(exp_list, dtype=torch.long)
            selector = torch.tensor(selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, fraction, exp, selector, y
class AgeNumberComparison_FCGPTDataset(Dataset):

    def __init__(self, dataset_name, vocab, input_type=1, block_size=15, debug=False, extra_config=None):
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
        if not extra_config is None:
            self.ne_exp_range_min = extra_config["ne_exp_range_min"]
            self.ne_exp_range_max = extra_config["ne_exp_range_max"]
            self.ne_exp_vocab = extra_config["ne_exp_vocab"]
        print('data size: %d, vocab size: %d.' % (self.data_size, self.vocab_size))
    
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sentence1 = sample["question"]
        #sentence2 = str(sample["answer"])
        label = sample["label"]
        if self.input_type == 5:
            #full_sentence = "{} {} {}".format(self.vocab.cls_token, sentence1, self.vocab.eos_token)
            input_ids, numeral_list, selector_idx = self.vocab.tokenize_transform_with_full_numeral_one_sentences(sentence1, self.block_size)
            if not ENABLE_NEW_TOKENIZER:
                input_ids, numeral_list, selector_idx = self.vocab.tokenize_transform_with_full_numeral_one_sentences(sentence1, self.block_size)
            else:
                input_ids, numeral_list, selector_idx = self.vocab.tokenize_transform_with_full_numeral_one_sentences_revised(sentence1, self.block_size)
            token_ids_list = input_ids
            #pad_token_ids = self.vocab.pad_token_ids(token_ids_list, self.block_size)
            #pad_numeral_list = self.vocab.pad_array(numeral_list, self.block_size)
            #pad_selector_idx = self.vocab.pad_array(selector_idx, self.block_size)
            if not ENABLE_NEW_NUMBER_TRANSFORM:
                sign_list, fraction_list, exp_list = transform_number_arr(numeral_list)
            else:
                sign_list, fraction_list, exp_list = transform_number_arr_2(numeral_list)
            x = torch.tensor(token_ids_list, dtype=torch.long)
            sign = torch.tensor(sign_list, dtype=torch.long)
            fraction = torch.tensor(fraction_list, dtype=torch.float)
            exp = torch.tensor(exp_list, dtype=torch.float)
            selector = torch.tensor(selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, sign, fraction, exp, selector, y
        elif self.input_type == 8:
            #full_sentence = "{} {} {}".format(self.vocab.cls_token, sentence1, self.vocab.eos_token)
            if not ENABLE_NEW_TOKENIZER:
                input_ids, numeral_list, selector_idx = self.vocab.tokenize_transform_with_full_numeral_one_sentences(sentence1, self.block_size)
            else:
                input_ids, numeral_list, selector_idx = self.vocab.tokenize_transform_with_full_numeral_one_sentences_revised(sentence1, self.block_size)
            token_ids_list = input_ids
            fraction_list, exp_list = transform_number_arr_3(numeral_list, self.ne_exp_vocab, self.ne_exp_range_min, self.ne_exp_range_max)
            x = torch.tensor(token_ids_list, dtype=torch.long)
            fraction = torch.tensor(fraction_list, dtype=torch.float)
            exp = torch.tensor(exp_list, dtype=torch.long)
            selector = torch.tensor(selector_idx, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.long)
            return x, fraction, exp, selector, y
'''
Sample command: 

python -u fngpt1v5_cls_task_train.py ProbingTask_FCGPT 8 | tee ./logs/log_fn2gpt1v5_MME_train.txt
python -u fngpt1v5_cls_task_train.py GeneralNumberComparison_FCGPT 8 | tee ./logs/log_fn2gpt1v5_GNC_train.txt
python -u fngpt1v5_cls_task_train.py MWPAS_FCGPT 8 | tee ./logs/log_fn2gpt1v5_MWPAS_train.txt
'''

parser = argparse.ArgumentParser(description='General probing task training.')
parser.add_argument('task')
parser.add_argument('input_type', type=int)
parser.add_argument('--ne_exp_range_min', type=int, default=-8)
parser.add_argument('--ne_exp_range_max', type=int, default=12)
parser.add_argument('--debug', action="store_const", default=False, const=True,
                        help='If true, run preprocess in debug mode.')

args = parser.parse_args()
print("VARS:", vars(args))
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
elif input_type == 8:
    print("Training FNGPT1V5 on {}".format(task))
    ne_exp_config = {
        "ne_exp_range_min":args.ne_exp_range_min,
        "ne_exp_range_max":args.ne_exp_range_max,
        "ne_exp_vocab":construct_exp_vocab(args.ne_exp_range_min, args.ne_exp_range_max)
    }
    print(len(ne_exp_config["ne_exp_vocab"]))

if task == "MWPAS_FCGPT":
    block_size = 128
    output_class = 2
    max_epochs = 50
    vocab = MyOpenAIGPTTokenizer.from_pretrained('./openai-gpt')
    vocab.add_special_tokens({'pad_token': '[PAD]'})
    vocab.add_special_tokens({'cls_token': '[CLS]'})
    vocab.add_special_tokens({'sep_token': '[SEP]'})
    vocab.add_special_tokens({'eos_token': '[EOS]'})
    ckpt_path = "./models/FNGPT1V5_FT_MWPAS_ne_sigma_0.5/"
    file_name = "./data/mwpas_training_samples.csv"
    train_dataset = MWPAS_FCGPTDataset(file_name, vocab, input_type=input_type, block_size=block_size, debug=debug, extra_config=ne_exp_config)
    test_file_name = "./data/mwpas_test_samples.csv"
    test_dataset = MWPAS_FCGPTDataset(test_file_name, vocab, input_type=input_type, block_size=block_size, debug=debug, extra_config=ne_exp_config)
    #print(train_dataset[0])  
elif task == "GeneralNumberComparison_FCGPT":
    block_size = 128
    output_class = 2
    max_epochs = 50
    vocab = MyOpenAIGPTTokenizer.from_pretrained('./openai-gpt')
    vocab.add_special_tokens({'pad_token': '[PAD]'})
    vocab.add_special_tokens({'cls_token': '[CLS]'})
    vocab.add_special_tokens({'sep_token': '[SEP]'})
    vocab.add_special_tokens({'eos_token': '[EOS]'})
    ckpt_path = "./models/FNGPT1V5_FT_GNC_ne_sigma_0.5/"
    file_name = "./data/gnc_training_samples.csv"
    train_dataset = AgeNumberComparison_FCGPTDataset(file_name, vocab, input_type=input_type, block_size=block_size, debug=debug, extra_config=ne_exp_config)
    test_file_name = "./data/gnc_test_samples.csv"
    test_dataset = AgeNumberComparison_FCGPTDataset(test_file_name, vocab, input_type=input_type, block_size=block_size, debug=debug, extra_config=ne_exp_config)  
elif task == "ProbingTask_FCGPT":
    block_size = 128
    output_class = 2
    max_epochs = 50
    vocab = MyOpenAIGPTTokenizer.from_pretrained('./openai-gpt')
    vocab.add_special_tokens({'pad_token': '[PAD]'})
    vocab.add_special_tokens({'cls_token': '[CLS]'})
    vocab.add_special_tokens({'sep_token': '[SEP]'})
    vocab.add_special_tokens({'eos_token': '[EOS]'})
    ckpt_path = "./models/FNGPT1V5_FT_MME_ne_sigma_0.5/"
    file_name = "./data/mme_training_samples.csv"
    train_dataset = MWPAS_FCGPTDataset(file_name, vocab, input_type=input_type, block_size=block_size, debug=debug, extra_config=ne_exp_config)
    test_file_name = "./data/mme_test_samples.csv"
    test_dataset = MWPAS_FCGPTDataset(test_file_name, vocab, input_type=input_type, block_size=block_size, debug=debug, extra_config=ne_exp_config)
if debug:
    max_epochs = 1
print("max_epochs:", max_epochs, "ckpt_path:", ckpt_path)
## Floating number 
def get_numeral_embedding_config_4(dimension=128, exp_vocab_size=23):
    ## Assume total dimension is 128
    exp_dimension = int(8/32 * dimension)
    fraction_dimension = dimension - exp_dimension
    ne_config = {
        "ne_fraction":{
            "sigma": 0.5,
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
from mingpt.model import GPTClassification, GPTClassificationV3, GPTClassificationV4,GPTClassificationV5,GPTClassificationV6,GPTClassificationV7, GPTConfig
#from mygpt_model import MixedGPTClassificationV6

model_block_size = 512
mconf = GPTConfig(train_dataset.vocab_size, model_block_size,
                  n_layer=12, n_head=12, n_embd=768, head_type="Linear", 
                  ne_sigma=400, ne_rangemin=0, ne_rangemax=40000,
                  ne_config=get_numeral_embedding_config_4(int(64), len(ne_exp_config["ne_exp_vocab"])),
                  input_type=train_dataset.input_type, output_class=output_class,
                 layer_norm_epsilon=1e-05, afn='gelu')


## Input Type 4
if input_type == 8:
    model = GPTClassificationV7(mconf)
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
    
#model_path = "./models/FCGPT1V3_FT_ANC2/9_epochs.pkl"
#model_path = "./models/FN2GPT1V6_ne_sigma_6_MME/49_epochs.pkl"
#load_partial_weight_v1(model, model_path)
#load_partial_weight_v1(model, model_path, token_embedding_patch=False)

from mingpt.trainer import Trainer, GPTCV7Trainer, TrainerConfig

import time
# initialize a trainer instance and kick off training
start_time = time.time()
#ckpt_path = None
tconf = TrainerConfig(max_epochs=max_epochs, batch_size=batch_size, learning_rate=6.25e-5,
                      lr_decay=False, warmup_tokens=batch_size*20, final_tokens=2*len(train_dataset)*train_dataset.block_size,
                      num_workers=4, ckpt_path=ckpt_path)

if input_type == 8:
    trainer = GPTCV7Trainer(model, train_dataset, None, tconf)
    
trainer.train()
print("Training time", time.time() - start_time)


print("Train:")
test_batch_size = 8
if input_type == 8:
    give_exam_type_8_verbose(model, trainer, train_dataset, batch_size)

if task == "MWPAS_FCGPT" or task == "AgeNumberComparison_FCGPT" or task == "ProbingTask_FCGPT" or task == "GeneralNumberComparison_FCGPT":
    print("Test:")
    if input_type == 8:
        give_exam_type_8_verbose(model, trainer, test_dataset, batch_size)