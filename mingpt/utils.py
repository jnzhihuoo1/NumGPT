import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x
class mGPTSampler:
    def __init__(self,  model, tokenizer, block_size=512):
        self.model = model
        self.block_size = block_size
        self.tokenizer = tokenizer
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
    def generate(self, context, steps=20, temperature=1.0, sample=False, top_k=None, crop_head=False):
        tokenized = self.tokenizer(context)
        input_ids = tokenized["input_ids"]
        input_length = len(input_ids)
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        model = self.model
        block_size = self.block_size
        model.eval()
        for k in range(steps):
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
            logits, _ = model(x_cond)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)
        x = x.squeeze(0)
        if crop_head:
            x = x[input_length:]
        text = self.tokenizer.decode(x)
        return text
    
class GPTSampler:
    def __init__(self,  model, tokenizer, block_size=512):
        self.model = model
        self.block_size = block_size
        self.tokenizer = tokenizer
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
    def generate(self, context, steps=20, temperature=1.0, sample=False, top_k=None):
        tokenized = self.tokenizer(context)
        input_ids = tokenized["input_ids"]
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        model = self.model
        block_size = self.block_size
        model.eval()
        for k in range(steps):
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
            output = model(x_cond)
            logits = output.logits
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)
        x = x.squeeze(0)
        text = self.tokenizer.decode(x)
        return text
    
    
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

def test_context(context, sampler):
    print("--- Begin testing --- ")
    print("--- Context:")
    print(context)
    start_time = time.time()
    results = sampler.generate(context, steps=200, temperature=1, sample=False, top_k=None)
    end_time = time.time()
    print("\n --- Results (steps = 200, temperature = 1, sample = False, top_k = None, time = {}):".format(end_time - start_time))
    print(results)
    start_time = time.time()
    results = sampler.generate(context, steps=200, temperature=2, sample=True, top_k=5)
    end_time = time.time()
    print("\n --- Results (steps = 200, temperature = 2, sample = True, top_k = 5, time = {}):".format(end_time - start_time))
    print(results)
    start_time = time.time()
    results = sampler.generate(context, steps=200, temperature=3, sample=True, top_k=15)
    end_time = time.time()
    print("\n --- Results (steps = 200, temperature = 3, sample = True, top_k = 15, time = {}):".format(end_time - start_time))
    print(results)
    
    print("\n --- End of testing")
    
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
        },
        "total_dimension": dimension
    }
    return ne_config



### test generation
def test_generation(context, sampler):
    print("--- Begin testing --- ")
    print("--- Context:")
    print(context)
    start_time = time.time()
    results = sampler.generate(context, steps=20, temperature=1, sample=False, top_k=None)
    end_time = time.time()
    print("\n --- Results (steps = 20, temperature = 1, sample = False, top_k = None, time = {}):".format(end_time - start_time))
    print(results)
    print("\n --- End of testing")