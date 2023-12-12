"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    output_class = 2
    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    
class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
         
        self.block_size = config.block_size
        self.apply(self._init_weights)

        #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        print("number of parameters: ", sum(p.numel() for p in self.parameters()))
        
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, )
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif isinstance(m, NeuralAccumulatorCell) or isinstance(m, NeuralArithmeticLogicUnitCell):
                    no_decay.add(fpn)
                    

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
class NumeralEmbeddingLayer(nn.Module):
    def __init__(self, sigma, rangemin, rangemax, dimension):
        super().__init__()    
        self.sigma = sigma
        self.rangemin = rangemin
        self.rangemax = rangemax
        self.dimension = dimension
        self.prototype = self.create_prototype()
    def getParamsDescription(self):
        describe = "sigma={}, rangemin={}, rangemax={}, dimension={}".format(self.sigma, self.rangemin, self.rangemax, self.dimension)
        return describe
    def __repr__(self):
        class_name = "{}({})".format(self.__class__.__name__, self.getParamsDescription())
        return class_name
    def create_prototype(self):
        prototype = []
        step = (self.rangemax - self.rangemin) / (self.dimension-1)
        for i in range(self.dimension):
            q = step * i + self.rangemin
            prototype.append(q)
        prototype = torch.FloatTensor(prototype)
        return prototype
    
    def forward(self, vector):
        ## vector: [batch size]
        ## prototype: [dimension]
        prototype = self.prototype
        prototype = prototype.to(vector.device)
        batch_size = vector.shape[0]
        dimension = prototype.shape[0]
        vector_expand = vector.unsqueeze(1).repeat(1, dimension)
        prototype_expand = prototype.unsqueeze(0).repeat(batch_size, 1)
        square_diff = (vector_expand - prototype_expand) ** 2
        #print(square_diff)
        embedding = torch.exp(-square_diff / (self.sigma ** 2))
        return embedding
    
class NumeralEmbeddingLayerV2(NumeralEmbeddingLayer):
    def forward(self, vector):
        ## vector: [batch size, sequence size]
        ## prototype: [dimension]
        
        prototype = self.prototype
        prototype = prototype.to(vector.device)
        #print(vector.shape, prototype.shape)
        
        ## torch.Size([64, 13]) torch.Size([128])
        batch_size = vector.shape[0]
        seq_size = vector.shape[1]
        dimension = prototype.shape[0]
        vector_expand = vector.unsqueeze(2).repeat(1, 1, dimension)
        prototype_expand = prototype.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_size, 1)
        square_diff = (vector_expand - prototype_expand) ** 2
        #print(square_diff)
        embedding = torch.exp(-square_diff / (self.sigma ** 2))
        return embedding

    
class NumeralEmbeddingLayerV4(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        fraction_config = config["ne_fraction"]
        fraction_dimension = fraction_config["dimension"]
        fraction_rangemin = fraction_config["rangemin"]
        fraction_rangemax = fraction_config["rangemax"]
        fraction_sigma = fraction_config["sigma"]
        self.fraction_emb = NumeralEmbeddingLayerV2(fraction_sigma, fraction_rangemin, fraction_rangemax, fraction_dimension)
        
        exp_config = config["ne_exp"]
        exp_dimension = exp_config["dimension"]
        exp_vocab_size = exp_config["vocab_size"]
        self.exp_emb = nn.Embedding(exp_vocab_size, exp_dimension)
        
    def forward(self, fraction, exp):
        fraction_embedding = self.fraction_emb(fraction)
        exp_embedding = self.exp_emb(exp)
        numeral_embedding = torch.cat((fraction_embedding, exp_embedding), dim=2)
        return numeral_embedding
    

class GPTClassification(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.output_class, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)

        #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        print("number of parameters: ", sum(p.numel() for p in self.parameters()))
        
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, )
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif isinstance(m, NeuralAccumulatorCell) or isinstance(m, NeuralArithmeticLogicUnitCell):
                    no_decay.add(fpn)
                    

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        predict = logits[:,-1,:]
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(predict.view(-1, predict.size(-1)), targets.view(-1))
        real_predict = F.softmax(predict, dim=1)
        return real_predict, loss
    

def compute_masked_cross_entropy_loss_2(logits, target, mask):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        mask: A mask containing values of 0/1 to indicate which words are taken into consideration.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    losses = losses * mask.float()
    loss = losses.mean()
    return loss
    
def get_square_diff(vector, prototype, sigma):
    #prototype = self.prototype
    prototype = prototype.to(vector.device)
    #print(vector.shape, prototype.shape)

    ## torch.Size([64, 13]) torch.Size([128])
    batch_size = vector.shape[0]
    seq_size = vector.shape[1]
    dimension = prototype.shape[0]
    vector_expand = vector.unsqueeze(2).repeat(1, 1, dimension)
    prototype_expand = prototype.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_size, 1)
    square_diff = (vector_expand - prototype_expand) ** 2 / (sigma ** 2)
    return square_diff


def compute_masked_mse_loss_2(output, target, mask):
    diff2 = (torch.flatten(output) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
    result = torch.mean(diff2)
    return result


class GPTNLMv10(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, int(config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.fusion_layer = FusionLayerV2(int(config.n_embd), int(config.ne_config["total_dimension"]), config.n_embd)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.selector_head = nn.Linear(config.n_embd, 2, bias=False)
        self.token_head = nn.Linear(config.n_embd, config.output_class, bias=False)
        #self.numeral_linear = nn.Linear(config.n_embd, int(config.ne_config["total_dimension"]), bias=False)
        self.numeral_fraction_linear = nn.Linear(config.n_embd, 1, bias=False)
        self.numeral_exp_linear = nn.Linear(config.n_embd, config.ne_config["ne_exp"]["vocab_size"], bias=False)
        self.ne_emb = NumeralEmbeddingLayerV4(config=config.ne_config)
        self.block_size = config.block_size
        self.apply(self._init_weights)

        #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        print("number of parameters: ", sum(p.numel() for p in self.parameters()))
        
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, )
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif isinstance(m, NeuralAccumulatorCell) or isinstance(m, NeuralArithmeticLogicUnitCell):
                    no_decay.add(fpn)
                    

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, fraction, exp, selector, token_targets=None, 
                numeral_fraction_targets=None,
                numeral_exp_targets=None,
                selector_targets=None,
                loss_mask=None
                ):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        numeral_embeddings = self.ne_emb(fraction, exp)
        combined_token_embeddings = self.fusion_layer(token_embeddings, numeral_embeddings, selector)
        #print("combined_token_embeddings:", combined_token_embeddings.device, "position_embeddings:", position_embeddings.device)
        #print(combined_token_embeddings)
        x = self.drop(combined_token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        token_logits = self.token_head(x)
        #numeral_logits = self.numeral_head(x)
        #numeral_transformed = self.numeral_linear(x)
        #numeral_output_sign_emb = numeral
        numeral_fraction_logits = self.numeral_fraction_linear(x)
        
        #numeral_fraction_logits = F.softmax(self.numeral_fraction_linear(x), dim=2)
        numeral_exp_logits = self.numeral_exp_linear(x)
        #numeral_exp_logits = F.softmax(self.numeral_exp_linear(x), dim=2)
        selector_logits = self.selector_head(x)
        loss = None
        return_package = (token_logits, numeral_fraction_logits, numeral_exp_logits, selector_logits)
        if token_targets is not None and numeral_fraction_targets is not None and numeral_exp_targets is not None and selector_targets is not None:
            token_loss = compute_masked_cross_entropy_loss_2(token_logits, token_targets, (1-selector_targets.float()))
            selector_loss = F.cross_entropy(selector_logits.view(-1, selector_logits.size(-1)), selector_targets.view(-1))
            numeral_fraction_loss = compute_masked_mse_loss_2(numeral_fraction_logits, numeral_fraction_targets, selector_targets)
            numeral_exp_loss = compute_masked_cross_entropy_loss_2(numeral_exp_logits, numeral_exp_targets, selector_targets)
            token_hvalue = 1
            selector_hvalue = 1
            numeral_fraction_hvalue = 1
            numeral_exp_hvalue = 1
            loss = token_hvalue * token_loss + selector_hvalue * selector_loss + numeral_fraction_hvalue * numeral_fraction_loss + numeral_exp_hvalue * numeral_exp_loss
            return_package = return_package + (token_loss, selector_loss, numeral_fraction_loss, numeral_exp_loss)
        return return_package, loss 
    
class GPTClassificationV7(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, int(config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.fusion_layer = FusionLayerV2(int(config.n_embd), int(config.ne_config["total_dimension"]), config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.output_class, bias=False)
        self.ne_emb = NumeralEmbeddingLayerV4(config=config.ne_config)
        self.block_size = config.block_size
        self.apply(self._init_weights)

        #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        print("number of parameters: ", sum(p.numel() for p in self.parameters()))
    
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, )
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif isinstance(m, NeuralAccumulatorCell) or isinstance(m, NeuralArithmeticLogicUnitCell):
                    no_decay.add(fpn)
                    

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
    
    def forward(self, idx, fraction, exp, selector, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        #print(idx.shape, sign.shape, fraction.shape, exp.shape, selector.shape)
        # forward the GPT model
        #print("idx:", idx.shape)
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        #print("token_embeddings:",token_embeddings.shape)
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        #print("position_embeddings:",position_embeddings.shape)
        numeral_embeddings = self.ne_emb(fraction, exp)
        #print(token_embeddings.shape, numeral_embeddings.shape, selector.shape)
        #token_embeddings = token_embeddings.unsqueeze(2)
        #numeral_embeddings = numeral_embeddings.unsqueeze(2)
        #concat_token_embeddings = torch.cat((token_embeddings, numeral_embeddings), dim=2)
        #selector_expand = selector.unsqueeze(2).repeat(1,1, token_embeddings.shape[2])
        #combined_token_embeddings = token_embeddings * (1-selector_expand) + numeral_embeddings * selector_expand
        combined_token_embeddings = self.fusion_layer(token_embeddings, numeral_embeddings, selector)
        x = self.drop(combined_token_embeddings + position_embeddings)
        #print("x1:", x.shape)
        x = self.blocks(x)
        #print("x2:", x.shape)
        x = self.ln_f(x)
        #print("x3:", x.shape)
        #x = torch.cat((x[:,-1,:],numeral_embeddings), dim=1)
        #x = x[:,-1,:] + numeral_embeddings
        x = x[:,-1,:]
        logits = self.head(x)
        predict = logits
        #predict = logits[:,-1,:]
        #print("logits:", logits.shape)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(predict.view(-1, predict.size(-1)), targets.view(-1))
        real_predict = F.softmax(predict, dim=1)
        return real_predict, loss   
    