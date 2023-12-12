import json
import os
import re
from typing import Optional, Tuple
import math
import numpy as np
import string
import time

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from transformers.models.bert.tokenization_bert import BasicTokenizer
from typing import Any, Dict, List, Optional, Tuple, Union, overload
import spacy
import en2an
spacy_nlp = spacy.load("en_core_web_sm")
logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"openai-gpt": "https://huggingface.co/openai-gpt/resolve/main/vocab.json"},
    "merges_file": {"openai-gpt": "https://huggingface.co/openai-gpt/resolve/main/merges.txt"},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai-gpt": 512,
}


def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus also does some whitespace standardization
    """
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = text.replace("―", "-")
    text = text.replace("…", "...")
    text = text.replace("´", "'")
    text = re.sub(r"""(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""", r" \1 ", text)
    text = re.sub(r"\s*\n\s*", " \n ", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()

def text_standardize_2(text):
    """
    fixes some issues the spacy tokenizer had on books corpus also does some whitespace standardization
    """
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = text.replace("―", "-")
    text = text.replace("…", "...")
    text = text.replace("´", "'")
    text = re.sub(r"""(=+|~+|!+|"+|;+|\?+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""", r" \1 ", text)
    text = re.sub(r"\s*\n\s*", " \n ", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()

def is_number(s):
    try:
        x = float(s)
        if math.isnan(x) or math.isinf(x):
            return False
        return True
    except ValueError:
        pass
 
    return False

def find_number(token, next_token):
    num_text = token
    percentFlag =  0
    if "%" in num_text:
        num_text = num_text.replace("%", "")
        percentFlag = 1
    if "%" == next_token:
        percentFlag = 1
    if "," in num_text or "+" in num_text:
        num_text = num_text.translate(str.maketrans('', '', ",+"))
    # print("num_text: ", num_text)
    try:
        number = float(num_text)
        if percentFlag == 1:
            number /= 100
        return number
    except Exception as e:
        # print("error: ", e)
        return None


def convert_sign_fraction_exp_to_str(sign, fraction, exp):
    if sign == 1:
        sign_t = ""
    else:
        sign_t = "-"
    return "{}{}e{}".format(sign_t, fraction, exp)
def convert_sign_fraction_exp_to_number(sign, fraction, exp):
    if sign == 1:
        sign_t = 1
    else:
        sign_t = -1
    exp = exp.item()
    fraction = fraction.item()
    return sign_t*fraction*10**exp
def check_inf(numeral_list):
    for number in numeral_list:
        if math.isinf(number) or math.isnan(number):
            return True
    return False
class MyOpenAIGPTTokenizer(PreTrainedTokenizer):
    """
    Construct a GPT Tokenizer. Based on Byte-Pair-Encoding with the following peculiarities:
    - lowercases all inputs,
    - uses :obj:`SpaCy` tokenizer and :obj:`ftfy` for pre-BPE tokenization if they are installed, fallback to BERT's
      :obj:`BasicTokenizer` if not.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.
    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        merges_file (:obj:`str`):
            Path to the merges file.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, merges_file, unk_token="<unk>", **kwargs):
        super().__init__(unk_token=unk_token, **kwargs)

        try:
            import ftfy
            from spacy.lang.en import English

            _nlp = English()
            self.nlp = _nlp.tokenizer
            #self.nlp = _nlp.Defaults.create_tokenizer(_nlp)
            self.fix_text = ftfy.fix_text
        except ImportError:
            print("Warning: Use BERT Basic Tokenizer.")
            logger.warning("ftfy or spacy is not installed using BERT BasicTokenizer instead of SpaCy & ftfy.")
            self.nlp = BasicTokenizer(do_lower_case=True)
            self.fix_text = None

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    @property
    def do_lower_case(self):
        return True

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        if word == "\n  </w>":
            word = "\n</w>"
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """ Tokenize a string. """
        split_tokens = []
        if self.fix_text is None:
            # Using BERT's BasicTokenizer
            text = self.nlp.tokenize(text)
            for token in text:
                split_tokens.extend([t for t in self.bpe(token).split(" ")])
        else:
            # Using SpaCy & ftfy (original tokenization process of OpenAI GPT)
            text = self.nlp(text_standardize(self.fix_text(text)))
            for token in text:
                split_tokens.extend([t for t in self.bpe(token.text.lower()).split(" ")])
        return split_tokens
    
    def tokenize_with_full_numeral(self, text):
        """ Tokenize a string. """
        split_tokens = []
        selector_idx = []
        numeral_list = []
        if self.fix_text is None:
            # Using BERT's BasicTokenizer
            text = self.nlp.tokenize(text)
        else:
            # Using SpaCy & ftfy (original tokenization process of OpenAI GPT)
            text = self.nlp(text_standardize(self.fix_text(text)))
            text = [token.text.lower() for token in text]
        for token in text:
            if is_number(token):
                selector_idx.append(1)
                numeral = float(token)
                numeral_list.append(numeral)
                split_tokens.append(self.unk_token)
            else:
                subtokens = [t for t in self.bpe(token).split(" ")]
                split_tokens.extend(subtokens)
                dummy = [0 for i in range(len(subtokens))]
                selector_idx.extend(dummy)
                numeral_list.extend(dummy)

        return split_tokens, numeral_list, selector_idx
    
    def tokenize_with_full_numeral_revised_v1(self, text):
        """ Tokenize a string. (enhanced with named entities)"""
        #import spacy
        #import en2an
        #spacy_nlp = spacy.load("en_core_web_sm")
        """ NER Type for numerals """
        ent_numerals = ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL"]
        original_text = text
        split_tokens = []
        selector_idx = []
        numeral_list = []
        """ Text Pre-Processing """
        # if self.fix_text is None:
        #     # Using BERT's BasicTokenizer
        #     text = self.nlp.tokenize(text)
        # else:
        #     # Using SpaCy & ftfy (original tokenization process of OpenAI GPT)
        #     text = self.nlp(text_standardize(self.fix_text(text)))
        
        # extract numeral-related entities
        ### TODO: handle units (e.g., money, date) and other descriptions (e.g., one month)
        spacy_text = spacy_nlp(text_standardize_2(text))
        # spacy_text = spacy_nlp("The stock decreases two percent.")
        # spacy_text = spacy_nlp("2,320")
        numbers_token_ids = []
        number_tokens = []
        time0 = time.time()

        for ent in spacy_text.ents:
            #print("ent: ", ent)
            if ent.label_ in ent_numerals:
                # PURE NUMBERS (mostly CARDINAL)
                # UPDATE: Handle "percent", "%" and "2,320" (commas):
                num_text = ent.text.lower()
                percentFlag = 0
                if "%" in ent.text or "percent" in ent.text:
                    num_text = num_text.replace("%", "")
                    num_text = num_text.replace("percent", "")
                    percentFlag = 1
                if "," in ent.text:
                    num_text = num_text.translate(str.maketrans('', '', ","))
                num_text = num_text.strip()
                # print("num_text: ", num_text)
                try:
                    #### to float
                    num_text_float = float(num_text)
                    if percentFlag:
                        number_tokens.append(num_text_float/100)
                    else:
                        number_tokens.append(num_text_float)
                    numbers_token_ids.append(list(range(ent.start, ent.end)))
                except Exception:
                    try: 
                        #### english to numbers
                        num_output = en2an.en2an(num_text, "smart")
                        if num_output:
                            # print("output: ", num_output, ent.label_)
                            numbers_token_ids.append(list(range(ent.start, ent.end)))
                            if percentFlag:
                                number_tokens.append(float(num_output)/100)
                            else:
                                number_tokens.append(float(num_output))
                            # print("output: ", type(num_output), num_output)
                    except Exception:
                        #### find all numbers
                        numbers = re.findall(r"\d+\.?\d*", num_text)
                        # print("numbers: ", numbers)
                        if len(numbers) > 0:
                            for eidx in range(ent.start, ent.end):
                                # print("spacy_text[eidx]", spacy_text[eidx].text.strip())
                                try:
                                    num_text_float = float(spacy_text[eidx].text.strip())
                                    # print("spacy_text[eidx]", num_text_float)
                                    if percentFlag:
                                        number_tokens.append(num_text_float/100)
                                    else:
                                        number_tokens.append(num_text_float)
                                    numbers_token_ids.append(list(range(eidx, eidx+1)))
                                    # print("spacy_text[eidx]", spacy_text[eidx], eval(spacy_text[eidx]))
                                except Exception:
                                    continue
                                    
                        else:
                            # print("can not covert to numbers: ", ent.text, ent.start, ent.end, ent.label_)
                            pass
        
        time1 = time.time()
        # print(number_tokens, numbers_token_ids)
        number_token_chunks = 0

        for token_id, token in enumerate(spacy_text):
            # print("token: ", token)
            if number_token_chunks == 0:
                flag = 0
                for i_idx, i in enumerate(numbers_token_ids):
                    if token_id in i:
                        selector_idx.append(1)
                        numeral = number_tokens[i_idx]
                        numeral_list.append(numeral)
                        flag = 1
                        split_tokens.append(self.unk_token)
                        number_token_chunks = len(i) - 1
                        break
                if flag == 0:
                    try:
                        # print("float(token.text.strip()): ", float(token.text.strip()))
                        if np.isnan(float(token.text.strip())) or np.isinf(float(token.text.strip())):
                            subtokens = [t for t in self.bpe(token.text).split(" ")]
                            split_tokens.extend(subtokens)
                            dummy = [0 for i in range(len(subtokens))]
                            selector_idx.extend(dummy)
                            numeral_list.extend(dummy)
                        else:
                            numeral_list.append(float(token.text.strip()))
                            selector_idx.append(1)
                            split_tokens.append(self.unk_token)
                    except Exception:
                        subtokens = [t for t in self.bpe(token.text).split(" ")]
                        split_tokens.extend(subtokens)
                        dummy = [0 for i in range(len(subtokens))]
                        selector_idx.extend(dummy)
                        numeral_list.extend(dummy)
                    number_token_chunks = 0
            else:
                number_token_chunks -= 1
        
        time2 = time.time()
        print("phase two: ", time2 - time0, time1 - time0, time2 - time1)
        if check_inf(numeral_list):
            print("\n --- INF Warning ---")
            print(" -- Original Text --")
            print(original_text)
            print(" -- tokens --")
            print(split_tokens)
            print(" -- numeral list --")
            print(numeral_list)
            print(" -- selector -- ")
            print(selector_idx)
            print(" --- End of INF Warning ---\n")
        
        # print("number_tokens: ", number_tokens)
        return split_tokens, numeral_list, selector_idx

    def tokenize_with_full_numeral_revised(self, text):
        """ Tokenize a string. (enhanced with named entities)"""
        """ NER Type for numerals """
        # ent_numerals = ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL"]
        original_text = text
        split_tokens = []
        selector_idx = []
        numeral_list = []
        """ Text Pre-Processing """        
        if self.fix_text is None:
            # Using BERT's BasicTokenizer
            text = self.nlp.tokenize(text)
        else:
            # Using SpaCy & ftfy (original tokenization process of OpenAI GPT)
            text = self.nlp(text_standardize_2(self.fix_text(text)))
            # Handle % and commas ([+-]?(?:\d+)((\d{1,3})*([\,\ ]\d{3})*)(\.\d+)?)
            text = [token.text.lower().strip() for token in text if token.text.lower().strip()]
        spacy_text = text
        #spacy_text = spacy_nlp(text_standardize_2(text.lower()))
        """ Extract Numeral-Related Entities """
        ### TODO: handle units (e.g., money, date) and other descriptions (e.g., one month, one point two (i.e., 1.2))
        for tokenid, token in enumerate(spacy_text):
            # print("token: ", token)
            ### Hnadle percent/%
            if "%" != token:
                # comma
                tokens = []
                if "," in token and "," != token:
                    if 4 * len(re.findall(",", token)) + 1 > len(token.strip()):
                        tokens = token.split(",")
                # percent
                if tokenid + 1 < len(spacy_text):
                    next_token = spacy_text[tokenid + 1]
                else:
                    next_token = ""
                
                if len(tokens) == 0:     
                    number_results = find_number(token, next_token)
                    # print("number_results: ", number_results)
                    ### TODO: handle str "percent" in the next token
                    if number_results is not None and np.isnan(number_results) == False and np.isinf(number_results) == False:
                        selector_idx.append(1)
                        numeral_list.append(number_results)
                        split_tokens.append(self.unk_token)
                    else:
                        #subtokens = [t for t in self.bpe(token.text).split(" ")]
                        subtokens = [t for t in self.bpe(token).split(" ")]
                        split_tokens.extend(subtokens)
                        dummy = [0 for i in range(len(subtokens))]
                        selector_idx.extend(dummy)
                        numeral_list.extend(dummy)
                else:
                    # print("comma tokens: ", tokens)
                    for token in tokens:
                        if token.strip() != "":
                            number_results = find_number(token, next_token)
                            # print("number_results: ", number_results)
                            ### TODO: handle str "percent" in the next token
                            if number_results is not None and np.isnan(number_results) == False and np.isinf(number_results) == False:
                                selector_idx.append(1)
                                numeral_list.append(number_results)
                                split_tokens.append(self.unk_token)
                            else:
                                #subtokens = [t for t in self.bpe(token.text).split(" ")]
                                subtokens = [t for t in self.bpe(token).split(" ")]
                                split_tokens.extend(subtokens)
                                dummy = [0 for i in range(len(subtokens))]
                                selector_idx.extend(dummy)
                                numeral_list.extend(dummy)
        # print("split_tokens, numeral_list, selector_idx: ", split_tokens, numeral_list, selector_idx)
        # assert len(split_tokens) == len(numeral_list) == len(selector_idx)
        return split_tokens, numeral_list, selector_idx

    def tokenize_transform_with_full_numeral(self, text):
        """ Tokenize a string. """
        split_tokens, numeral_list, selector_idx = self.tokenize_with_full_numeral(text)
        input_ids = self.convert_tokens_to_ids(split_tokens)
        return input_ids, numeral_list, selector_idx
    def tokenize_transform_with_full_numeral_revised(self, text):
        """ Tokenize a string. """
        split_tokens, numeral_list, selector_idx = self.tokenize_with_full_numeral_revised(text)
        input_ids = self.convert_tokens_to_ids(split_tokens)
        return input_ids, numeral_list, selector_idx
    def pad_token_ids(self, token_ids_list, fixed_length, pad_token_id):
        encoded = token_ids_list[:fixed_length]
        while len(encoded) < fixed_length:
            encoded.append(pad_token_id)
        return encoded
    def pad_array(self, token_ids_list, fixed_length):
        encoded = token_ids_list[:fixed_length]
        while len(encoded) < fixed_length:
            encoded.append(0)
        return encoded
    def tokenize_transform_with_full_numeral_one_sentences(self, text1, block_size, no_extra_id=False):
        """ Assume vocab has defined pad_token, cls_token, sep_token, eos_token """
        pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        cls_token_id = self.convert_tokens_to_ids(self.cls_token)
        sep_token_id = self.convert_tokens_to_ids(self.sep_token)
        eos_token_id = self.convert_tokens_to_ids(self.eos_token)
        input_ids_1, numeral_list_1, selector_idx_1 = self.tokenize_transform_with_full_numeral(text1)
        if no_extra_id:
            input_ids = input_ids_1
            numeral_list = numeral_list_1
            selector_idx = selector_idx_1
        else:
            input_ids = [cls_token_id] + input_ids_1 + [eos_token_id]
            numeral_list = [0] + numeral_list_1 + [0]
            selector_idx = [0] + selector_idx_1 + [0]
        input_ids = self.pad_token_ids(input_ids, block_size, pad_token_id)
        numeral_list = self.pad_array(numeral_list, block_size)
        selector_idx = self.pad_array(selector_idx, block_size)
        return input_ids, numeral_list, selector_idx
    def tokenize_transform_with_full_numeral_one_sentences_plus_mask(self, text1, block_size):
        """ Assume vocab has defined pad_token, cls_token, sep_token, eos_token """
        pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        cls_token_id = self.convert_tokens_to_ids(self.cls_token)
        sep_token_id = self.convert_tokens_to_ids(self.sep_token)
        eos_token_id = self.convert_tokens_to_ids(self.eos_token)
        input_ids_1, numeral_list_1, selector_idx_1 = self.tokenize_transform_with_full_numeral(text1)
        input_ids = [cls_token_id] + input_ids_1 + [eos_token_id]
        numeral_list = [0] + numeral_list_1 + [0]
        selector_idx = [0] + selector_idx_1 + [0]
        attention_masks = [1] * len(input_ids)
        input_ids = self.pad_token_ids(input_ids, block_size, pad_token_id)
        numeral_list = self.pad_array(numeral_list, block_size)
        selector_idx = self.pad_array(selector_idx, block_size)
        attention_mask = self.pad_array(attention_masks, block_size)
        return input_ids, numeral_list, selector_idx, attention_mask
    def tokenize_transform_with_full_numeral_one_sentences_revised(self, text1, block_size, no_extra_id=False):
        """ Assume vocab has defined pad_token, cls_token, sep_token, eos_token """
        pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        cls_token_id = self.convert_tokens_to_ids(self.cls_token)
        sep_token_id = self.convert_tokens_to_ids(self.sep_token)
        eos_token_id = self.convert_tokens_to_ids(self.eos_token)
        input_ids_1, numeral_list_1, selector_idx_1 = self.tokenize_transform_with_full_numeral_revised(text1)
        if no_extra_id:
            input_ids = input_ids_1
            numeral_list = numeral_list_1
            selector_idx = selector_idx_1
        else:
            input_ids = [cls_token_id] + input_ids_1 + [eos_token_id]
            numeral_list = [0] + numeral_list_1 + [0]
            selector_idx = [0] + selector_idx_1 + [0]
        input_ids = self.pad_token_ids(input_ids, block_size, pad_token_id)
        numeral_list = self.pad_array(numeral_list, block_size)
        selector_idx = self.pad_array(selector_idx, block_size)
        return input_ids, numeral_list, selector_idx
    def tokenize_transform_with_full_numeral_two_sentences(self, text1, text2, block_size, mask_token_enable=False):
        """ Assume vocab has defined pad_token, cls_token, sep_token, eos_token """
        pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        cls_token_id = self.convert_tokens_to_ids(self.cls_token)
        if not mask_token_enable:
            sep_token_id = self.convert_tokens_to_ids(self.sep_token)
        else:
            sep_token_id = self.convert_tokens_to_ids(self.mask_token)
        eos_token_id = self.convert_tokens_to_ids(self.eos_token)
        input_ids_1, numeral_list_1, selector_idx_1 = self.tokenize_transform_with_full_numeral(text1)
        input_ids_2, numeral_list_2, selector_idx_2 = self.tokenize_transform_with_full_numeral(text2)
        input_ids = [cls_token_id] + input_ids_1 + [sep_token_id] + input_ids_2 + [eos_token_id]
        numeral_list = [0] + numeral_list_1 + [0] + numeral_list_2 + [0]
        selector_idx = [0] + selector_idx_1 + [0] + selector_idx_2 + [0]
        input_ids = self.pad_token_ids(input_ids, block_size, pad_token_id)
        numeral_list = self.pad_array(numeral_list, block_size)
        selector_idx = self.pad_array(selector_idx, block_size)
        return input_ids, numeral_list, selector_idx
    def tokenize_transform_with_full_numeral_fcgptv6_two_sentences(self, text1, text2, block_size):
        """ Assume vocab has defined pad_token, cls_token, sep_token, eos_token """
        pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        cls_token_id = self.convert_tokens_to_ids(self.cls_token)
        sep_token_id = self.convert_tokens_to_ids(self.sep_token)
        eos_token_id = self.convert_tokens_to_ids(self.eos_token)
        input_ids_1, numeral_list_1, selector_idx_1 = self.tokenize_transform_with_full_numeral(text1)
        input_ids_2, numeral_list_2, selector_idx_2 = self.tokenize_transform_with_full_numeral(text2)
        input_ids = input_ids_1 + input_ids_2 
        numeral_list = numeral_list_1 + numeral_list_2
        selector_idx = selector_idx_1 + selector_idx_2 
        loss_mask = [0]*len(input_ids_1) + [1]*len(input_ids_2)
        input_ids = self.pad_token_ids(input_ids, block_size, pad_token_id)
        numeral_list = self.pad_array(numeral_list, block_size)
        selector_idx = self.pad_array(selector_idx, block_size)
        loss_mask = self.pad_array(loss_mask, block_size)
        return input_ids, numeral_list, selector_idx, loss_mask
    def tokenize_transform_with_full_numeral_two_sentences_plus_mask(self, text1, text2, block_size):
        """ Assume vocab has defined pad_token, cls_token, sep_token, eos_token """
        pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        cls_token_id = self.convert_tokens_to_ids(self.cls_token)
        sep_token_id = self.convert_tokens_to_ids(self.sep_token)
        eos_token_id = self.convert_tokens_to_ids(self.eos_token)
        input_ids_1, numeral_list_1, selector_idx_1 = self.tokenize_transform_with_full_numeral(text1)
        input_ids_2, numeral_list_2, selector_idx_2 = self.tokenize_transform_with_full_numeral(text2)
        input_ids = [cls_token_id] + input_ids_1 + [sep_token_id] + input_ids_2 + [eos_token_id]
        numeral_list = [0] + numeral_list_1 + [0] + numeral_list_2 + [0]
        selector_idx = [0] + selector_idx_1 + [0] + selector_idx_2 + [0]
        attention_masks = [1] * len(input_ids)
        input_ids = self.pad_token_ids(input_ids, block_size, pad_token_id)
        numeral_list = self.pad_array(numeral_list, block_size)
        selector_idx = self.pad_array(selector_idx, block_size)
        attention_mask = self.pad_array(attention_masks, block_size)
        return input_ids, numeral_list, selector_idx, attention_mask
    def tokenize_transform_with_full_numeral_two_sentences_revised(self, text1, text2, block_size, mask_token_enable=False):
        """ Assume vocab has defined pad_token, cls_token, sep_token, eos_token """
        pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        cls_token_id = self.convert_tokens_to_ids(self.cls_token)
        if not mask_token_enable:
            sep_token_id = self.convert_tokens_to_ids(self.sep_token)
        else:
            sep_token_id = self.convert_tokens_to_ids(self.mask_token)
        eos_token_id = self.convert_tokens_to_ids(self.eos_token)
        input_ids_1, numeral_list_1, selector_idx_1 = self.tokenize_transform_with_full_numeral_revised(text1)
        input_ids_2, numeral_list_2, selector_idx_2 = self.tokenize_transform_with_full_numeral_revised(text2)
        input_ids = [cls_token_id] + input_ids_1 + [sep_token_id] + input_ids_2 + [eos_token_id]
        numeral_list = [0] + numeral_list_1 + [0] + numeral_list_2 + [0]
        selector_idx = [0] + selector_idx_1 + [0] + selector_idx_2 + [0]
        input_ids = self.pad_token_ids(input_ids, block_size, pad_token_id)
        numeral_list = self.pad_array(numeral_list, block_size)
        selector_idx = self.pad_array(selector_idx, block_size)
        return input_ids, numeral_list, selector_idx
    def tokenize_transform_with_full_numeral_three_sentences_revised(self, text1, text2, text3, block_size, no_end_token=False):
        """ Assume vocab has defined pad_token, cls_token, sep_token, eos_token """
        pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        cls_token_id = self.convert_tokens_to_ids(self.cls_token)
        maks_token_id = self.convert_tokens_to_ids(self.mask_token)
        eos_token_id = self.convert_tokens_to_ids(self.eos_token)
        input_ids_1, numeral_list_1, selector_idx_1 = self.tokenize_transform_with_full_numeral_revised(text1)
        input_ids_2, numeral_list_2, selector_idx_2 = self.tokenize_transform_with_full_numeral_revised(text2)
        input_ids_3, numeral_list_3, selector_idx_3 = self.tokenize_transform_with_full_numeral_revised(text3)
        if no_end_token:
            input_ids = [cls_token_id] + input_ids_1 + [maks_token_id] + input_ids_2 + [eos_token_id, cls_token_id] + input_ids_3
            numeral_list = [0] + numeral_list_1 + [0] + numeral_list_2 + [0, 0] + numeral_list_3
            selector_idx = [0] + selector_idx_1 + [0] + selector_idx_2 + [0, 0] + selector_idx_3
            input_seq_length = len(input_ids)
        else:
            input_ids = [cls_token_id] + input_ids_1 + [maks_token_id] + input_ids_2 + [eos_token_id, cls_token_id] + input_ids_3 + [eos_token_id]
            numeral_list = [0] + numeral_list_1 + [0] + numeral_list_2 + [0, 0] + numeral_list_3 + [0]
            selector_idx = [0] + selector_idx_1 + [0] + selector_idx_2 + [0, 0] + selector_idx_3 + [0]
        input_ids = self.pad_token_ids(input_ids, block_size, pad_token_id)
        numeral_list = self.pad_array(numeral_list, block_size)
        selector_idx = self.pad_array(selector_idx, block_size)
        if no_end_token:
            return input_ids, numeral_list, selector_idx, input_seq_length
        else:
            return input_ids, numeral_list, selector_idx
    def tokenize_transform_with_full_numeral_three_sentences(self, text1, text2, text3, block_size, no_end_token=False):
        """ Assume vocab has defined pad_token, cls_token, sep_token, eos_token """
        pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        cls_token_id = self.convert_tokens_to_ids(self.cls_token)
        maks_token_id = self.convert_tokens_to_ids(self.mask_token)
        eos_token_id = self.convert_tokens_to_ids(self.eos_token)
        input_ids_1, numeral_list_1, selector_idx_1 = self.tokenize_transform_with_full_numeral(text1)
        input_ids_2, numeral_list_2, selector_idx_2 = self.tokenize_transform_with_full_numeral(text2)
        input_ids_3, numeral_list_3, selector_idx_3 = self.tokenize_transform_with_full_numeral(text3)
        if no_end_token:
            input_ids = [cls_token_id] + input_ids_1 + [maks_token_id] + input_ids_2 + [eos_token_id, cls_token_id] + input_ids_3
            numeral_list = [0] + numeral_list_1 + [0] + numeral_list_2 + [0, 0] + numeral_list_3
            selector_idx = [0] + selector_idx_1 + [0] + selector_idx_2 + [0, 0] + selector_idx_3
            input_seq_length = len(input_ids)
        else:
            input_ids = [cls_token_id] + input_ids_1 + [maks_token_id] + input_ids_2 + [eos_token_id, cls_token_id] + input_ids_3 + [eos_token_id]
            numeral_list = [0] + numeral_list_1 + [0] + numeral_list_2 + [0, 0] + numeral_list_3 + [0]
            selector_idx = [0] + selector_idx_1 + [0] + selector_idx_2 + [0, 0] + selector_idx_3 + [0]
        input_ids = self.pad_token_ids(input_ids, block_size, pad_token_id)
        numeral_list = self.pad_array(numeral_list, block_size)
        selector_idx = self.pad_array(selector_idx, block_size)
        if no_end_token:
            return input_ids, numeral_list, selector_idx, input_seq_length
        else:
            return input_ids, numeral_list, selector_idx
    def tokenize_transform_with_full_numeral_two_sentences_v2(self, text1, text2, block_size):
        """ Assume vocab has defined pad_token, cls_token, sep_token, eos_token """
        pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        cls_token_id = self.convert_tokens_to_ids(self.cls_token)
        maks_token_id = self.convert_tokens_to_ids(self.mask_token)
        eos_token_id = self.convert_tokens_to_ids(self.eos_token)
        input_ids_1, numeral_list_1, selector_idx_1 = self.tokenize_transform_with_full_numeral(text1)
        input_ids_2, numeral_list_2, selector_idx_2 = self.tokenize_transform_with_full_numeral(text2)
        input_ids = input_ids_1 + [eos_token_id, cls_token_id] + input_ids_2
        numeral_list = numeral_list_1 + [0, 0] + numeral_list_2
        selector_idx = selector_idx_1 + [0, 0] + selector_idx_2
        input_seq_length = len(input_ids)
        input_ids = self.pad_token_ids(input_ids, block_size, pad_token_id)
        numeral_list = self.pad_array(numeral_list, block_size)
        selector_idx = self.pad_array(selector_idx, block_size)
        return input_ids, numeral_list, selector_idx, input_seq_length        
    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an id in a token (BPE) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = "".join(tokens).replace("</w>", " ").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!".format(merge_file)
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file
    def decode_with_numeral(
        self,
        token_ids: List[int],
        sign_list: List[int],
        fraction_list: List[float],
        exp_list: List[float],
        selector_list: List[int],
        sci_mode: bool = False,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True,
        **kwargs
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        #print("filtered_tokens length {} : {}".format(len(filtered_tokens), filtered_tokens))
        #print("sign length: {}, fraction length: {}, exp length: {}, selector length: {}".format(len(sign_list), len(fraction_list), len(exp_list), len(selector_list))) 
        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        #print(self.added_tokens_encoder)
        for index, token in enumerate(filtered_tokens):
            selector = selector_list[index]
            #print(sub_texts, current_sub_text)
            if selector == 1:
                sign = sign_list[index]
                fraction = fraction_list[index]
                exp = exp_list[index]
                if sci_mode:
                    current_sub_text.append(str(convert_sign_fraction_exp_to_str(sign, fraction, exp))+"</w>")
                else:
                    current_sub_text.append(str(convert_sign_fraction_exp_to_number(sign, fraction, exp))+"</w>")
                
            else:
                if skip_special_tokens and token in self.all_special_ids:
                    continue
                if token in self.added_tokens_encoder:
                    if current_sub_text:
                        sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                        current_sub_text = []
                    sub_texts.append(token)
                else:
                    current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text
    def decode_with_numeral_2(
        self,
        token_ids: List[int],
        numeral_list: List[str],
        selector_list: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True,
        **kwargs
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        #print("filtered_tokens length {} : {}".format(len(filtered_tokens), filtered_tokens))
        #print("sign length: {}, fraction length: {}, exp length: {}, selector length: {}".format(len(sign_list), len(fraction_list), len(exp_list), len(selector_list))) 
        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        #print(self.added_tokens_encoder)
        for index, token in enumerate(filtered_tokens):
            selector = selector_list[index]
            #print(sub_texts, current_sub_text)
            if selector == 1:
                numeral = numeral_list[index]
                current_sub_text.append(str(numeral)+"</w>")
            else:
                if skip_special_tokens and token in self.all_special_ids:
                    continue
                if token in self.added_tokens_encoder:
                    if current_sub_text:
                        sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                        current_sub_text = []
                    sub_texts.append(token)
                else:
                    current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    