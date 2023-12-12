# NumGPT

This repository contains the code and information of datasets on reproducing the experiments in the paper "NumGPT: Improving numeracy ability of generative pre-trained models". 

It contains the code for generating the GPT1 / NumGPT training dataset, pretraining the GPT1 and NumGPT models, finetuning the models on the downstream train set, and evaluating them. Datasets information is also provided.

## Prerequisites

### 1. Packages

You need to install the following packages to make it run:

```bash
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install transformers==4.3.3
pip install spacy==2.3.0 ftfy==4.4.3 en2an
python -m spacy download en
```

### 2. Datasets

You may need to download the following dataset. For synthetic datasets, you can check this [link](https://drive.google.com/file/d/1jfLlYmg0cGOSpq7lDipRw9TxFrtoWAd-/view?usp=sharing).

#### 2.1 Wikipedia
 
- Download the latest dump.  https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
- Download the wikiextractor. https://github.com/attardi/wikiextractor

Ref: https://github.com/google-research/bert

#### 2.2 OpenAI Tokenizer

- Download the following two files:

https://huggingface.co/openai-gpt/resolve/main/vocab.json

https://huggingface.co/openai-gpt/resolve/main/merges.txt


## GPT1 Pretraining and Finetuning

This is a baseline implementation mainly based on minGPT and Huggingface transformers. We adopt modules in Hugginface for data pre-processing and minGPT for model architecture.

### 1. Pregenerate the GPT1 training data

You need to prepare the corpus and open-ai gpt tokenizer. Set the output dir path and working dir path in the code. The output dir files will be used in the next stage. 

Sample command:
```
python -u preprocess_wikipedia_gpt1.py | tee ./logs/preprocess_wikipedia_gpt1.txt
```

### 2. Pretrain the GPT1 with Wikipedia

Sample command:
```
python -u mingpt_wikipedia_train.py Wikipedia_minGPT_Pretrain 1 | tee ./logs/log_mingpt_wikipedia_pretrain.txt
```

### 3. Finetune the GPT1 on Classification Tasks and evaluate

Sample command:
```
python -u mingpt_cls_task_train.py MWPAS_GPT 1 | tee ./logs/log_mingpt_MWPAS_train.txt
python -u mingpt_cls_task_train.py ProbingTask_GPT 1 | tee ./logs/log_mingpt_ProbingTask_train.txt
python -u mingpt_cls_task_train.py GeneralNumberComparison_GPT 1 | tee ./logs/log_mingpt_GNC_train.txt

```


## NumGPT Pretraining and Finetuning

This is an implementation of NumGPT mainly based on minGPT and Huggingface transformers. We adopt modules in Hugginface for data pre-processing and minGPT for model architecture.

### 1. Pregenerate the NumGPT training data

Sample command:
```
python -u preprocess_wikipedia_fngpt1.py | tee ./logs/preprocess_wikipedia_fngpt1.txt
```

### 2. Pretrain the NumGPT with Wikipedia

Sample command:
```
python -u fngpt1v6_wikipedia_train.py Wikipedia_GPT1_Pretrain 8 | tee ./logs/log_fngpt1v6_wikipedia_pretrain.txt
```

### 3. Finetune the NumGPT on classification tasks and evaluate

Sample command:
```
python -u fngpt1v5_cls_task_train.py ProbingTask_FCGPT 8 | tee ./logs/log_fn2gpt1v5_MME_train.txt
python -u fngpt1v5_cls_task_train.py GeneralNumberComparison_FCGPT 8 | tee ./logs/log_fn2gpt1v5_GNC_train.txt
python -u fngpt1v5_cls_task_train.py MWPAS_FCGPT 8 | tee ./logs/log_fn2gpt1v5_MWPAS_train.txt
```

## Citation
If you find our work useful, please consider cite our work.
```
@article{jin2021numgpt,
  title={NumGPT: Improving numeracy ability of generative pre-trained models},
  author={Jin, Zhihua and Jiang, Xin and Wang, Xingbo and Liu, Qun and Wang, Yong and Ren, Xiaozhe and Qu, Huamin},
  journal={arXiv preprint arXiv:2109.03137},
  year={2021}
}
```