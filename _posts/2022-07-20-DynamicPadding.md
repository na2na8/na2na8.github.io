---
title: "[NLP] Huggingface BERT 계열 Dynamic Padding"
layout: post
tags:
- NLP
- DynamicPadding
- RelationExtraction
categories:
- FinancialDomainLM
- NLP
---

## Intro
FinRED라고 하는 Relation Extraction Downstream Task를 시도하다가 데이터의 최대 토큰 길이, 평균 길이를 구하게 되었는데 Train에서 최대 토큰 길이는 2317, 평균 토큰 길이가 90이라 padding을 2317로 하기엔 비효율적이라 생각하여 Batch마다 Padding을 다르게 할 수 있는 Dynamic Padding을 찾아보게 되었다.

## Dynamic Padding
참고한 영상은 [What is Dynamic Padding](https://www.youtube.com/watch?v=7q5NyFT8REg)으로 huggingface에서 올린 영상이고 바로 밑에 나오는 두 코드 모두 해당 영상에서 나오는 코드이다.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example) :
    # padding 넣어주지 않음
    return tokenizer(example['sentence1'], example['sentence2'], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence1", "sentence2"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets = tokenized_datasets.with_format("torch")
```

```python
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer)
train_dataloader = DataLoader(
    tokenized_dataset["train"], batch_size=16, shuffle=True, collate_fn=data_collator
)
for step, batch in enumerate(train_dataloader) :
    print(batch['input_ids'].shape)
    if step > 5 :
        break
```
`DataCollatorWithPadding`으로 Dynamic Padding을 해줄 수 있다.

이를 Pytorch Lightning을 사용하는 내 코드에 적용시켰다.

## With Pytorch Lightning
```python
from collections import deque
import numpy as np
import pandas as pd

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding

class FinREDDataset(Dataset) :
    def __init__(self, path, tokenizer, special_tokens) :
        self.tokenizer = tokenizer

        # special_tokens dictionary
        self.special_tokens = special_tokens

        # class_dict
        self.class_dict = self.make_class_dict()
        # text iterator
        self.txt = open(path, 'r')
        # make dataframe from iterator
        self.dataset = self.make_df(self.txt)
        # remove nan 
        self.datasef = self.dataset.dropna(axis=0)
        # remove duplicates
        self.dataset.drop_duplicates(subset=['sentence'], inplace=True)
    
    def __len__(self) :
        return len(self.dataset)

    def __getitem__(self, idx) :
        sentence = self.dataset['sentence'].iloc[idx]
        head = self.dataset['head'].iloc[idx]
        tail = self.dataset['tail'].iloc[idx]
        relation = self.dataset['relation'].iloc[idx]

        # apply head tokens
        sentence = sentence.replace(head, ' '.join([special_tokens['head_start'], head, special_tokens['head_end']]))
        # apply tail tokens
        sentence = sentence.replace(tail, ' '.join([special_tokens['tail_start'], tail, special_tokens['tail_end']]))

        inputs = self.tokenizer(
            sentence,
            truncation=False,
            add_special_tokens=True
        )

        return {
            'input_ids' : torch.tensor(inputs['input_ids']),
            'attention_mask' : torch.tensor(inputs['attention_mask']),
            'relation' : relation
        }
    
	  ...
		
		
class FinREDDataModule(pl.LightningDataModule) :
    def __init__(self, path, args, tokenizer, special_tokens) :
        super().__init__()
        self.train_path = path['train']
        self.valid_path = path['valid']
        self.test_path = path['test']

        self.batch_size = args['batch_size']
        self.num_workers = args['num_workers']

        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        
        self.args = args

    def setup(self) :
        self.set_train = FinREDDataset(self.train_path, self.tokenizer, self.special_tokens)
        self.set_valid = FinREDDataset(self.valid_path, self.tokenizer, self.special_tokens)
        self.set_test = FinREDDataset(self.test_path, self.tokenizer, self.special_tokens)

    def train_dataloader(self) :
        train = DataLoader(self.set_train, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.data_collator, shuffle=True)
        return train

    def val_dataloader(self) :
        valid = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.data_collator)
        return valid

    def test_dataloader(self) :
        test = DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.data_collator)
        return test
```


jupyter notebook에서 사용한 코드는 위와 같다.

1. `tokenizer`의 `truncation`을 `False`로 해야 사이즈가 512가 넘는 토큰 시퀀스를 자르지 않고 온전하게 유지할 수 있다.
2. `FinREDDataset`의 tokenizer에 padding을 설정하지 않는다.
3. 밑의 `FinREDDataModule`의 `DataLoader`부분에서 `__init__`부분에서 만든 collator를 `collate_fn=self.data_collator`로 넣어준다.


해당 코드에서 데이터 모듈을 생성하고 생성한 batch를 확인해 보면 다음과 같다.

### Code

```python
dm = FinREDDataModule(path, args, tokenizer, special_tokens)
dm.setup()
t = dm.train_dataloader()
for step, batch in enumerate(t) :
    print(batch['input_ids'].shape)
```

### Result
```python
torch.Size([64, 826])
torch.Size([64, 780])
torch.Size([64, 515])
torch.Size([64, 116])
torch.Size([64, 551])
torch.Size([64, 322])
torch.Size([64, 409])
torch.Size([64, 126])
...
```

성공적으로 Dynamic Padding이 된 것을 볼 수 있다.
