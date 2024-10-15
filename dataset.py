# adapted from https://github.com/PeterGriffinJin/Patton/blob/main/src/OpenLP/dataset/train_dataset.py

from dataclasses import dataclass
from transformers import DataCollatorWithPadding, DefaultDataCollator, PreTrainedTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import Dataset
from arguments import DataArguments
from transformers import Trainer
import os
import glob
import random
import pandas as pd
from typing import List, Tuple, Dict
import torch
        
class TrainDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args, shuffle_seed: int = None, cache_dir: str = None, k=16) -> None:
        super(TrainDataset, self).__init__()
        self.data_files = [data_args.train_path] if data_args.train_dir is None else glob.glob(os.path.join(data_args.train_dir, "*.jsonl"))
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else self.dataset
        self.tokens = pd.read_csv(data_args.corpus_path, sep='\t', names=['asin', 'title'])['title'].values if data_args.corpus_path is not None else None
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len
        self.k = k #negative sampling

    def create_one_example(self, text_encoding: List[int]):

        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):  # __len__ is required by huggingface trainer
        count = len(self.dataset)
        return count

    def process_multi(self, example):
        encoded_query = self.create_one_example(example['q_text'])
        encoded_key = self.create_one_example(example['k_text'])
        encoded_query_n = [self.create_one_example(q_n) if q_n != [] else self.create_one_example([0]) for q_n in example['q_n_text']]
        encoded_key_n = [self.create_one_example(k_n) if k_n != [] else self.create_one_example([0]) for k_n in example['k_n_text']]
        query_n_mask = [1 if q_n != [] else 0 for q_n in example['q_n_text']]
        key_n_mask = [1 if k_n != [] else 0 for k_n in example['k_n_text']]
        return {"query": encoded_query, "key": encoded_key, 'query_n':encoded_query_n, 'key_n':encoded_key_n, 'query_n_mask':query_n_mask, 'key_n_mask':key_n_mask}

    
    def process_fn(self, example):
        encoded_query = self.create_one_example(example['q_text'])
        encoded_key = self.create_one_example(example['k_text'])
        return {"query": encoded_query, "key": encoded_key}


    def __getitem__(self, index):
        return self.process_fn(self.dataset[index])

class DPTrainDataset(TrainDataset):
    def process_fn(self, example):
        encoded_query = self.create_one_example(example['q_text'])
        encoded_key = self.create_one_example(example['k_text'])
        neg_sample = [self.create_one_example(neg) for neg in random.choices(self.tokens, k=self.k)]
        return {"query": encoded_query, "key": encoded_key, "key_neg": neg_sample}


class EvalDataset(TrainDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args, shuffle_seed: int = None, cache_dir: str = None) -> None:
        super(EvalDataset, self).__init__(tokenizer, data_args, None, cache_dir=cache_dir)

        self.data_files = [data_args.eval_path]
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else self.dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len

    def process_fn(self, example):
        encoded_query = self.create_one_example(example['q_text'])
        encoded_key = self.create_one_example(example['k_text'])
        return {"query": encoded_query, "key": encoded_key}

    def __getitem__(self, index):
        return self.process_fn(self.dataset[index])
    
class TrainNCCDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, trainer: Trainer = None, shuffle_seed: int = None, cache_dir: str = None) -> None:
        super(TrainNCCDataset, self).__init__()
        self.data_files = [data_args.train_path] if data_args.train_dir is None else glob.glob(os.path.join(data_args.train_dir, "*.jsonl"))
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else self.dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len
        self.trainer = trainer

    def create_one_example(self, text_encoding: List[int]):

        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):  # __len__ is required by huggingface trainer
        count = len(self.dataset)
        return count

    def process_fn(self, example):
        encoded_query = self.create_one_example(example['q_text'])
        return {"query": encoded_query, 'label':example['label']}

    def __getitem__(self, index):
        return self.process_fn(self.dataset[index])


class EvalNCCDataset(TrainNCCDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, shuffle_seed: int = None, cache_dir: str = None, eval=True) -> None:
        super(EvalNCCDataset, self).__init__(tokenizer, data_args, None, cache_dir=cache_dir)

        self.data_files = [data_args.eval_path] if eval else [data_args.test_path]
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else self.dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len

    def __getitem__(self, index):

        return self.process_fn(self.dataset[index])



@dataclass
class TrainCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):
        qq = [f["query"] for f in features]
        kk = [f["key"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(kk[0], list):
            kk = sum(kk, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        k_collated = self.tokenizer.pad(
            kk,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )

        if "key_neg" in features[0]:
            vv = [f["key_neg"] for f in features]
            if isinstance(vv[0], list):
                # vv = sum(vv, [])
                # [[1,2,3], [a,b,c], [x,y,z]] -> [1,a,x,2,b,y,3,c,z]
                # easy to compute the gradient norms for each tuple
                vv = [item for sublist in vv for item in sublist]
            v_collated = self.tokenizer.pad(
                            vv,
                            padding='max_length',
                            max_length=self.max_len,
                            return_tensors="pt",
                        )
            return q_collated, k_collated, v_collated
        else:
            return  q_collated, k_collated


@dataclass
class TrainNCCCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):

        qq = [f["query"] for f in features]
        labels = [f["label"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        labels = torch.LongTensor(labels)
        
        return {'center_input': q_collated}, labels