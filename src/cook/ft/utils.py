# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2022-present, The HuggingFace Team.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py by HuggingFace which is licensed under the Apache License, Version 2.0 (the "License").
# You may obtain a copy of the License at 
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# 
#################################################################################### 

import random
import torch
from torch.utils.data import IterableDataset, Dataset
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import transformers
from typing import Dict, Any
from transformers.trainer_pt_utils import LabelSmoother
import os, copy
import pickle

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def rank0_print(*args):
    print(*args)

def create_and_prepare_model(args):
    device_map = None
    bnb_config = None
    load_in_8bit = args.use_8bit_qunatization

    if args.use_4bit_qunatization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_qunatization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.use_4bit_qunatization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)

    if args.use_4bit_qunatization or args.use_8bit_qunatization:
        device_map = "auto"

    if args.load_from_config:
        config = AutoConfig.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_config(config)
        model.cuda()
    else:    
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            load_in_8bit=load_in_8bit,
            quantization_config=bnb_config,
            device_map=device_map,
            use_cache=not args.use_gradient_checkpointing,
            trust_remote_code=True,
        )

    peft_config = None
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(","),
        )
        if (args.use_4bit_qunatization or args.use_8bit_qunatization) and args.use_peft_lora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.use_gradient_checkpointing)

        if args.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if 'llama-2' in args.model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.padding_side = 'right'
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path,
                use_fast=False,
                trust_remote_code=True
                )
        except:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path,
                use_fast=True,
                trust_remote_code=True
                )

        tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    return model, peft_config, tokenizer

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, json_data, tokenizer: transformers.PreTrainedTokenizer):
        
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")

        self.json_data = json_data
        self.tokenizer = tokenizer
        self.ignore_index = IGNORE_TOKEN_ID

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, i):

        return self.json_data[i]



def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    rank0_print("Loading data...")



    train_dataset = IterSupervisedDataset(args.dataset_path, tokenizer=tokenizer)

    with open(args.eval_dataset_path, 'rb') as f:
        eval_raw_data = pickle.load(f)
    
    eval_json = [
        {
            "nl_loc_info": rd['text']['nl_loc_info'],
            "init_state": rd['text']['init_state'],
            "final_state": rd['text']['final_state'],
            "operations_hist": rd['text']['operations_hist'],
            "actions": rd['text']['actions'],
        } for rd in eval_raw_data
    ]
    eval_dataset = SupervisedDataset(eval_json, tokenizer=tokenizer)

    return train_dataset, eval_dataset
    

def preprocess(
    json_data,
    tokenizer: transformers.PreTrainedTokenizer,
    formatting_func,
    sep_token=None,
    debug=False,
) -> Dict:
    # Apply prompt templates
    assert sep_token is not None
    text_data = []
    for jd in json_data:
        text_data.append(formatting_func(jd))

    input_ids = tokenizer(
        text_data,
        return_tensors="pt",
        padding="longest",
        max_length=4096,
        truncation=True,
    ).input_ids
    
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.

    for j, input_ in enumerate(text_data):
        parts = input_.split(sep_token)

        cur_len = 1
        # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
        instruction_len = len(tokenizer(
                sep_token.join(parts[:-1])+sep_token
            ).input_ids) 
        if sep_token != '\n':
            instruction_len = instruction_len - 2
            targets[j, :cur_len + instruction_len] = IGNORE_TOKEN_ID
        else:
            targets[j, :instruction_len] = IGNORE_TOKEN_ID


    if debug:  # Inspect and check the correctness of masking
        z = targets.clone()
        z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.pad_token_id, z)
        rank0_print(tokenizer.decode(z))

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

@dataclass
class DataCollatorWithPadding:
    tokenizer: transformers.PreTrainedTokenizerBase
    formatting_func: Any = None
    sep_token: str = None

   
    def __call__(self, features):
        return preprocess(features, self.tokenizer, self.formatting_func, sep_token=self.sep_token)


class IterSupervisedDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
    """

    def __init__(
        self,
        data_dir,
        tokenizer,
        split="train",
        shuffle=False,
        infinite=False,
        batch_size=32,
    ):
        fns = os.listdir(data_dir)
        self.fns = [os.path.join(data_dir, x) for x in fns if split in x]
        self.infinite = infinite
        self.tokenizer = tokenizer
        self.ignore_index = IGNORE_TOKEN_ID
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        n = 0
        for fn in self.fns:
            with open(fn, 'rb') as f:
                n += len(pickle.load(f))
        return n

    def __iter__(self):
        fns = copy.deepcopy(self.fns)
        if self.shuffle:
            random.shuffle(fns)

        for fn in fns:
            with open(fn, 'rb') as f:
                data = pickle.load(f)
                if self.shuffle:
                    random.shuffle(data)
                for rd in data:
                    yield {
                        "nl_loc_info": rd['text']['nl_loc_info'],
                        "init_state": rd['text']['init_state'],
                        "final_state": rd['text']['final_state'],
                        "operations_hist": rd['text']['operations_hist'],
                        "actions": rd['text']['actions'],
                    }

    def __call__(self, features):
        return preprocess(features, self.tokenizer, self.formatting_func, sep_token=self.sep_token)