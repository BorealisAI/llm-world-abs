# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import re
import pickle
import tqdm

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str,
    )
    parser.add_argument(
        "--output_dataset_path", type=str,
    )
    parser.add_argument(
        "--ckpt_dir", type=str,
    )
    parser.add_argument(
        "--predicate_model_dir", type=str,
    )
    parser.add_argument(
        "--adapter_dir", type=str, default=None
    )
    parser.add_argument(
        "--prompt_method", type=str, choices=['inst-ft']
    )
    parser.add_argument(
        "--num_demo", type=int, default=1
    )
    parser.add_argument(
        "--n_box_order_cands", type=int, default=5
    )

    parser.add_argument(
        "--n_data", type=int, default=-1
    )

    parser.add_argument(
        "--n_layer", type=int, default=-1
    )

    parser.add_argument(
        "--record_meta", action="store_true"
    )

    parser.add_argument(
        "--use_embed", action="store_true"
    )

    return parser.parse_args()

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def extract_list(text):
    items = [re.findall(r"\d{1,2}\.\s((?:(?!,\s\d{1,2}\.),?[^,]*)+)", x) for x in text.split('\n')]
    items = [x[0] for x in items if len(x)>0]
    return " ".join(items)


def batch_infer(model, tokenizer, state_and_ops, n_layer):
    batch_size = 6
    llm_repr = []
    for batch_input in tqdm.tqdm(batch_split(state_and_ops, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model(
            **encode_inputs,
            return_dict=True,
            output_hidden_states=True
            )['hidden_states']
        outputs = outputs[n_layer][:, -1, :].detach().cpu()
        llm_repr.extend(outputs)

    return llm_repr

def batch_infer_embedding(model, tokenizer, labels):
    batch_size = 6
    llm_repr = []
    for batch_input in tqdm.tqdm(batch_split(labels, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model(
            **encode_inputs,
            return_dict=True,
            output_hidden_states=True
            )['hidden_states']
        outputs = torch.sum(
                outputs[0] * encode_inputs['attention_mask'].float().unsqueeze(2), dim=1
            ) / encode_inputs['attention_mask'].float().sum(dim=1, keepdim=True)
        llm_repr.extend(outputs.detach().cpu())

    return llm_repr

def load(ckpt_dir, adapter_dir=None):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            ckpt_dir,
            use_fast=False,
            padding_side="left",
            trust_remote_code=True
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            ckpt_dir,
            use_fast=True,
            padding_side="left",
            trust_remote_code=True
        )
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    
    if "pythia" in ckpt_dir:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, trust_remote_code=True
        )
        model.cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, low_cpu_mem_usage=True, trust_remote_code=True, load_in_8bit=True)

    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()

    return model, tokenizer


def main():
    args = parse_args()

    dataset = []
    for fn in os.listdir(args.dataset_path):
        with open(
            os.path.join(args.dataset_path, fn), 
            'rb') as f:
            data = pickle.load(f)
        
        for x in data:
            for _, v1 in x['predicate_data'].items():
                for _, v2 in v1.items():
                    for k, v3 in v2.items():
                        if 'cand' in k:
                            if not isinstance(v3, list): 
                                v3 = [v3]
                            dataset += v3

    dataset = list(set(dataset))

    model, tokenizer = load(args.ckpt_dir, args.adapter_dir)
    if args.use_embed:
        llm_repr_label = batch_infer_embedding(model, tokenizer, dataset)
    else:
        llm_repr_label = batch_infer(model, tokenizer, dataset, args.n_layer)
    

    label_and_lmr = {}
    for label, lmr in zip(dataset, llm_repr_label):
        label_and_lmr[label] = lmr
    print(list(label_and_lmr.values())[0])

    #sanity check: 
    #for pd in probe_data:
    #    print("-"*60)
    #    print('situation', pd['situation'])
    #    print('predicate', pd['predicates'])
    #    print("-"*60)
   
    with open(args.output_dataset_path, 'wb') as f:
        pickle.dump(label_and_lmr, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()