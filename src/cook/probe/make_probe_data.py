# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import random
import re
import pickle
import tqdm

from peft import PeftModel

from transformers import AutoTokenizer, AutoModelForCausalLM

from inference_hf import HFPrompterICL, HFPrompter

import torch
from probe_dataset import get_predicates, get_predicates_with_final_world
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str,
    )
    parser.add_argument(
        "--split", type=str, required=True
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
        "--icl_dataset_path", type=str, default=None
    )
    parser.add_argument(
        "--prompt_method", type=str, choices=['inst-ft', 'icl', 'ft']
    )
    parser.add_argument(
        "--num_demo", type=int, default=1
    )
    parser.add_argument(
        "--bsz", type=int, default=6
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
        "--num_neg", type=int, default=None
    )

    parser.add_argument(
        "--record_meta", action="store_true"
    )

    parser.add_argument(
        "--do_segment", action="store_true"
    )

    parser.add_argument(
        "--for_obj_ctrl", action="store_true"
    )

    parser.add_argument(
        "--final_predicate", action="store_true"
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


def batch_infer(bsz, model, tokenizer, state_and_ops, n_layer):
    batch_size = bsz
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

def dict_infer(model, tokenizer, cand_predicate, n_layer):
    batch_size = 8
    llm_repr = []

    for batch_input in batch_split(
            [v for k, v in cand_predicate.items() if k != "label"],
            batch_size
        ):

        print("batch_input", batch_input)
        encode_inputs = prepare_input(tokenizer, batch_input)
        print('encode_inputs', encode_inputs.keys())
        outputs = model(
            **encode_inputs,
            return_dict=True,
            output_hidden_states=True
            )['hidden_states']
        outputs = outputs[n_layer][:, -1, :].detach().cpu()
        llm_repr.extend(outputs)

    return {k: lmr for (k, _), lmr in zip(cand_predicate.items(), llm_repr)}

def load(ckpt_dir, adapter_dir=None):
    n_gpus = torch.cuda.device_count()
    print('n_gpus', n_gpus)
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
    if 'llama-2' in ckpt_dir.lower():
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
    else:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    if "pythia" in ckpt_dir:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, trust_remote_code=True
        )
        model.cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, low_cpu_mem_usage=True, trust_remote_code=True, load_in_8bit=True)

    if adapter_dir is not None and adapter_dir != "None":
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()

    return model, tokenizer


def main(debug=False):
    args = parse_args()

    
    Path(args.output_dataset_path).mkdir(parents=True, exist_ok=True)

    if args.split == 'train':
        print("for TRAIN split!!!!!!")
        dataset = []
        for fn in os.listdir(args.dataset_path):
            with open(
                os.path.join(args.dataset_path, fn), 'rb') as f:
                dataset += pickle.load(f)
    else:
        with open(args.dataset_path, 'rb') as f:
            dataset = pickle.load(f)

    random.shuffle(dataset)
    dataset = dataset[:args.n_data]


    model, tokenizer = load(args.ckpt_dir, args.adapter_dir)
    if 'mistral' in args.ckpt_dir.lower():
        tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = system_message + '\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

    if args.adapter_dir is not None: assert args.prompt_method in ['base', 'icl', 'ft', 'inst-ft']
    if args.prompt_method == 'icl':
        formatter = HFPrompterICL(args.prompt_method, tokenizer)
        with open(args.icl_dataset_path, 'rb') as f:
            icl_demos = pickle.load(f)
            random.shuffle(icl_demos)
            icl_demos = icl_demos[:args.num_demo]
        formatter.proc_demos(icl_demos)
    else:
        formatter = HFPrompter(args.prompt_method, tokenizer)
    func = get_predicates_with_final_world if args.final_predicate else get_predicates
    probe_data = [
            func(
                x, formatter, args.n_box_order_cands, 
                record_meta=args.record_meta, do_segment=args.do_segment, 
                ncands=args.num_neg, for_obj_ctrl=args.for_obj_ctrl
            ) for x in dataset
        ]
    probe_data = [x for x in probe_data if x is not None]
    situation = [x['situation'] for x in probe_data]

    llm_repr_situation = batch_infer(args.bsz, model, tokenizer, situation, args.n_layer)

    predicate_data_and_repr = []
    for pd, lmr in zip(probe_data, llm_repr_situation):
        pdr = {
                'ft_lmr': lmr, # tensor 
                'predicate_data': pd["predicates"],
                "nops_after_prompt": pd["nops_after_prompt"]
            }
        if args.record_meta:
            pdr["meta_info"] = pd['meta_info']

        predicate_data_and_repr.append(pdr)
        
        if debug and random.uniform(0, 1) < 0.1:
            print("-"*60)
            print('situation', pd['situation'])
            print('predicate_data')
            for k, v in pd['predicates'].items():
                print(k)
                for k2, v2 in v.items():
                    print(k2)
                    print(v2)
            print('actions')
            print(pd["actions"])
            print("-"*60)
        

    i = 0
    for pdr in tqdm.tqdm(batch_split(predicate_data_and_repr, 100)):
        with open(
                os.path.join(
                    args.output_dataset_path,
                    f"{i}.pkl"), 'wb') as f:
            pickle.dump(pdr, f, pickle.HIGHEST_PROTOCOL)
            i += 1

if __name__ == "__main__":
    main()