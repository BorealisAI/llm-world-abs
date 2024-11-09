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
import pickle
import tqdm

from peft import PeftModel

from transformers import AutoTokenizer, AutoModelForCausalLM

from inference_hf import HFPrompterICL, HFPrompter
import torch
from probe_dataset_dyna import get_predicates
from pathlib import Path
from utils import get_start_and_end_char_ents

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str,
    )
    parser.add_argument(
        "--output_dataset_path", type=str,
    )
    parser.add_argument(
        "--split", type=str,
    )
    parser.add_argument(
        "--ckpt_dir", type=str,
    )
    parser.add_argument(
        "--predicate_model_dir", type=str,
    )
    parser.add_argument(
        "--tasks", type=str,
    )
    parser.add_argument(
        "--obj_mention_method", type=str, default=None, choices=['all', 'last', 'two', 'middle', None]
    )
    parser.add_argument(
        "--adapter_dir", type=str, default=None
    )
    parser.add_argument(
        "--final_predicate", action="store_true"
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
        "--avg_all_mentions", action="store_true"
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


def batch_infer(model, tokenizer, situation_and_ents, n_layer, obj_mention_method=None):
    batch_size = 6
    llm_repr = []
    llm_repr_ents = []
    for batch_input in tqdm.tqdm(batch_split(situation_and_ents, batch_size)):
        sttn = [x['situation'] for x in batch_input]
        ents = [x['ents'] for x in batch_input]
        start_and_end_char = [
                get_start_and_end_char_ents(
                    x, y, method=obj_mention_method
                ) for x, y in zip(sttn, ents)
            ]

        encode_inputs = tokenizer(
            sttn, 
            return_tensors="pt", 
            padding=True, 
            return_offsets_mapping=True,
        )

        for t in encode_inputs:
            if torch.is_tensor(encode_inputs[t]):
                encode_inputs[t] = encode_inputs[t].to('cuda')
        outputs = model(
            encode_inputs['input_ids'],
            encode_inputs['attention_mask'],
            return_dict=True,
            output_hidden_states=True
            )['hidden_states']
        outputs = outputs[n_layer].detach().cpu()

        for lmr, ids, offset, elist, sne in zip(
                outputs,
                encode_inputs['input_ids'], 
                encode_inputs['offset_mapping'], 
                ents,
                start_and_end_char
            ):
            lmr_ents = {}
            for e in elist:
                _, end_char = sne[e][0], sne[e][1]
                lmr_ents[e] = torch.zeros_like(lmr[0, :])
                for ec in end_char:
                    idx = int(ids.eq(tokenizer.pad_token_id).float().sum().item())
                    idx = len(ids) - 1
                    while idx >= 0 and offset[idx][1] >= ec:
                        idx -= 1
                    end_pos = idx + 1
                    lmr_ents[e] += lmr[end_pos, :]
                lmr_ents[e] = lmr_ents[e] / len(end_char)
            llm_repr_ents.append(lmr_ents)
        llm_repr.extend(outputs[:, -1, :])
    assert len(llm_repr) == len(llm_repr_ents)
    return llm_repr, llm_repr_ents


def dict_infer(model, tokenizer, cand_predicate, n_layer):
    batch_size = 8
    llm_repr = []
    for batch_input in batch_split(
            [v for k, v in cand_predicate.items() if k != "label"],
            batch_size):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model(
            **encode_inputs,
            return_dict=True,
            output_hidden_states=True
            )['hidden_states']
        outputs = outputs[n_layer][:, -1, :].detach().cpu()
        llm_repr.extend(outputs)

    return {k: lmr for (k, _), lmr in zip(cand_predicate.items(), llm_repr)}


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

    if 'phi-3' in ckpt_dir.lower():
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
    
    # we use tensor parallel for loading llama
    if 'pythia' in ckpt_dir.lower():
        model = AutoModelForCausalLM.from_pretrained(
                    ckpt_dir, trust_remote_code=True
            )
        model.cuda()
    else:    
        model = AutoModelForCausalLM.from_pretrained(
                    ckpt_dir, low_cpu_mem_usage=True, trust_remote_code=True, load_in_8bit=True, device_map="auto"
            )
    if adapter_dir is not None and adapter_dir != "None":
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()

    return model, tokenizer


def main(debug=False):
    args = parse_args()

    Path(args.output_dataset_path).mkdir(parents=True, exist_ok=True)

    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    random.shuffle(dataset)
    dataset = dataset[:args.n_data]
    model, tokenizer = load(args.ckpt_dir, args.adapter_dir)
    if 'mistral' in args.ckpt_dir.lower():
        tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = system_message + '\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

    if args.adapter_dir is not None: 
        assert args.prompt_method in ['base', 'inst-ft', 'icl', 'ft']
    if args.prompt_method == 'icl':
        formatter = HFPrompterICL(args.prompt_method, tokenizer)
        with open(args.icl_dataset_path, 'rb') as f:
            icl_demos = pickle.load(f)
            random.shuffle(icl_demos)
            icl_demos = icl_demos[:args.num_demo]
        formatter.proc_demos(icl_demos)
    else:
        formatter = HFPrompter(args.prompt_method, tokenizer)
    tasks = args.tasks.split(',')
    probe_data = [
            get_predicates(
                tasks, x, formatter, args.n_box_order_cands, 
                record_meta=args.record_meta, do_segment=args.do_segment, 
                ncands=args.num_neg
            ) for x in dataset
        ]
    probe_data = [x for x in probe_data if x is not None]

    i = 0
    for pdb in tqdm.tqdm(batch_split(probe_data, 2)):
        situation_and_ents = [
                {
                    'situation': x['situation'],
                    'ents': x['ents']
                } for x in pdb
            ]

        llm_repr_situation, llm_repr_ents = batch_infer(
            model, tokenizer, situation_and_ents, args.n_layer, obj_mention_method=args.obj_mention_method)
    
        predicate_data_and_repr = []
        
        for pd, lmr, lmr_ents in zip(pdb, llm_repr_situation, llm_repr_ents):
            pdr = {
                    'ft_lmr': lmr,  # tensor
                    'ents_lmr': lmr_ents,  # {e: lmr}
                    'predicate_data': pd["predicates"],
                    "nops_after_prompt": pd["nops_after_prompt"]
                }
            if args.record_meta:
                pdr["meta_info"] = pd['meta_info']

            predicate_data_and_repr.append(pdr)

            if debug and random.uniform(0, 1) < 0.01:
                print("-"*60)
                print('situation', pd['situation'])
                print('meta_info', pd['meta_info'])
                print('predicate_data')
                for k, v in pd['predicates'].items():
                    print(k)
                    for k2, v2 in v.items():
                        print(k2)
                        print(v2)
                print('actions')
                print(pd["actions"])
                print("-"*60)

        with open(
            os.path.join(
                args.output_dataset_path,
                f"{i}.pkl"), 'wb') as f:
            pickle.dump(predicate_data_and_repr, f, pickle.HIGHEST_PROTOCOL)
        i += 1


if __name__ == "__main__":
    main()
