# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import defaultdict
from world import Environment, Agent
import argparse
import copy
import re
import pickle
import tqdm

from peft import PeftModel

from transformers import LlamaTokenizer, AutoModelForCausalLM


import torch

from generate_large_dataset import _OPERATIONS_DICT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str,
    )
    parser.add_argument(
        "--to_path", type=str,
    )
    parser.add_argument(
        "--prompt_method", type=str, choices=['cot', 'icl', 'inst-ft', 'base']
    )
    parser.add_argument(
        "--num_demo", type=int, default=1
    )
    parser.add_argument(
        "--ntest", type=int, default=-1
    )
    parser.add_argument(
        "--preserve_ops", action="store_true"
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

def batch_infer(model, tokenizer, prompts):
    batch_size = 2
    answers = []
    for batch_input in tqdm.tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(
            **encode_inputs, 
            max_new_tokens=1024, temperature=0.1
            )
        answers.extend(tokenizer.batch_decode(outputs[:, encode_inputs['input_ids'].size(1):], skip_special_tokens=True))

    return [x.strip() for x in answers]

def load(ckpt_dir, adapter_dir=None):
    n_gpus = torch.cuda.device_count()
    print('n_gpus', n_gpus)
    tokenizer = LlamaTokenizer.from_pretrained(
        ckpt_dir,
        use_fast=False,
        padding_side="left",
    )
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1

    # we use tensor parallel for loading llama
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, low_cpu_mem_usage=True, load_in_8bit=True)
    if not adapter_dir is None:
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()

    return model, tokenizer

def main():
    args = parse_args()

    with open(args.data_path, 'rb') as f:
        dataset = pickle.load(f)

    new_dataset = []
    for i, data in enumerate(dataset):

        init_s = data['init_state']
        init_loc = data['init_loc']
        
        agent = Agent(location=init_loc)
        env = Environment(
            agent=agent, 
            state=copy.deepcopy(init_s)
        )
        objs = env.gallery
        ops_hist = [act.replace('You ', "") if act.strip().startswith("You ") else act for act in data['text']['operations_hist'].split(".")]
        ops_hist = env.parse_actions('.'.join(ops_hist).strip())
        ops_hist = [act for act in ops_hist if act is not None]
        ops = env.parse_actions(data["text"]['actions'])
        ops = [act for act in ops if act is not None]
        noi = len(ops_hist)
        if args.preserve_ops:
            for box in env.state.boxes:
                box.max_items = 1000  # rm constraint
            obj_fixed_names = list(set([
                act.obj.nickname() for act in (ops_hist+ops) if act.atype!='move'
            ]))
            print('obj_fixed_names', obj_fixed_names)
            env.flip(obj_fixed=obj_fixed_names)
        else:
            env.flip()
        init_s = copy.deepcopy(env.state)
        init_state_copy = copy.deepcopy(init_s)
        init_loc = env.agent.location


        if args.preserve_ops:
            curr_s, exec_rate_1 = env.execute_operations(ops_hist)
            curr_loc = env.agent.location
            final_state, exec_rate_2 = env.execute_operations(ops)
            print('exec_rate_1', exec_rate_1)
            if noi>0: assert exec_rate_1 == 1.
            if not exec_rate_2 == 1.:
                print(noi)
                print(exec_rate_2)
                print([op.atype for op in ops])
            assert exec_rate_2 == 1.
            ops = ops_hist + ops
        else:
            trial = 0
            final_state = copy.deepcopy(init_state_copy)
            prefix2bad = defaultdict(lambda : [])
            while final_state == init_state_copy and trial < 500:
                agent = Agent(init_loc)
                init_state = copy.deepcopy(init_state_copy)
                env = Environment(agent, init_state)

                final_state, curr_loc, curr_s, operation_sequence, is_deadend = env.sample_traj(noi, len(ops))
                ops_record = [x.signature for x in operation_sequence]
                if is_deadend or final_state == init_state_copy:
                    prefix2bad[tuple(ops_record[:-1])].append(ops_record[-1])
                trial += 1
            ops = operation_sequence
        final_loc = env.agent.location
        
        nl_loc_info = Environment.textify_location(init_s, init_loc)  # location of boxes and agent
        has_lex_var = 'Modifier=both' in data['from']
        if 'obs=partial' in data['from']:
            nl_init_s = Environment.texify_state(init_s, is_initial=True, lex_variant=has_lex_var)
            nl_curr_s = Environment.texify_state(curr_s, is_initial=False, lex_variant=has_lex_var)
            nl_final_s = Environment.texify_state(final_state, is_initial=False, lex_variant=has_lex_var)
        elif 'obs=full' in data['from']:
            nl_init_s = Environment.texify_world(init_s, objs, is_initial=True, lex_variant=has_lex_var)
            nl_curr_s = Environment.texify_world(curr_s, objs, is_initial=False, lex_variant=has_lex_var)
            nl_final_s = Environment.texify_world(final_state, objs, is_initial=False, lex_variant=has_lex_var)

        ops_rollin = ops[:noi]
        ops_rollout = ops[noi:]

        agent_behavior = Environment.texify_agent_behavior(
            Environment.merge_ops(ops_rollin), 
            _OPERATIONS_DICT,
            init_loc,
            append_eos=False
        )
        oracle = Environment.texify_agent_behavior(
            ops_rollout, 
            _OPERATIONS_DICT,
            append_eos=True
        )
        new_dataset.append({
            "text":
                {
                'nl_loc_info': nl_loc_info,
                'init_state': nl_init_s,
                'curr_state': nl_curr_s,
                'final_state': nl_final_s,
                'operations_hist': agent_behavior,
                'actions': oracle
                },
            'init_state': init_s,
            'curr_state': curr_s,
            'final_state': final_state,
            'init_loc': init_loc,
            'curr_loc': curr_loc,
            'final_loc': final_loc
        })
    with open(args.to_path, 'wb') as f:
        pickle.dump(new_dataset, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()