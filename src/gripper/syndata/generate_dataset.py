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
from world import WorldState, Environment, Agent, Entity, Box
import csv
import copy
import random

import os
import string
import pickle

import tqdm

from types import SimpleNamespace

# Possible operations
_OPERATIONS_DICT = {
    "move": "Move {direction}.",
    "move_n": "Move {direction} for {nstep} steps.",
    "put": "Put {content} in the nearby container.",
    "grab": "Grab {content} out of the nearby container.",
    "paint": "Brush {color} paint onto {content}."
}


_MODIFIERS = {
    "size": ["big", "small", "tiny"],
    "color": ["blue", "green", "red", "yellow"]
}

_SPLITS_PROP = {"train": 0.5, "dev": 0.1, "test": 0.4}

_CONTAINER_NAMES = ['Box', 'Basket', 'Bucket', 'Crate']

def create_objects(args, obj_types):
    objs = []
    nobj = args.expected_num_items_per_box * args.num_boxes
    for i, ot in enumerate(obj_types):
        if args.modifier == 'color':
            objs.append(
                Entity(
                    ot, i,
                    color=random.choice(_MODIFIERS['color'])
                )
            )
        elif args.modifier == 'size':
            objs.append(
                Entity(
                    ot, i,
                    size=random.choice(_MODIFIERS['size'])
                )
            )
        elif args.modifier == 'both':
            objs.append(
                Entity(
                    ot, i,
                    color=random.choice(_MODIFIERS['color']),
                    size=random.choice(_MODIFIERS['size'])
                )
            )
        else:
            objs.append(
                Entity(ot, i)
            )
    return objs
        
def create_containers(num_boxes, max_items_per_box):
    boxes = []
    container_to_idx = defaultdict(lambda : string.ascii_uppercase)
    for i in range(num_boxes):
        box_name = random.choice(_CONTAINER_NAMES)
        boxes.append(
            Box(container_to_idx[box_name][0], box_name, max_items_per_box)
        )
        container_to_idx[box_name] = container_to_idx[box_name][1:]
    
    return boxes


def load_objects_from_csv(csv_path):
    object_list = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            object_list.append(row["object_name"])
    return frozenset(object_list)

def syn_data(args):

    objects_set = load_objects_from_csv(args.object_vocabulary_file)

    operations = list(_OPERATIONS_DICT.keys())
    sampled_sequences = []
    for _ in tqdm.tqdm(range(args.num_data)):
        entities = create_objects(args, objects_set)
        boxes = create_containers(args.num_boxes, args.max_items_per_box)
        init_state = WorldState.sample_initial_world_state(
                    boxes,
                    entities,
                    max_items_per_box=args.max_items_per_box,
                    expected_num_items_per_box=args.expected_num_items_per_box
                )
        init_state_copy = copy.deepcopy(init_state)
        
        final_state = copy.deepcopy(init_state)
        init_loc = random.choice([0, 1, 3, 4])
        prefix2bad = defaultdict(lambda : [])
        noi = random.choice(list(range(args.num_operations_init+1)))
        trial = 0
        while final_state.eq(init_state_copy) and trial < 500:
            agent = Agent(init_loc)
            init_state = copy.deepcopy(init_state_copy)
            env = Environment(agent, init_state)

            
            final_state, curr_loc, curr_state, operation_sequence, is_deadend = env.sample_traj(
                    noi, args.num_operations, args.min_op_dist
            )
            ops_record = [x.signature for x in operation_sequence]
            if is_deadend or final_state.eq(init_state_copy):
                prefix2bad[tuple(ops_record[:-1])].append(ops_record[-1])
            trial += 1
        final_loc = env.agent.location

        sampled_sequences.append(
            (init_state_copy, curr_state, final_state, init_loc, curr_loc, final_loc, operation_sequence, noi, env.gallery)
        )  

    dataset = []
    existing_states_pairs = []
    has_lex_var = args.modifier=='both'
    for init_s, curr_s, final_s, init_loc, curr_loc, final_loc, ops, noi, objs in sampled_sequences:
        if (init_s, final_s) in existing_states_pairs:
            continue
        else:
            existing_states_pairs.append((init_s, final_s))

        nl_loc_info = Environment.textify_location(init_s, init_loc) # location of boxes and agent
        
        if args.observable == 'partial':
            nl_init_s = Environment.texify_state(init_s, is_initial=True, lex_variant=has_lex_var)
            nl_curr_s = Environment.texify_state(curr_s, is_initial=False, lex_variant=has_lex_var)
            nl_final_s = Environment.texify_state(final_s, is_initial=False, lex_variant=has_lex_var)
        elif args.observable == 'full':
            nl_init_s = Environment.texify_world(init_s, objs, is_initial=True, lex_variant=has_lex_var)
            nl_curr_s = Environment.texify_world(curr_s, objs, is_initial=False, lex_variant=has_lex_var)
            nl_final_s = Environment.texify_world(final_s, objs, is_initial=False, lex_variant=has_lex_var)

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

        dataset.append({
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
            'final_state': final_s,
            'init_loc': init_loc,
            'curr_loc': curr_loc,
            'final_loc': final_loc
        })

    splits_size = {
            split: int(len(dataset) * prop) for split, prop in _SPLITS_PROP.items()
        }
    random.shuffle(dataset)
    train_set = dataset[:splits_size["train"]]
    val_set = dataset[splits_size["train"]:splits_size["train"]+splits_size["dev"]]
    test_set = dataset[-splits_size["test"]:]

    return train_set, val_set, test_set


def merge_data(datadir, split):
    all_data = []
    for file in os.listdir(datadir):
        if file.endswith(f"_{split}.pkl"):
            with open(os.path.join(datadir, file), 'rb') as f:
                all_data += pickle.load(f)
    random.shuffle(all_data)
    return all_data

def merge_data(savedir, split):
    all_data = []
    for file in os.listdir(savedir):
        if file.endswith(f"_{split}.pkl"):
            with open(os.path.join(savedir, file), 'rb') as f:
                all_data += pickle.load(f)
    return all_data

def main():
    params = SimpleNamespace(
        to_path = './gribber_data/', 
        object_vocabulary_file = None,
        num_boxes = 5,
        expected_num_items_per_box = 2,
        num_operations = None,
        num_operations_init = None,
        num_data = 10000,
        modifier = None,
        observable = None,
        max_items_per_box = 4 # NOTE: this attribute is deprecated
        min_op_dist = 3
        )

    for variants in [
        {
            "object_vocabulary_file": './objects_with_bnc_frequency.csv',
            "modifier": "none",
            "observable": "full",
            "num_operations_init": 0
        },
        {
            "object_vocabulary_file": './objects_with_bnc_frequency.csv',
            "modifier": "none",
            "observable": "partial",
            "num_operations_init": 3
        },
        {
            "object_vocabulary_file": './objects_not_in_bnc.csv',
            "modifier": "both",
            "observable": "full",
            "num_operations_init": 0
        },
        {
            "object_vocabulary_file": './objects_with_bnc_frequency.csv',
            "modifier": "none",
            "observable": "partial",
            "num_operations_init": 0
        },
    ]:
        params.__dict__.update(variants)
        train_set_all, val_set_all, test_set_all = [], [], []
        for numop in tqdm.tqdm(range(2, 6)):
            params.num_operations = numop

            train_set, val_set, test_set = syn_data(params)
            train_set_all += train_set
            val_set_all += val_set
            test_set_all += test_set

        fn = f'obs={params.observable}_Modifier={params.modifier}_{params.num_boxes}boxes_{params.expected_num_items_per_box}itemsPerBox_{2-6}ops_{params.num_operations_init}PrefixOps_train.pkl'
        with open(os.path.join(params.to_path, fn), 'wb') as f:
            pickle.dump(train_set_all, f, pickle.HIGHEST_PROTOCOL)

        fn = f'obs={params.observable}_Modifier={params.modifier}_{params.num_boxes}boxes_{params.expected_num_items_per_box}itemsPerBox_{2-6}ops_{params.num_operations_init}PrefixOps_val.pkl'
        with open(os.path.join(params.to_path, fn), 'wb') as f:
            pickle.dump(val_set_all, f, pickle.HIGHEST_PROTOCOL)

        fn = f'obs={params.observable}_Modifier={params.modifier}_{params.num_boxes}boxes_{params.expected_num_items_per_box}itemsPerBox_{2-6}ops_{params.num_operations_init}PrefixOps_test.pkl'
        with open(os.path.join(params.to_path, fn), 'wb') as f:
            pickle.dump(test_set_all, f, pickle.HIGHEST_PROTOCOL)



    train_data = merge_data(params.to_path, "train")
    val_data = merge_data(params.to_path, "val")
    test_data = merge_data(params.to_path, "test")

    with open(
        os.path.join(params.to_path, "train.pkl"), 'wb') as f:
        pickle.dump(train_data[:100000], f)

    with open(
        os.path.join(params.to_path, "val.pkl"), 'wb') as f:
        pickle.dump(val_data[:1000], f)

    with open(
        os.path.join(params.to_path, "val.pkl"), 'wb') as f:
        pickle.dump(test_data[:1000], f)