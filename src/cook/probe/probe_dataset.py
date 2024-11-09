# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from world import WorldState, Environment, Agent, Entity, Box
import copy
import random
import os
import pickle
import dataclasses

from torch.utils.data import IterableDataset
from numpy.random import 

import torch

from predicate import make_raw_predicates, make_world_predicates, make_policy_predicates, make_legal_predicates, make_ndest_predicates


def get_meta_info(task_prompt, 
                future_ops, 
                init_s, curr_s,
                init_loc, curr_loc):
    hidden_info = []
    # aloc is hidden? --> (init_loc!=curr_loc OR move in prefix ops)
    if init_loc != curr_loc:
        hidden_info.append('agent_loc')
    
    if init_s != curr_s:
        hidden_info.append('obj_loc')

    obj_to_ctrl = [x.obj.nickname() for x in future_ops if x.obj is not None]

    if any([x in task_prompt for x in [" blue ", " green ", " red ", " yellow "]]):
        modifier = True
    else:
        modifier = False
    return {
        "obj_to_ctrl": obj_to_ctrl,
        "modifier": modifier,
        "hidden_info": hidden_info, # [agent_loc, obj_loc]
    }

def get_predicates(
        world_data, formatter, n_box_order, 
        record_meta=False, do_segment=False, ncands=None, for_obj_ctrl=False):
    situation = formatter.make_query_prompt_fix(world_data["text"])
    all_objects = Environment.list_all_objects(world_data['init_state'])
    curr_s = copy.deepcopy(world_data['curr_state'])
    agent = Agent(
            location=copy.deepcopy(world_data['curr_loc']),
            objects=Environment.infer_agent_objects(curr_s, all_objects)
        )
    env = Environment(agent=agent, state=curr_s)
    ops = env.parse_actions(world_data['text']['actions'])
    ops = [x for x in ops if x is not None]
    while ops[-1].atype == "move":
        ops = ops[:-1]
        if len(ops) == 0:
            return None
    assert len(ops) > 0
    nops_for_situ = random.choice(list(range(len(ops))))
    future_ops = ops[nops_for_situ:]
    ops = ops[:nops_for_situ]
    if len(ops) > 0:
        new_s, _ = env.execute_operations(ops)
    else:
        new_s = copy.deepcopy(world_data['curr_state'])
    
    situation += ". ".join(
            world_data['text']['actions'].strip().split(". ")[:nops_for_situ]
        ) 
        
    if nops_for_situ > 0:
        situation += ". "
    
    if for_obj_ctrl:
        situation += world_data['text']['actions'].split(". ")[nops_for_situ].split()[0] + " "

    predicates = {
        "raw": make_raw_predicates(new_s.boxes, ncands=ncands),
        "world": make_world_predicates(all_objects, new_s.boxes, env.agent, ncands=ncands),
        "world_dream": make_world_predicates(all_objects, new_s.boxes, env.agent, ncands=ncands),
        "ndest": make_ndest_predicates(future_ops, new_s, world_data['final_state'], env.agent, ncands=ncands),
        'legal': make_legal_predicates(future_ops, all_objects, new_s.boxes, env.agent, ncands=ncands),
        "policy": make_policy_predicates(future_ops, all_objects, ncands=ncands)
    }

    
    ret = {
        "situation": situation,
        "predicates": predicates,
        "nops_after_prompt": nops_for_situ,
        "actions": world_data['text']['actions']
    }

    if record_meta:
        # get task_prompt, future_ops,  obj_predicates, init_s, curr_s, init_loc, curr_loc
        meta_info = get_meta_info(
            situation, future_ops, 
            world_data['init_state'], new_s, 
            world_data['init_loc'], env.agent.location
        )
        ret["meta_info"] = meta_info
        return ret
    return ret


def get_predicates_with_final_world(
        world_data, formatter, n_box_order, 
        record_meta=False, do_segment=False, ncands=None):
    situation = formatter.make_query_prompt_fix(world_data["text"])
    all_objects = Environment.list_all_objects(world_data['init_state'])
    curr_s = copy.deepcopy(world_data['curr_state'])
    agent = Agent(
            location=copy.deepcopy(world_data['curr_loc']),
            objects=Environment.infer_agent_objects(curr_s, all_objects)
        )
    env = Environment(agent=agent, state=curr_s)
    ops = env.parse_actions(world_data['text']['actions'])
    ops = [x for x in ops if x is not None]

    while ops[-1].atype == "move":
        ops = ops[:-1]
        if len(ops) == 0:
            return None
    new_s, _ = env.execute_operations(ops)
    assert len(ops) > 0
    nops_for_situ = random.choice(list(range(len(ops))))
    future_ops = ops[nops_for_situ:]
    ops = ops[:nops_for_situ]
    
    situation += ". ".join(
            world_data['text']['actions'].strip().split(". ")[:nops_for_situ]
        ) 
        
    if nops_for_situ > 0:
        situation += ". "
    
    predicates = {
        "world_dream": make_world_predicates(all_objects, new_s.boxes, env.agent, ncands=ncands),
        'legal': make_legal_predicates(future_ops, all_objects, new_s.boxes, env.agent, ncands=ncands)
    }

    ret = {
        "situation": situation,
        "predicates": predicates,
        "nops_after_prompt": nops_for_situ,
        "actions": world_data['text']['actions']
    }

    if record_meta:
        # task_prompt, future_ops,  obj_predicates, init_s, curr_s, init_loc, curr_loc
        meta_info = get_meta_info(
            situation, future_ops, 
            world_data['init_state'], new_s, 
            world_data['init_loc'], env.agent.location
        )
        ret["meta_info"] = meta_info
        return ret
    return ret

@dataclasses.dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    def __call__(self, features):
        '''
        {pred_key: {"cand{i}": torch.tensor}}
        output: {pred_key: {"options": [bsz, ncand, dim], "label": int}}
        '''
        predicate_data = [x['predicate_data'] for x in features] # {pk: {"cand{i}": text, "label": int}}
        llm_repr_situation = [x['ft_lmr'] for x in features]
        llm_repr_predicate = [x['predicate_lmr'] for x in features]

        lmr_options = {}
        lmr_ctxt = {}
        labels = {}
        for pk in self.predicate_keys:
            ncand = list(predicate_data[0][pk].keys())
            lmr_options[pk] = torch.stack(
                    [
                        torch.stack(
                            [x[pk][f'cand{i}'] for i in range(ncand)],
                            dim=0
                        ) for x in llm_repr_predicate

                    ], 
                    dim=0
                )
            lmr_ctxt[pk] = torch.stack(
                [x['ft_lmr'] for x in features], dim=0
            )
            labels[pk] = torch.stack(
                [x[pk]['label'] for x in predicate_data]
            )

        return {
            "llm_repr_ctxt": llm_repr_situation,
            "llm_repr_options": lmr_options,
            "labels": labels
        }



class ProbeDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
    """

    def __init__(
        self,
        data_dir,
        label_repr_dir,
        shuffle=False,
        infinite=False,
        batch_size=32,
    ):
        fns = os.listdir(data_dir)
        self.fns = [os.path.join(data_dir, x) for x in fns if x.endswith(".pkl")]
        self.infinite = infinite
        self.batch_size = batch_size
        self.shuffle = shuffle

        with open(label_repr_dir, 'rb') as f:
            self.label_repr = pickle.load(f)

    def __iter__(self):
        fns = copy.deepcopy(self.fns)
        if self.shuffle:
            random.shuffle(fns)

        for fn in fns:
            with open(fn, 'rb') as f:
                data = pickle.load(f)
                if self.shuffle:
                    random.shuffle(data)

                for x in data:
                    yield x

        