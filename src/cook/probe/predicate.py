# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random
import copy
from world import get_subgoal, get_optimal_plan


def make_dynamic_label(target, cands, ncands=None):
    '''
    return: {f"cand{}": str, "label": int}
    '''
    assert target in cands
    cands_copy = copy.deepcopy(cands)
    cands_copy = list(set(cands_copy))
    if ncands is not None and ncands < len(cands_copy):
        cands_copy = random.sample(cands_copy, ncands)
        if target not in cands_copy:
            cands_copy[0] = target
    random.shuffle(cands_copy)
    ret = {f"cand{i}": x for i, x in enumerate(cands_copy)}
    ret['label'] = cands_copy.index(target)
    return ret


def make_binary_label(targets, cands, ncands=None):
    '''
    return: {"cands": [str], "label": [binary]}
    '''
    assert all([x in cands for x in targets])
    cands_copy = copy.deepcopy(cands)
    cands_copy = list(set(cands_copy))
    neg_cands = [x for x in cands_copy if x not in targets]
    if ncands is not None and len(targets) > ncands:
        targets = targets[:ncands]
        ret = {f"cand{i}": x for i, x in enumerate(targets)}
        ret["label"] = [1.]*len(targets)
        return ret

    if ncands is not None and ncands-len(targets) < len(neg_cands):
        cands_copy = targets + random.sample(
                neg_cands, 
                ncands-len(targets)
            )
        assert len(cands_copy) <= ncands
    random.shuffle(cands_copy)
    ret = {f"cand{i}": x for i, x in enumerate(cands_copy)}
    ret["label"] = [1. if x in targets else 0. for x in cands_copy]
    return ret


'''
raw state predicates: obj_pos(box-name), boxes_order, agent_pos(box-name)
'''
def make_raw_predicates(boxes, ncands=None):
    all_bn = [x.name for box_row in boxes for x in box_row]
    ret = {}
    bid = 0
    for row in range(len(boxes)):
        for box in boxes[row]:
            ret[f"box_name_{bid}"] = make_dynamic_label(
                box.name, all_bn, ncands=ncands
            )
            bid += 1
    
    return ret


'''
world state predicates: obj_pos(abs), agent_pos
'''
def make_world_predicates(all_obj, boxes, agent, ncands=None):
    ret = {}
    all_on = [x.nickname() for x in all_obj]
    bid = 0
    for row in range(len(boxes)):
        for box in boxes[row]:
            ret[f"box_content_{bid}"] = make_binary_label(
                [x.nickname() for x in box.content],
                all_on, ncands=ncands
            )
            bid += 1

    ret["agent_content"] = make_binary_label(
                [x.nickname() for x in agent.objects],
                all_on, ncands=ncands
            )
    
    return ret


def make_ndest_predicates(future_ops, cur_state, final_state, agent, **kwargs):
    ret = {}
    task_prefix_row = 'goal_{}_row'
    task_prefix_col = 'goal_{}_col'
    aloc = agent.location
    subgoals = get_subgoal(cur_state, final_state)
    _, optimal_dest, _ = get_optimal_plan(subgoals, aloc)
    op = future_ops[0]
    for gidx, x in enumerate(optimal_dest):
        rp_hist_row = aloc['row'] - x['row']
        rp_hist_col = aloc['col'] - x['col']

        if gidx == 0 and op.atype == 'put':
            ret[task_prefix_row.format(gidx)] = {
                "label": 3
            }
            ret[task_prefix_col.format(gidx)] = {
                "label": 5
            }
            
        else:
            ret[task_prefix_row.format(gidx)] = {
                "label": int(rp_hist_row + 1)
            }
            ret[task_prefix_col.format(gidx)] = {
                "label": int(rp_hist_col + 2)
            }
    if len(optimal_dest) < 6: # eos
        for gidx in range(len(optimal_dest), 6):
            ret[task_prefix_row.format(gidx)] = {
                "label": 4
            }
            ret[task_prefix_col.format(gidx)] = {
                "label": 6 
            }

    return ret


'''
legal predicates
'''

def make_legal_predicates(ops, all_obj, boxes, agent, ncands=None):
    ret = {
        "row": {
            "label": agent.location['row'],
        },
        "col": {
            "label": agent.location['col'],
        },
        'overall': {
            "label": int(agent.location['row']*2+agent.location['col']),
        }

    }

    all_on = [x.nickname() for x in all_obj]
    for row in range(len(boxes)):
        for i, box in enumerate(boxes[row]):
            if row == agent.location['row'] and i == agent.location['col']:
                ret["box_content"] = make_binary_label(
                    [x.nickname() for x in box.content],
                    all_on, ncands=ncands
                )

    ret["agent_content"] = make_binary_label(
                [x.nickname() for x in agent.objects],
                all_on, ncands=ncands
            )
    return ret

'''
policy-abs state predicates: action_type, obj, direct
'''

_DIRECT_TO_LABEL = {
    's': 0,
    'n': 1,
    'w': 2,
    'e': 3
}

_ATYPE_TO_LABEL = {
    "move": 0,
    "grab": 1,
    "put": 2
}


def make_policy_predicates(ops, all_obj, ncands=None):
    op = ops[0]

    ret = {}
    if op.atype== 'put':
        ret["direct"] = {
            "label": 4
        }
    elif op.atype== 'grab':
        ret["direct"] = {
            "label": 5
        }
    else:
        if op.direct is not None:
            ret["direct"] = {
                    "label": _DIRECT_TO_LABEL[op.direct]
                }

    if op.obj is not None:
        ret['obj_ctrl'] = make_dynamic_label(
                op.obj.nickname(), 
                [x.nickname() for x in all_obj],
                ncands=ncands
            )
    return ret

