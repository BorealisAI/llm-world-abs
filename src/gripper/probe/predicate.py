# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random
import copy
from collections import defaultdict
from world import get_subgoal, get_optimal_plan


def make_dynamic_label(target, cands, ncands=None):
    '''
    return {f"cand{}": str, "label": int}
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
    return {"cands": [str], "label": [binary]}
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
def make_raw_predicates(all_obj, boxes, agent, ncands=None):
    all_bn = [x.name for x in boxes]
    all_on = [x.nickname() for x in all_obj]
    ret = {}
    for i, box in enumerate(boxes):
        ret[f"box_name_{i}"] = make_dynamic_label(
            box.name, all_bn, ncands=ncands
        )

    obj2color = {}
    rep_obj = []
    for box in boxes:
        for obj in box.content:
            if obj.nickname() in rep_obj:
                rep_obj.append(obj.nickname())
            obj2color[obj.nickname()] = obj.color


    if all_obj[0].color != '':
        color2obj = defaultdict(lambda : [])
        for on, col in obj2color.items():
            if on in rep_obj: continue
            color2obj[col].append(on)

        for col, objs in color2obj.items():
            if len(objs) > 0:
                ret[col] = make_binary_label(
                    objs,
                    [x for x in all_on if x not in rep_obj],
                    ncands=ncands
                )
    return ret


'''
world state predicates: obj_pos(abs), agent_pos
'''
def make_world_predicates(all_obj, boxes, agent, ncands=None):
    ret = {}
    all_on = [x.nickname() for x in all_obj]
    for i, box in enumerate(boxes):
        ret[f"box_content_{i}"] = make_binary_label(
            [x.nickname() for x in box.content],
            all_on, ncands=ncands
        )
    ret["agent_content"] = make_binary_label(
                [x.nickname() for x in agent.objects],
                all_on, ncands=ncands
            )

    return ret


def make_ndest_predicates(future_ops, cur_state, final_state, agent, ncands=None):
    rp_hist = 0
    ret = {}
    task_prefix = 'goal_{}'
    aloc = agent.location
    subgoals = get_subgoal(cur_state, final_state)
    _, optimal_dest, _ = get_optimal_plan(subgoals, aloc)
    op = future_ops[0]
    for gidx, x in enumerate(optimal_dest):
        rp_hist = aloc - x

        if gidx == 0 and op.atype == 'put':
            ret[task_prefix.format(gidx)] = {
                "label": 9
            }

        else:
            ret[task_prefix.format(gidx)] = {
                "label": int(rp_hist + 4)
            }

    for gidx in range(len(optimal_dest), 4):
        ret[task_prefix.format(gidx)] = {
                "label": 10
            }

    return ret


def make_nopt_predicates(ops, all_obj, boxes, agent, ncands=None):
    ret = {
        'nopt': {
            "label": len([x for x in ops if x is not None])
        }
    }
    return ret


'''
legal predicates
'''


def make_legal_predicates(ops, all_obj, boxes, agent, ncands=None):
    ret = {
        "aloc": {
            "label": agent.location,
        }
    }

    all_on = [x.nickname() for x in all_obj]
    for i, box in enumerate(boxes):
        if i != agent.location: continue
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
    'left': 0,
    'right': 1
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
            "label": 2
        }
    elif op.atype== 'grab':
        ret["direct"] = {
            "label": 3
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
