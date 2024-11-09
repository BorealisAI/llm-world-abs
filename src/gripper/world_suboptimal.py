# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2023-present, Sebastian Schuster
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on (https://aclanthology.org/2023.acl-long.213.pdf) implementation from https://github.com/sebschu/entity-tracking-lms/blob/main/src/dataset_generation/generate_boxes_data.py by Sebastian Schuster which is licensed by CC BY-NC-SA
# You may obtain a copy of the License at 
# 
#   https://creativecommons.org/licenses/by-nc-sa/4.0/
# 
# 
#################################################################################### 

import collections
from numpy.random import poisson
import numpy as np
import random
import copy
import re
import Levenshtein
import itertools

_CONTAINS_VERB = ["contains", 'has', 'holds', "is occupied by"]
_EMPTY_EXPR = ['is empty', 'contains nothing', 'has nothing', 'is vacant']

def get_dest(ops, cur_loc):
    aloc = copy.deepcopy(cur_loc)
    dest = []
    for op in ops:
        if op.atype == 'move':
            if op.direct == 'left':
                aloc -= 1
            else:
                aloc += 1
        elif op.atype in ['grab', 'put']:
            dest.append(aloc)
    return dest


def get_subgoal(sinit, sgoal):
    obj_to_put = {}
    obj_to_grab = {}
    for i, (b1, b2) in enumerate(zip(sinit.boxes, sgoal.boxes)):
        b1_objs = [x.nickname() for x in b1.content]
        b2_objs = [x.nickname() for x in b2.content]
        for obj in b1_objs:
            if obj not in b2_objs:
                obj_to_grab[obj] = i
        for obj in b2_objs:
            if obj not in b1_objs:
                obj_to_put[obj] = i

    subgoals = []
    for obj, i in obj_to_put.items():
        if obj in obj_to_grab.keys():
            subgoals.append(
                (obj, obj_to_grab[obj], i)
            )
        else:
            subgoals.append(
                (obj, None, i)
            )

    for obj, i in obj_to_grab.items():
        if obj not in obj_to_put.keys():
            subgoals.append(
                (obj, i, None)
            )
    
    return subgoals
    
def get_optimal_plan(subgoals, aloc):
    min_cost = 1000000
    best_goal_traj = None
    best_dest = None
    gt2cost = []
    for goal_traj in itertools.permutations(subgoals):
        cost = 0.
        cur_end = None
        dest = []
        cur_loc = copy.deepcopy(aloc)
        for i, goal in enumerate(goal_traj):
            
            if i == 0:
                if goal[1] is not None:
                    dest.append(goal[1])
                    cost += abs(cur_loc - goal[1])
                    cur_loc = goal[1]
                
                if goal[2] is not None:
                    cur_end = goal[2]
                
                continue
            else:
                if cur_end is None or goal[1] is None or (goal[1] >= cur_loc and goal[1] <= cur_end) or (goal[1] <= cur_loc and goal[1] >= cur_end):
                    if goal[1] is not None:
                        cost += abs(cur_loc - goal[1])
                        cur_loc = goal[1]
                        dest.append(cur_loc)
                    
                    if goal[2] is not None:
                        
                        if cur_end is None:
                            cur_end = goal[2]
                        else:
                            if (goal[2] >= cur_loc and goal[2] <= cur_end) or (goal[2] <= cur_loc and goal[2] >= cur_end):
                                cost += abs(cur_loc - goal[2])
                                cur_loc = goal[2]
                                dest.append(cur_loc)
                                
                            elif abs(goal[2]-cur_loc) <= abs(cur_end-cur_loc):
                                cost += abs(cur_loc - goal[2])
                                cur_loc = goal[2]
                                dest.append(cur_loc)
                                
                            else: # finish cur_end and update the new cur_end to goal[2]
                                cost += abs(cur_loc - cur_end)
                                cur_loc = cur_end
                                cur_end = goal[2]
                                dest.append(cur_loc)
                
                else:
                    if cur_end is not None:
                        cost += abs(cur_loc - cur_end)
                        cur_loc = cur_end
                        dest.append(cur_loc)
                    
                    if goal[1] is not None:
                        cost += abs(cur_loc - goal[1])
                        cur_loc = goal[1]
                        dest.append(cur_loc)
                    
                    if goal[2] is not None:
                        cur_end = goal[2]
                    else:
                        cur_end = None
        if cur_end is not None:
            cost += abs(cur_loc - cur_end)
            cur_loc = cur_end
            dest.append(cur_loc)
        
        gt2cost.append(
            (dest, goal_traj, cost)
        )
        if cost < min_cost:
            min_cost = cost
            best_goal_traj = goal_traj
            best_dest = dest
    return best_goal_traj, best_dest, gt2cost

def check_manipulation_dist(s1, s2, dist_min):
    obj2loc_init = {}
    obj2loc_targ = {}
    for loc, box in enumerate(s1.boxes):
        for obj in box.content:
            obj2loc_init[obj.nickname()] = loc

    for loc, box in enumerate(s2.boxes):
        for obj in box.content:
            obj2loc_targ[obj.nickname()] = loc
    
    for obj, loc in obj2loc_targ.items():
        if abs(loc - obj2loc_init[obj]) >= dist_min:
            return True

    return False

class Entity:
    color = ""
    size = ""
    etype = None
    idx = None
    
    def __init__(self, etype, idx, color='', size=''):
        self.idx = idx
        self.color = color
        self.size = size
        self.etype = etype

    def name(self):
        return " ".join(
            [x for x in [self.size, self.color, self.etype] if x!='']
        )

    def nickname(self):
        return self.etype

    def signature(self):
        return " ".join([self.size, self.color, self.etype])

    def __eq__(self, other):
        return self.nickname() == other.nickname()


class Box(collections.MutableSequence):

    def __init__(self, idx, name, max_items, ignore_max_limit=True):
        self.idx = idx
        self.name = name + " " + str(idx)
        self.max_items = max_items
        self.ignore_max_limit = ignore_max_limit
        self.content = list()

    def __len__(self): return len(self.content)

    def __getitem__(self, i): return self.content[i]

    def __delitem__(self, i): 
        del self.content[i]

    def full(self): # deprecated
        return False
    
    def remove(self, value):
        '''S.remove(value) -- remove first occurrence of value.
           Raise ValueError if the value is not present.
        '''
        eq_idx = [i for (i, x) in enumerate(self.content) if x.signature()==value.signature()]
        del self[eq_idx[0]]

    def __setitem__(self, i, v):
        self.content[i] = v

    def insert(self, i, v):
        if len(self.content) == self.max_items and not self.ignore_max_limit:
            raise ValueError
        
        self.content.insert(i, v)

    def __str__(self):
        return str(self.name)

    def __eq__(self, o):
        return sorted([x.nickname() for x in self.content]) == sorted([x.nickname() for x in o.content])


class WorldState:
    """
    A class representing a world state.
    """
    
    def __init__(
        self,
        boxes,
        all_objects,
        max_items_per_box,
        expected_num_items_per_box,
        contents=None,
        zero_shot=False,
    ):
        """Initialize WorldState.
        """
        self.boxes = boxes
        self.all_objects = all_objects
        self.num_boxes = len(boxes)
        self.max_items_per_box = max_items_per_box
        self.expected_num_items_per_box = expected_num_items_per_box
        self.zero_shot = zero_shot

        if contents is not None:
            for i, b in enumerate(contents):
                for c in b:
                    if c not in self.all_objects:
                        raise KeyError(f"{c} is not a valid object!")
                if self.max_items_per_box > 0 and len(b) > self.max_items_per_box:
                    raise ValueError(
                        f"Attempted to add more than MAX_ITEMS_PER_BOX \
                         (={self.max_items_per_box}) items to box #{i}"
                    )
                self.void.difference_update(b)
                self.boxes[i].update(b)

    def update_box(self, box, content, operation):
        assert operation in ['in', 'out']

        if operation == 'in':
            self.boxes[box].append(content)
        
        if operation == 'out':
            self.boxes[box].remove(content)

    def permute(self, obj_fixed=None):
        
        def box_has_obj(box, obj_name):
            return any([x.nickname() in obj_name for x in box.content])

        def is_diff(boxes1, boxes2):
            for b1, b2 in zip(boxes1, boxes2):
                if b1 != b2:
                    return True
            return False

        boxes_idx = list(
                          range(len(self.boxes))
                        )
        boxes_hist = copy.deepcopy(self.boxes)

        while not is_diff(boxes_hist, self.boxes):
            if random.uniform(0, 1) > 0.5:
                # large flip
                if obj_fixed is None:
                    boxes_idx_swap = boxes_idx
                else:
                    boxes_idx_swap = [i for i in boxes_idx if not box_has_obj(self.boxes[i], obj_fixed)]
                if len(boxes_idx_swap) < 2: continue
                swap_pos = random.sample(boxes_idx_swap, 2)
                swap_dict = {swap_pos[0]: swap_pos[1], swap_pos[1]: swap_pos[0]}
                new_boxes = [self.boxes[swap_dict[i]] if i in swap_dict.keys() else self.boxes[i] for i in range(len(self.boxes))]
                self.boxes = new_boxes
            else:
                # small flip
                for _ in range(random.choice(list(range(1, 4)))):
                    i = random.choice(boxes_idx)
                    
                    if obj_fixed is None:
                        content = self.boxes[i].content
                    else:
                        content = [x for x in self.boxes[i].content if not x.nickname() in obj_fixed]
                    if len(content) == 0:
                        continue
                    obj = random.choice(content)
                    to_ins_idx = [j for j, box in enumerate(self.boxes) if len(box.content)<box.max_items]
                    j = random.choice(to_ins_idx)
                    self.update_box(j, obj, 'in')
                    self.update_box(i, obj, 'out')

    @staticmethod
    def sample_initial_world_state(
        boxes,
        all_objects,
        max_items_per_box,
        expected_num_items_per_box,
        zero_shot=False,
    ):
        num_boxes = len(boxes)
        
        s = WorldState(
            boxes,
            all_objects,
            max_items_per_box,
            expected_num_items_per_box,
            zero_shot=zero_shot,
        )

        num_items = poisson(expected_num_items_per_box, num_boxes)

        while sum(num_items) > len(all_objects):
            num_items = poisson(expected_num_items_per_box, num_boxes)

        if max_items_per_box > 0:
            num_items = np.minimum(num_items, [max_items_per_box])

        for box, n in enumerate(num_items):
            for obj in random.sample(all_objects, n):
                s.update_box(
                    box, obj, 'in'          
                )
        
        return s

    def __eq__(self, o):
        for box1, box2 in zip(self.boxes, o.boxes):
            if box1 != box2:
                return False

        return True

    def __hash__(self):
        sub_hashes = []
        for box in self.boxes:
            sub_hashes.append(hash(tuple(box)))

        return hash(tuple(sub_hashes))


class Action:

    def __init__(self, atype, direct=None, step=None, obj=None):
        assert atype in ['move', 'put', 'grab']
        self.atype = atype
        if atype == 'move':
            assert direct is not None and step is not None
        self.direct = direct
        self.step = step

        if atype in ['put', 'grab']:
            assert obj is not None
        self.obj = obj
    
    @property
    def signature(self):
        return str(self.atype) + str(self.direct) + str(self.obj.nickname() if self.obj is not None else None)

    def __eq__(self, o):
        if self.atype != o.atype:
            return False
        
        if self.direct is not None and self.direct != o.direct:
            return False

        if self.obj is not None and self.obj.nickname() != o.obj.nickname():
            return False
        
        return True

class Agent:
    location = None
    objects = []

    def __init__(self, location=None, objects=[]):
        self.location = location
        self.objects = []

    def random_action(self, legal_actions):
        return random.choice(legal_actions)
    
    def edit_backpack(self, obj, put_or_grab):
        assert put_or_grab in ['put', 'grab']
        if put_or_grab == 'put':
            self.objects.remove(obj)
        else:
            self.objects.append(obj)
    
class Environment:
    def __init__(self, agent, state):
        self.agent = agent
        self.state = state
        self.gallery = self.get_gallery()

    def get_gallery(self):
        gallery = []
        gallery += self.agent.objects
        for box in self.state.boxes:
            gallery += box.content
        return gallery

    def get_legal_actions(self, prev):
        acts_ok = []
        # get legal move 

        for obj in self.state.boxes[self.agent.location]:
            acts_ok.append(
                Action('grab', obj=obj)
            )

        for obj in self.agent.objects:
            if not self.state.boxes[self.agent.location].full():
                acts_ok.append(
                    Action('put', obj=obj)
                )

        if self.agent.location > 0:
            acts_ok.append(
                Action('move', direct='left', step=1)
            )
        
        if self.agent.location < self.state.num_boxes - 1:
            acts_ok.append(
                Action('move', direct='right', step=1)
            )
            
        return acts_ok
    def check_legal_action(self, action, ignore_max_limit=False):
        # get legal move 

        if action.atype == 'grab':
            return action.obj in self.state.boxes[self.agent.location]
        
        elif action.atype == 'put':
            if ignore_max_limit:
                return action.obj in self.agent.objects
            else:
                return action.obj in self.agent.objects and not self.state.boxes[self.agent.location].full()

        elif action.atype == 'move':
            if action.direct == 'left':
                return self.agent.location - action.step >= 0
            else:
                return self.agent.location + action.step < len(self.state.boxes)

        else:
            assert False

    def sample_traj(self, rollin_steps, num_operations, lesson_learned=None, op_dist=None):
        '''
        lesson_learned: {prefix: [hopeless actions]}
        '''
        action_traj = []
        prev_action = None
        curr_loc = copy.deepcopy(self.agent.location)
        init_state = copy.deepcopy(self.state)
        rollin_state = copy.deepcopy(self.state)
        is_deadend = False
        for t in range(int(rollin_steps+num_operations)):
            legal_actions = self.get_legal_actions(prev=prev_action)
            if lesson_learned is not None:
                prefix_ops_signature = tuple([x.signature for x in action_traj])
                legal_actions = [op for op in legal_actions if op.signature not in lesson_learned[prefix_ops_signature]]
                if len(legal_actions) == 0: 
                    is_deadend = True
                    break
            action = self.agent.random_action(legal_actions)
            prev_action = action
            if action.atype == 'move':
                self.move_agent(action.direct, action.step)
            else:
                self.agent.edit_backpack(action.obj, action.atype)
                self.state.update_box(
                    self.agent.location,
                    action.obj, 
                    'in' if action.atype=='put' else 'out'
                )
            action_traj.append(action)
            if t == rollin_steps-1:
                curr_loc = copy.deepcopy(self.agent.location)
                rollin_state = copy.deepcopy(self.state)
        if op_dist is not None:
            if not check_manipulation_dist(init_state, self.state, op_dist):
                is_deadend = True
        return self.state, curr_loc, rollin_state, action_traj, is_deadend

    def execute_operations(self, action_sequence, ignore_max_limit=False):
        executable = 0
        if len(action_sequence) == 0:
            return self.state, 0.
        for op in action_sequence:
            # check if op is legal
            if op is None or not self.check_legal_action(op, ignore_max_limit=ignore_max_limit):
                continue
            executable += 1
            if op.atype == 'move':
                self.move_agent(op.direct, op.step)
            else:
                self.agent.edit_backpack(op.obj, op.atype)
                self.state.update_box(
                    self.agent.location,
                    op.obj, 
                    'in' if op.atype=='put' else 'out'
                )
            
        return copy.deepcopy(self.state), executable/len(action_sequence)

    def move_agent(self, direct, step):
        assert direct in ['left', 'right']
        
        if direct == 'left':
            self.agent.location = self.agent.location - step
        else:
            self.agent.location = self.agent.location + step
        
        assert self.agent.location in list(range(self.state.num_boxes))

    def flip(self, obj_fixed=None):
        self.state.permute(obj_fixed=obj_fixed)
        if obj_fixed is None:
            self.agent.location = random.choice(list(range(self.state.num_boxes)))

    @staticmethod
    def texify_ops(action, lexicon_map):
        assert action.atype in ['move', 'put', 'grab']
        if action.atype == 'move':
            if action.step > 1:
                return lexicon_map['move_n'].format(direction=action.direct, nstep=action.step)
            else:
                return lexicon_map['move'].format(direction=action.direct)
        else:
            return lexicon_map[action.atype].format(content=action.obj.nickname())
    
    @staticmethod
    def texify_agent_behavior(action_sequence, lexicon_map, agent_location=None, append_eos=False):
        text = ''
        for op in action_sequence:
            if agent_location is not None: # for "operations applied"
                text += " You " + Environment.texify_ops(op, lexicon_map)
            else:
                text += " " + Environment.texify_ops(op, lexicon_map)
        if append_eos:
            text += " Terminate."
        return text

    @staticmethod
    def texify_state(state, is_initial, lex_variant=False):
        text = ''

        for box in state.boxes:
            text += f"{box.name} "
            for j, obj in enumerate(box.content):
                if j > 0: 
                    text += ', '
                else:
                    if lex_variant:
                        text += random.choice(_CONTAINS_VERB) + " "
                    else:
                        text += "contains "
                text += f"a {obj.name() if is_initial else obj.nickname()}"
            if len(box.content) == 0: 
                if lex_variant:
                    text += random.choice(_EMPTY_EXPR)
                else:
                    text += "is empty"
            text += '. '
        return text

    @staticmethod
    def textify_location(state, agent_loc, lex_variant=False):
        text = "A sequence of containers are ordered from left to right as follows: "
        for i, box in enumerate(state.boxes):
            if i > 0: text += ', '
            text += box.name
        text += '. '
        text += f"You are at the {state.boxes[agent_loc].name}."
        return text

    @staticmethod
    def texify_world(state, objects, is_initial, lex_variant=False):
        text = ''
        obj_in_boxes = []

        for box in state.boxes:
            text += f"{box.name} "
            for j, obj in enumerate(box.content):
                obj_in_boxes.append(obj)
                if j > 0: 
                    text += ', '
                else:
                    if lex_variant:
                        text += random.choice(_CONTAINS_VERB) + " "
                    else:
                        text += "contains "
                text += f"a {obj.name() if is_initial else obj.nickname()}"
            if len(box.content) == 0: 
                if lex_variant:
                    text += random.choice(_EMPTY_EXPR)
                else:
                    text += "is empty"
            text += '. '
        
        obj_in_agent = [x for x in objects if x not in obj_in_boxes]
        if len(obj_in_agent) == 0:
            text += "You don't hold anything"
        else:
            text += "You hold "
        
        for k, obj in enumerate(obj_in_agent):
            if k > 0:
                text += ", "
            text += obj.nickname()

        text += "."
        
        return text
        
    
    @staticmethod
    def merge_ops(action_sequence):
        if len(action_sequence) <= 1: return action_sequence
        short_action_sequence = []
        prev_action = None
        for i, ops in enumerate(action_sequence):
            if i > 0:
                if ops.atype == 'move' and ops.direct == prev_action.direct:
                    prev_action.step += ops.step # merge same direct
                else:
                    short_action_sequence.append(prev_action) # 
                    prev_action = ops
            else:
                prev_action = ops

        short_action_sequence.append(prev_action) # 
        return short_action_sequence

    @staticmethod
    def list_all_objects(state):
        objs = []
        for box in state.boxes:
            objs += box.content
        return (objs)

    @staticmethod
    def infer_agent_objects(state, all_objects):
        obj_in_boxes = []
        for box in state.boxes:
            obj_in_boxes += box.content
        
        return [x for x in all_objects if x not in obj_in_boxes]

    def parse_actions(self, text):
        def _ptrn_match(_text):
            move = re.search(r'move ((?:right|left))(?: for (\d+) steps)?', _text, flags=re.IGNORECASE)
            if move is not None:
                return Action(
                    'move', direct=move[1], 
                    step=(1 if move[2] is None else int(move[2]))
                )

            put = re.search(r'put ([a-zA-Z0-9]+)', _text, flags=re.IGNORECASE)
            if put is not None:
                name = put[1]
                if len(self.gallery) == 0:
                    return None
                slct_name = fuzzy_str_match(name, [x.nickname() for x in self.gallery])
                obj_slct = [x for x in self.gallery if x.nickname()==slct_name][0]
                return Action(
                    'put', obj=obj_slct
                )
            
            grab = re.search(r'grab ([a-zA-Z0-9]+)', _text, flags=re.IGNORECASE)
            if grab is not None:
                name = grab[1]
                
                if len(self.gallery) == 0:
                    return None
                slct_name = fuzzy_str_match(name, [x.nickname() for x in self.gallery])
                obj_slct = [x for x in self.gallery if x.nickname()==slct_name][0]
                return Action(
                    'grab', obj=obj_slct
                )

            return None
        action_sequence = [_ptrn_match(x) for x in text.split(". ")]
        return action_sequence


def fuzzy_str_match(text, candidates):
    best_cand = None
    max_score = -1000
    for cand in candidates:
        sim = Levenshtein.ratio(cand, text)
        if max_score < sim:
            max_score = sim
            best_cand = cand

    assert best_cand is not None
    return best_cand

