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

_OPERATIONS_DICT = {
    "move": "go {direction}.",
    "move_n": "go {direction} for {nstep} steps.",
    "put": "drop {content} in the room.",
    "grab": "take {content} from the room."
}

_CONTAINS_TMPL = [
    "{0} contains {1}.", 
    'In {0}, you can see {1}.', 
    'In {0}, you can find {1}.', 
    "{0} has {1} in it."
    ]

_EMPTY_TMPL = [
    '{} is empty.', 
    '{} contains nothing.', 
    '{} has nothing.', 
    '{} is vacant.']


'''
   N
W     E
   S
'''
_ROOM_LOC_TMPL = """
{0} in the northwest connects east to {1} and south to {3}. \
{1} links further east to {2} and south to {4}. \
{2} connects south to {5}. \
{3} leads east to {4}. \
{4} connects east to {5}.\
"""


def calc_cost(cur_loc, dest):
    '''
    dest: list of aloc
    '''
    def _dist(loc1, loc2):
        return abs(loc1['row']-loc2['row']) + abs(loc1['col']-loc2['col'])

    cost = 0.
    last_loc = copy.deepcopy(cur_loc)
    for next_loc in dest:
        cost += _dist(last_loc, next_loc)
        last_loc = next_loc
    return cost


def get_dest(ops, cur_loc):
    aloc = copy.deepcopy(cur_loc)
    dest = []
    for op in ops:
        if op.atype == 'move':
            if op.direct == 'n':
                aloc['row'] -= 1
            elif op.direct == 's':
                aloc['row'] += 1
            elif op.direct == 'w':
                aloc['col'] -= 1
            elif op.direct == 'e':
                aloc['col'] += 1
            else:
                assert False
        elif op.atype in ['grab', 'put']:
            dest.append(aloc)
        else:
            assert ValueError
    return dest


def get_subgoal(sinit, sgoal):
    obj_to_put = {}
    obj_to_grab = {}
    for row in range(len(sinit.boxes)):
        for col, (b1, b2) in enumerate(zip(
                sinit.boxes[row], 
                sgoal.boxes[row])
        ):
            b1_objs = [x.nickname() for x in b1.content]
            b2_objs = [x.nickname() for x in b2.content]
            for obj in b1_objs:
                if obj not in b2_objs:
                    obj_to_grab[obj] = {"row": row, "col": col}
            for obj in b2_objs:
                if obj not in b1_objs:
                    obj_to_put[obj] = {"row": row, "col": col}

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


def leq(p1, p2):
    return (p1['row'] + p1['col']) <= (p2['row'] + p2['col'])


def in_between(p, s, e):
    return abs_dist(s, p) + abs_dist(e, p) <= abs_dist(s, e)


def abs_dist(p1, p2):
    return abs(p1['row']-p2['row']) + abs(p1['col']-p2['col'])


def get_optimal_plan(subgoals, aloc, debug=False):
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
                    cost += abs_dist(cur_loc, goal[1])
                    cur_loc = goal[1]

                if goal[2] is not None:
                    cur_end = goal[2]

                continue
            else:
                # test if this dest and end in between cur_loc & cur_end
                #
                if cur_end is None or goal[1] is None or in_between(goal[1], cur_loc, cur_end):
                    if debug:
                        print(cur_loc, goal[1], goal[2], 'yes')
                    if goal[1] is not None:
                        cost += abs_dist(cur_loc, goal[1])
                        cur_loc = goal[1]
                        dest.append(cur_loc)

                    if goal[2] is not None:

                        if cur_end is None:
                            cur_end = goal[2]
                        else:
                            if in_between(goal[2], cur_loc, cur_end):
                                cost += abs_dist(cur_loc, goal[2])
                                cur_loc = goal[2]
                                dest.append(cur_loc)

                            elif abs_dist(goal[2], cur_loc) <= abs_dist(cur_end, cur_loc): # 
                                cost += abs_dist(cur_loc, goal[2])
                                cur_loc = goal[2]
                                dest.append(cur_loc)

                            else:  # finish cur_end and update the new cur_end to goal[2]
                                cost += abs_dist(cur_loc, cur_end)
                                cur_loc = cur_end
                                cur_end = goal[2]
                                dest.append(cur_loc)

                else:
                    if cur_end is not None:
                        cost += abs_dist(cur_loc, cur_end)
                        cur_loc = cur_end
                        dest.append(cur_loc)

                    if goal[1] is not None:
                        cost += abs_dist(cur_loc, goal[1])
                        cur_loc = goal[1]
                        dest.append(cur_loc)

                    if goal[2] is not None:
                        cur_end = goal[2]
                    else:
                        cur_end = None
        if cur_end is not None:
            cost += abs_dist(cur_loc, cur_end)
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
    for loc, box in enumerate([x for boxes in s1.boxes for x in boxes]):
        for obj in box.content:
            obj2loc_init[obj.nickname()] = loc

    for loc, box in enumerate([x for boxes in s2.boxes for x in boxes]):
        for obj in box.content:
            obj2loc_targ[obj.nickname()] = loc

    for obj, loc in obj2loc_targ.items():
        if abs(loc - obj2loc_init[obj]) >= dist_min:
            return True

    return False


class Entity:
    cut = ""
    cook = ""
    etype = None
    idx = None

    def __init__(self, etype, idx, cut='', cook=''):
        self.idx = idx
        self.cut = cut
        self.cook = cook
        self.etype = etype

    def name(self):
        return " ".join(
            [x for x in [self.cook, self.cut, self.etype] if x != '']
        )

    def nickname(self):
        return self.etype

    def signature(self):
        return " ".join([self.cut, self.cook, self.etype])

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

    def full(self):  # deprecated
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

        Args:
            all_objects (list): List of all possible objects.
            num_boxes (int): Number of boxes.
            max_items_per_box (int): Maximum number of objects per box.
            expected_num_items_per_box (int): Expected number of objects per box.
            contents (dict[list], optional): Initial contents of all boxes. Defaults to None.
            zero_shot (bool, optional): Whether to use zero-shot/in-context learning data format. Defaults to False.

        Raises:
            KeyError: Raised if invalid object is added to box.
            ValueError: Raised if too many objects are added to box.
        """
        self.boxes = boxes
        self.all_objects = all_objects
        self.num_boxes = len(boxes) * len(boxes[0])
        self.max_items_per_box = max_items_per_box
        self.expected_num_items_per_box = expected_num_items_per_box
        self.zero_shot = zero_shot

    def __str__(self):
        ret = "Boxes:" + str(self.boxes) + "\n"
        ret += "Void:" + str(self.void)
        return ret

    def remove_from_box(self, row, col, content):
        """ Remove content from Box #box.

        Args:
            box (int): Box number.
            row: row idx
            col: col idx 
            content (set): Set of objects to be removed. 

        Raises:
            KeyError: Raised if non-exstent object is removed.
        """
        if isinstance(content, (list, set)):
            for c in content:
                if c not in self.boxes[row][col]:
                    raise KeyError(f"{c} not in box #{row,col}")
            self.boxes[row][col].difference_update(content)
            self.void.update(content)
        else:
            # throws KeyError if content is not in box
            self.boxes[row][col].remove(content)
            self.void.add(content)

    def update_box(self, row, col, content, operation):
        assert operation in ['in', 'out']

        if operation == 'in':
            self.boxes[row][col].append(content)
        
        if operation == 'out':
            self.boxes[row][col].remove(content)

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
                if len(boxes_idx_swap) < 2:
                    continue
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
                    to_ins_idx = [j for j, box in enumerate(self.boxes) if len(box.content) < box.max_items]
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
        nrow = len(boxes)
        ncol = len(boxes[0])
        num_boxes = int(nrow * ncol)

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
            row = box // 3  # WARNING: hard code
            col = box % 3
            for obj in random.sample(all_objects, n):
                s.update_box(
                    row, col, obj, 'in'
                )

        return s

    def __eq__(self, o):
        for box1, box2 in zip(
            [x for boxes in self.boxes for x in boxes], 
            [x for boxes in o.boxes for x in boxes]):
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
        return str(self.atype) + \
               str(self.direct if self.direct is not None else None) + \
               str(self.obj.nickname() if self.obj is not None else None)

    def __eq__(self, o):
        if self.atype != o.atype:
            return False

        if self.direct is not None and not self.direct == o.direct:
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
        for i in range(len(self.state.boxes)):
            for box in self.state.boxes[i]:
                gallery += box.content
        return gallery

    def get_legal_actions(self, prev_acts):

        def get_on_traj(acts_hist):
            on_traj = []
            for x in reversed(acts_hist):
                if x.atype == 'grab':
                    break
                else:
                    if x.atype == 'move':
                        on_traj.append(x.direct)
            return on_traj

        prev = prev_acts[-1] if len(prev_acts)>0 else None
        acts_ok = []
        # get legal move 
        aloc = self.agent.location
        for obj in self.state.boxes[aloc['row']][aloc['col']]:
            if prev is None or prev.atype != 'put' or prev.obj.name != obj.name:
                acts_ok.append(
                    Action('grab', obj=obj)
                )

        for obj in self.agent.objects:
            if (prev is None or prev.atype != 'grab' or prev.obj.name != obj.name) and not self.state.boxes[aloc['row']][aloc['col']].full():
                acts_ok.append(
                    Action('put', obj=obj)
                )
        #       N
        #   W       E
        #       S
        on_traj = get_on_traj(prev_acts)
        if aloc['row'] > 0:
            if prev is None or 's' not in on_traj:
                acts_ok.append(
                    Action('move', direct='n', step=1)
                )
        
        if aloc['row'] < len(self.state.boxes) - 1:
            if prev is None or 'n' not in on_traj:
                acts_ok.append(
                    Action('move', direct='s', step=1)
                )

        if aloc['col'] < len(self.state.boxes[0]) - 1:
            if prev is None or 'w' not in on_traj:
                acts_ok.append(
                    Action('move', direct='e', step=1)
                )

        if aloc['col'] > 0:
            if prev is None or 'e' not in on_traj:
                acts_ok.append(
                    Action('move', direct='w', step=1)
                )
        return acts_ok

    def check_legal_action(self, action, ignore_max_limit=False):
        # get legal move 

        aloc = self.agent.location
        if action.atype == 'grab':
            return action.obj in self.state.boxes[aloc['row']][aloc['col']]
        
        elif action.atype == 'put':
            if ignore_max_limit:
                return action.obj in self.agent.objects
            else:
                return action.obj in self.agent.objects and not self.state.boxes[aloc['row']][aloc['col']].full()

        elif action.atype == 'move':
            if action.direct == 'n':
                return aloc['row'] - action.step >= 0
            elif action.direct == 's':
                return aloc['row'] + action.step < len(self.state.boxes)
            elif action.direct == 'w':
                return aloc['col'] - action.step >= 0
            elif action.direct == 'e':
                return aloc['col'] + action.step < len(self.state.boxes[0])
            else:
                raise ValueError

        else:
            assert False

    def sample_traj(self, rollin_steps, num_operations, lesson_learned=None, op_dist=None):
        '''
        lesson_learned: {prefix: [hopeless actions]}
        '''
        action_traj = []

        curr_loc = copy.deepcopy(self.agent.location)
        init_state = copy.deepcopy(self.state)
        rollin_state = copy.deepcopy(self.state)
        is_deadend = False
        for t in range(int(rollin_steps+num_operations)):
            legal_actions = self.get_legal_actions(action_traj)
            if lesson_learned is not None:
                prefix_ops_signature = tuple([x.signature for x in action_traj])
                legal_actions = [op for op in legal_actions if op.signature not in lesson_learned[prefix_ops_signature]]
            if len(legal_actions) == 0:
                is_deadend = True
                break
            action = self.agent.random_action(legal_actions)
            
            if action.atype == 'move':
                self.move_agent(action.direct, action.step)
            else:
                self.agent.edit_backpack(action.obj, action.atype)
                self.state.update_box(
                    self.agent.location['row'],
                    self.agent.location['col'],
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
                    self.agent.location['row'],
                    self.agent.location['col'],
                    op.obj, 
                    'in' if op.atype=='put' else 'out'
                )
            
        return copy.deepcopy(self.state), executable/len(action_sequence)

    def move_agent(self, direct, step):
        assert direct in ['n', 's', 'w', 'e']
        
        if direct == 'n':
            self.agent.location['row'] = self.agent.location['row'] - step
        elif direct == 's':
            self.agent.location['row'] = self.agent.location['row'] + step
        elif direct == 'w':
            self.agent.location['col'] = self.agent.location['col'] - step
        elif direct == 'e':
            self.agent.location['col'] = self.agent.location['col'] + step
        else:
            raise ValueError
        assert self.agent.location['row'] in list(range(len(self.state.boxes)))
        if not self.agent.location['col'] in list(range(
                len(self.state.boxes[0]))):
            print(self.agent.location['col'])
        assert self.agent.location['col'] in list(range(
                len(self.state.boxes[0])
            ))

    def flip(self, obj_fixed=None):
        self.state.permute(obj_fixed=obj_fixed)
        if obj_fixed is None:
            self.agent.location = random.choice(list(range(self.state.num_boxes)))

    @staticmethod
    def texify_ops(action, lexicon_map):
        assert action.atype in ['move', 'put', 'grab']
        direct_map = {
            "s": "south",
            "n": "north",
            "e": "east",
            "w": "west"
        }
        if action.atype == 'move':
            if action.step > 1:
                return lexicon_map['move_n'].format(direction=direct_map[action.direct], nstep=action.step)
            else:
                return lexicon_map['move'].format(direction=direct_map[action.direct])
        else:
            return lexicon_map[action.atype].format(content=action.obj.nickname())
    
    @staticmethod
    def texify_agent_behavior(action_sequence, lexicon_map, agent_location=None, append_eos=False):
        if agent_location is None: 
            text = ''
        else:  # for "operations applied"
            if len(action_sequence) > 0:
                text = " You"
            else:
                text = ""
        for op in action_sequence:
            text += " " + Environment.texify_ops(op, lexicon_map)
        if append_eos:
            text += " Terminate."
        return text

    @staticmethod
    def texify_state(state, is_initial, lex_variant=False):
        text = ''

        for i in range(len(state.boxes)):
            for box in state.boxes[i]:
                if len(box.content) == 0:
                    text += random.choice(_EMPTY_TMPL).format(box.name) 
                else:
                    content_text = ', '.join([
                            obj.name() if is_initial else obj.nickname() for obj in box.content
                        ])
                    text += random.choice(_CONTAINS_TMPL).format(box.name, content_text)
                text += ' '
        return text.rstrip()

    @staticmethod
    def textify_location(state, agent_loc, lex_variant=False):
        text = "You open the map of TextWorld. "
        text += _ROOM_LOC_TMPL.format(
            *[x.name for boxes in state.boxes for x in boxes]
        )
        text += " "
        text += f"You are in {state.boxes[agent_loc['row']][agent_loc['col']].name}."
        return text

    @staticmethod
    def texify_world(state, objects, is_initial, lex_variant=False):
        text = ''
        obj_in_boxes = []

        for i in range(len(state.boxes)):
            for box in state.boxes[i]:
                if len(box.content) == 0:
                    text += random.choice(_EMPTY_TMPL).format(box.name) 
                else:
                    obj_in_boxes += box.content
                    content_text = ', '.join([
                            obj.name() if is_initial else obj.nickname() for obj in box.content
                        ])
                    text += random.choice(_CONTAINS_TMPL).format(box.name, content_text)
                    text += ' '
        
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
                    prev_action.step += ops.step  # merge same direct
                else:
                    short_action_sequence.append(prev_action) 
                    prev_action = ops
            else:
                prev_action = ops

        short_action_sequence.append(prev_action) 
        return short_action_sequence

    @staticmethod
    def list_all_objects(state):
        objs = []
        for i in range(len(state.boxes)):
            for box in state.boxes[i]:
                objs += box.content
        return (objs)

    @staticmethod
    def infer_agent_objects(state, all_objects):
        obj_in_boxes = []
        for row in range(len(state.boxes)):
            for box in state.boxes[row]:
                obj_in_boxes += box.content
        
        return [x for x in all_objects if x not in obj_in_boxes]

    def parse_actions(self, text):
        dir_map = {
            "north": "n",
            "west": "w",
            "south": "s",
            "east": "e"
        }
        
        def _ptrn_match(_text):
            move = re.search(r'go ((?:north|west|south|east))(?: for (\d+) steps)?', _text, flags=re.IGNORECASE)
            if move is not None:
                return Action(
                    'move', direct=dir_map[move[1]], 
                    step=(1 if move[2] is None else int(move[2]))
                )

            put = re.search(r'drop ([a-zA-Z0-9]+)', _text, flags=re.IGNORECASE)
            if put is not None:
                name = put[1]
                if len(self.gallery) == 0:
                    return None
                slct_name = fuzzy_str_match(name, [x.nickname() for x in self.gallery])
                obj_slct = [x for x in self.gallery if x.nickname() == slct_name][0]
                return Action(
                    'put', obj=obj_slct
                )
            
            grab = re.search(r'take ([a-zA-Z0-9]+)', _text, flags=re.IGNORECASE)
            if grab is not None:
                name = grab[1]
                
                if len(self.gallery) == 0:
                    return None
                slct_name = fuzzy_str_match(name, [x.nickname() for x in self.gallery])
                obj_slct = [x for x in self.gallery if x.nickname() == slct_name][0]
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
