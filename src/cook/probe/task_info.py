# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

Nbox = 6

_TASKS = {
    "policy": {
            "direct": {
                "task_type": 'cls',
                "label_size": 6,
                "reweight": True
            },
            "obj_ctrl": {
                'task_type': 'rank',
                'dense': False,
            }
        }
}

_TASKS['nopt'] = {
    "nopt": {
        "task_type": 'cls',
        "label_size": 8,
        "reweight": True
    }
}

_TASKS['dist'] = {
    "goal_1": {
        "task_type": 'cls',
        "label_size": 10,
        "reweight": True
    }
}

_TASKS['dest'] = {}
for i in range(5):
    _TASKS['dest'][f"goal_{i}"] = {
            "task_type": 'cls',
            "label_size": 10,
            "reweight": True
    }

_TASKS['ndest'] = {}
for i in range(6):
    _TASKS['ndest'][f"goal_{i}_row"] = {
            "task_type": 'cls',
            "label_size": 5,
            "reweight": True
    }
    _TASKS['ndest'][f"goal_{i}_col"] = {
            "task_type": 'cls',
            "label_size": 7,
            "reweight": True
    }

_TASKS['qopt'] = {}
for i in range(1, Nbox):
    _TASKS['qopt'][f'nobj_{i}'] = {
        "task_type": 'cls',
        "label_size": 3,
        "reweight": True
    }

    _TASKS['qopt'][f'nobj_{-i}'] = {
        "task_type": 'cls',
        "label_size": 3,
        "reweight": True,
    }

_TASKS['ablation_rel'] = {}
for i in range(1, Nbox):
    _TASKS['ablation_rel'][f'nobj_{i}'] = {
        "task_type": 'cls',
        "label_size": 3,
        "reweight": True
    }
    _TASKS['ablation_rel'][f'nobj_{-i}'] = {
        "task_type": 'cls',
        "label_size": 3,
        "reweight": True
    }

_TASKS['world'] = {
    "agent_content":{
        "task_type": 'rank',
        "dense": True
    }
}
for i in range(Nbox):
    _TASKS['world'][f'box_content_{i}'] = {
        "task_type": 'rank',
        "dense": True
    }

_TASKS['diff_world'] = _TASKS['world']


_TASKS['world_dream'] = {
    "agent_content": {
        "task_type": 'rank',
        "dense": True,
        "reweight": False
    }
}
for i in range(Nbox):
    _TASKS['world_dream'][f'box_content_{i}'] = {
        "task_type": 'rank',
        "dense": True,
        "reweight": False
    }

_TASKS['raw'] = {}
for i in range(Nbox):
    _TASKS['raw'][f'box_name_{i}'] = {
        "task_type": 'rank',
        "dense": False
    }

_TASKS['legal'] = {
    "row": {
        "task_type": "cls",
        "label_size": 2,
        "reweight": True
    },
    "col": {
        "task_type": "cls",
        "label_size": 3,
        "reweight": True
    },
    "overall": {
        "task_type": "cls",
        "label_size": 6,
        "reweight": True
    },
    "box_content": {
        "task_type": "rank",
        "dense": True
    },
    "agent_content": {
        "task_type": "rank",
        "dense": True
    },
}
