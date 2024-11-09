# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


Nbox = 5

_TASKS = {
    "policy": {
            "direct": {
                "task_type": 'cls',
                "label_size": 4,
                "reweight": True
            },
            "obj_ctrl": {
                'task_type': 'rank',
                'dense': False,
            }
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
for i in range(5):
    _TASKS['ndest'][f"goal_{i}"] = {
            "task_type": 'cls',
            "label_size": 11,
            "reweight": True
    }



_TASKS['world'] = {
    "agent_content": {
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
    "agent_content":{
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
    "aloc": {
        "task_type": "cls",
        "label_size": 5,
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
