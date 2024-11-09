# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from probe_model import _IGNORE_INDEX
from task_info import _TASKS
from utils import calc_label_wght


class Metrics:
    def __init__(self):
        super().__init__()

    def record(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError


class LegalMetric(Metrics):
    def __init__(self):
        self.metrics = {}
        for k in list(_TASKS['legal'].keys()):
            self.metrics[f'{k}_hit'] = []

    @staticmethod
    def proc_k(k):
        return k

    def make_rand_pred(self, labels):
        lsz = 2
        wght = calc_label_wght(
                [x for lab in labels.tolist() for x in lab if x != _IGNORE_INDEX],
                lsz
            )
        return [np.random.choice(
                        lsz, 
                        len(labels[0]), 
                        p=wght
                    ) for _ in range(len(labels))]

    def is_correct(self, pred, targ):
        pred = [x for x, y in zip(pred, targ) if y != _IGNORE_INDEX]
        targ = [x for x in targ if x != _IGNORE_INDEX]
        return pred == targ

    def record(self, preds, labels):

        for k in list(labels.keys()):
            assert len(labels[k]) == len(preds[k])
            if 'content' in k:
                self.metrics[f'{k}_hit'] += [1. if self.is_correct(p, q) else 0. for p, q in zip(preds[k].cpu(), labels[k])]
            else:
                self.metrics[f'{k}_hit'] += [1. if (p == q).all() else 0. for p, q in zip(preds[k].cpu(), labels[k])]

    def reset(self):
        for k in self.metrics:
            self.metrics[k] = []

    def log(self):
        msge = ''
        for k in self.metrics:
            msge += f"{self.proc_k(k)}: {np.mean(self.metrics[k]):.4f}; "
        msge += f"Avg: {np.mean([np.mean(x) for x in list(self.metrics.values())]):.4f}"
        return msge


STATE_TO_METRIC = {
    'legal': LegalMetric,
}
