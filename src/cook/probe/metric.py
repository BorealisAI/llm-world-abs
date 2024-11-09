# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from probe_model import _IGNORE_INDEX
from task_info import _TASKS
from sklearn.metrics import f1_score
import torch

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


class RawMetric(Metrics):
    def __init__(self):
        self.metrics = {}
        for k in list(_TASKS['raw'].keys()):
            self.metrics[f'{k}_hit'] = []

    @staticmethod
    def proc_k(k):
        return "_".join(k.split("_")[-3:-1])

    def record(self, preds, labels):
        np.random.seed(42)
        for k in list(labels.keys()):
            assert len(labels[k]) == len(preds[k])
            self.metrics[f'{k}_hit'] += [1. if (p == q).all() else 0. for p, q in zip(preds[k].cpu(), labels[k])]
            '''
            rand perf.
            rand_preds = np.random.choice(
                list(range(5)), len(labels[k]),# p=wght
            )
            self.metrics[f'{k}_hit'] += [1. if (p == q).all() else 0. for p, q in zip(rand_preds, labels[k])]
            '''

    def reset(self):
        for k in self.metrics:
            self.metrics[k] = []

    def log(self):
        msge = ''
        for k in self.metrics:
            msge += f"{self.proc_k(k)}: {np.mean(self.metrics[k]):.4f}; "
        msge += f"Avg: {np.mean([np.mean(x) for x in list(self.metrics.values())]):.4f}"
        return msge


class WorldMetric(Metrics):

    def __init__(self):
        
        self.metrics = {}
        for k in list(_TASKS['world'].keys()):
            self.metrics[f'{k}_hit'] = []

    @staticmethod
    def proc_k(k):
        return k.split("_")[-2]

    def make_rand_pred(self, labels):
        lsz = 2
        np.random.seed(42)
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

            self.metrics[f'{k}_hit'] += [
                1. if (p == q).all() else 0.
                for p, q in zip(preds[k].cpu(), labels[k])
            ]
            # calc random performance:
            # preds_rand = self.make_rand_pred(labels[k])
            # self.metrics[f'{k}_hit'] += [1. if self.is_correct(p, q.tolist()) else 0. for p, q in zip(preds_rand, labels[k])]

    def reset(self):
        for k in self.metrics:
            self.metrics[k] = []

    def log(self):
        msge = ''
        for k in self.metrics:
            msge += f"{self.proc_k(k)}: {np.mean(self.metrics[k]):.4f}; "
        msge += f"Avg: {np.mean([np.mean(x) for x in list(self.metrics.values())]):.4f}"
        return msge


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
        all_logits = [x for lab in labels for x in lab]
        np.random.seed(42)
        wght = calc_label_wght(
                [x for lab in labels.tolist() for x in lab if x!=_IGNORE_INDEX], 
                lsz
            )
        return [np.random.choice(
                        lsz, 
                        len(labels[0]), 
                        p=wght
                    ) for _ in range(len(labels))]

    def make_rand_loc_pred(self, labels, lsz=5):
        wght = calc_label_wght(labels, lsz)
        return np.random.choice(
            lsz,
            len(labels),
            p=wght
        )

    def is_correct(self, pred, targ):
        pred = [x for x,y in zip(pred, targ) if y!=_IGNORE_INDEX]
        targ = [x for x in targ if x != _IGNORE_INDEX]
        return pred==targ

    def record(self, preds, labels):

        for k in list(labels.keys()):
            assert len(labels[k]) == len(preds[k])
            if 'content' in k:
                '''
                rand perf:
                preds_rand = self.make_rand_pred(labels[k])
                self.metrics[f'{k}_hit'] += [1. if self.is_correct(p, q) else 0. for p, q in zip(preds_rand, labels[k])]
                '''
                self.metrics[f'{k}_hit'] += [1. if self.is_correct(p, q) else 0. for p, q in zip(preds[k].cpu(), labels[k])]
            elif 'row' in k:
                '''
                rand perf:
                preds_rand = self.make_rand_loc_pred(labels[k], 2)
                self.metrics[f'{k}_hit'] += [1. if (p == q).all() else 0. for p, q in zip(preds_rand, labels[k])]
                '''
                self.metrics[f'{k}_hit'] += [1. if (p == q).all() else 0. for p, q in zip(preds[k].cpu(), labels[k])]
            elif 'col' in k:
                '''
                rand perf:
                preds_rand = self.make_rand_loc_pred(labels[k], 3)
                self.metrics[f'{k}_hit'] += [1. if (p == q).all() else 0. for p, q in zip(preds_rand, labels[k])]
                '''
                self.metrics[f'{k}_hit'] += [1. if (p == q).all() else 0. for p, q in zip(preds[k].cpu(), labels[k])]
            elif 'overall' in k:
                '''
                rand perf:
                preds_rand = self.make_rand_loc_pred(labels[k], 6)
                self.metrics[f'{k}_hit'] += [1. if (p == q).all() else 0. for p, q in zip(preds_rand, labels[k])]
                '''
                self.metrics[f'{k}_hit'] += [1. if (p == q).all() else 0. for p, q in zip(preds[k].cpu(), labels[k])]

            else:
                '''
                rand perf:
                preds_rand = self.make_rand_loc_pred(labels[k], 6)
                self.metrics[f'{k}_hit'] += [1. if (p == q).all() else 0. for p, q in zip(preds_rand, labels[k])]
                '''
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

class NDestMetrics(Metrics):

    def __init__(self):
        self.preds_record = {}
        self.labels_record = {}
        for k in list(_TASKS['ndest'].keys()):
            self.preds_record[k] = []
            self.labels_record[k] = []

    @staticmethod
    def proc_k(k):
        return "".join(k.split("_")[1:3])[:2]

    def record(self, preds, labels):

        for k in list(labels.keys()):
            assert len(labels[k]) == len(preds[k])
            mask = labels[k].ne(_IGNORE_INDEX)
            pk = torch.masked_select(preds[k].cpu(), mask)
            qk = torch.masked_select(labels[k], mask)
            self.preds_record[k] += pk.tolist()
            self.labels_record[k] += qk.tolist()

    def reset(self):
        for k in self.preds_record:
            self.preds_record[k] = []
            self.labels_record[k] = []

    def make_rand_pred(self, labels, lsz=1):
        np.random.seed(42)
        wght = calc_label_wght(labels, lsz)
        return np.random.choice(
            lsz,
            len(labels),
            p=wght
        )

    def log(self):
        msge = ''
        all_f1 = []
        for k in list(_TASKS['ndest'].keys()):            
            if len(self.labels_record[k]) == 0 or 'goal_4' in k or 'goal_5' in k:
                continue
            f1 = np.mean(f1_score(
                self.labels_record[k],
                self.preds_record[k],
                average='macro'
            ))
            '''
            calc rand score:
            f1_rand = f1_score(
                self.labels_record[k],
                self.make_rand_pred(
                    self.labels_record[k],
                    lsz=(
                        _TASKS['ndest']['goal_0_row']['label_size'] if "row" in k else _TASKS['ndest']['goal_0_col']['label_size'])
                ),
                average="macro"
            )
            '''
            msge += f"{self.proc_k(k)}: {f1 * 100:.1f}; "
            all_f1.append(f1)
            
        msge += f"Avg: {np.mean(all_f1) * 100:.1f}"
        return msge

class PolicyMetrics(Metrics):
    obj_hits = []
    direct_hits = []
    direct_pred = []
    direct_labels = []

    def record(self, preds, labels):
        pobj = preds['obj_ctrl']  
        qobj = labels['obj_ctrl']  
        pobj = [x for x, y in zip(pobj, qobj) if y!=_IGNORE_INDEX]
        qobj = [x for x in qobj if x!=_IGNORE_INDEX]
        
        self.obj_hits += [1. if x == y else 0. for x, y in zip(pobj, qobj)]
        '''
        rand perf:
        pobj_rand = self.make_rand_pred(qobj, 5)
        self.obj_hits += [1. if x == y else 0. for x, y in zip(pobj_rand, qobj)]
        '''
        self.direct_pred += preds['direct'].tolist()
        self.direct_labels += labels['direct'].tolist()

    def make_rand_pred(self, labels, lsz=4):
        np.random.seed(42)
        wght = calc_label_wght(labels, lsz)

        return np.random.choice(
            lsz,
            len(labels),
            p=wght
        )

    def reset(self):
        self.obj_hits = []
        self.direct_pred = []
        self.direct_labels = []

    def log(self):
        '''
        calc random f1 score:
        f1_rand = np.mean(f1_score(
            self.direct_labels,
            self.make_rand_pred(self.direct_labels, lsz=6),
            average='macro'
        ))
        '''
        f1 = np.mean(f1_score(
            self.direct_labels,
            self.direct_pred,
            average='macro'
        ))
        msge = (
            f"Obj Acc:  {np.mean(self.obj_hits):.4f}; "
            f"Direct Acc:  {np.mean(f1):.4f}; "
        )
        return msge


STATE_TO_METRIC = {
    'raw': RawMetric,
    'world': WorldMetric,
    'world_dream': WorldMetric,
    'legal': LegalMetric,
    'ndest': NDestMetrics,
    'policy': PolicyMetrics
}