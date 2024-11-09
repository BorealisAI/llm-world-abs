# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2023-present, Kenneth Li
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the othello-gpt (https://arxiv.org/pdf/2210.13382) implementation from https://github.com/likenneth/othello_world/blob/master/mingpt/probe_trainer.py by Kenneth Li which is licensed under the MIT License.
# You may obtain a copy of the License at 
# 
#   https://github.com/likenneth/othello_world/blob/master/LICENSE
# 
# 
#################################################################################### 

import os
import logging

from tqdm import tqdm
import numpy as np
import json
import torch
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from probe_model import _IGNORE_INDEX, _TASKS
import pandas
from pathlib import Path
from metric import STATE_TO_METRIC
logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

def pad_lmr(label_repr_bank, pred_data, task):
    '''
    label_repr_bank: {label: lmr}
    pred_data: {{f'cand{i}': str, "label": [int]}}
    '''
    try:
        max_ncand = max([
            len(list(pd[task].keys())) for pd in pred_data if task in pd.keys()
        ]) - 1
    except:
        return None

    llm_repr_dummy = torch.zeros_like(
        list(label_repr_bank.values())[0]
    )
    label_cand_lmr = []
    for pd in pred_data:
        label_cand_lmr.append(
            torch.stack(
                [
                    label_repr_bank[pd[task][f'cand{i}']]
                    if task in pd.keys() and f'cand{i}' in pd[task] and pd[task][f'cand{i}'] in label_repr_bank.keys()
                    else llm_repr_dummy
                    for i in range(max_ncand)
                ],
                dim=0
            )
        )
    return torch.stack(label_cand_lmr, dim=0)


def pad_labels(pred_data, pred_key, is_dense):
    '''
    case1: [int, int, ...]
    case2: [[int, ..., int], [int, ...]]
    '''

    def _concat_labels(labels, npad):
        return torch.tensor(labels + [_IGNORE_INDEX]*npad)

    if is_dense:
        max_nlabel = max([len(x[pred_key]['label']) for x in pred_data if pred_key in x.keys()])

        return torch.stack(
            [
              _concat_labels(
                  x[pred_key]['label'], 
                  max_nlabel - len(x[pred_key]['label'])
                ) if pred_key in x.keys() else torch.tensor([_IGNORE_INDEX]*max_nlabel)
                  for x in pred_data
            ]
        )
    else:

        return torch.tensor([x[pred_key]['label'] if pred_key in x.keys() else _IGNORE_INDEX for x in pred_data])


def collate_fn(features, state_type, label_repr_bank):
    '''
    {pred_key: {"cand{i}": torch.tensor}}
    output: {pred_key: {"options": [bsz, ncand, dim], "label": int}}
    '''
    task2sub = _TASKS[state_type]
    predicate_data = [x['predicate_data'][state_type] for x in features] 

    lmr_options = {}
    lmr_ctxt = torch.stack(
            [x['ft_lmr'] for x in features], dim=0
        )
    labels = {}
    for task, subs in task2sub.items():
        if subs['task_type'] == 'rank':
            # bsz x ncand x dim
            tmp = pad_lmr(label_repr_bank, predicate_data, task)
            if tmp is None:
                continue  # no such subtask in this batch
            else:
                lmr_options[task] = tmp
                labels[task] = pad_labels(predicate_data, task, subs['dense'])

        else:
            tmp = pad_labels(predicate_data, task, False)
            if tmp.eq(_IGNORE_INDEX).all():
                continue
            labels[task] = tmp

    ret = {
        "llm_repr_ctxt": lmr_ctxt,
        "llm_repr_options": lmr_options,
        "labels": labels
    }
    if "meta_info" in features[0].keys():
        ret['meta_info'] = {
            k: [x['meta_info'][k] for x in features] for k in list(features[0]['meta_info'].keys())
        }
    return ret


def proc_acc_metrics(metrics):
    return {
        "agent_loc": metrics['agent_loc'],
        "boxes_order": metrics['boxes_order'],
        "object_pos": np.mean([v for k, v in metrics.items() if "obj" in k])
    }


class Trainer:
    def __init__(self, state_type, model, train_dataset, test_dataset, label_repr, config):
        self.model = model
        self.state_type = state_type
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.label_repr = label_repr
        self.metric = STATE_TO_METRIC[self.state_type]()
        # take over whatever gpus are on the system
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            assert False
        # log something for plotting
        self.train_loss_cont = []
        self.test_loss_cont = []
        self.train_acc_cont = []
        self.test_acc_cont = []
        # would be a list of T-long, each is a lits of 60-long, for stratified accuracies
        self.train_strat_acc_cont = []
        self.test_strat_acc_cont = []
        Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)
        
    def save_traces(self, ):
        tbd = {
            "train_loss_cont": self.train_loss_cont, "test_loss_cont": self.test_loss_cont, 
            "train_acc_cont": self.train_acc_cont, "test_acc_cont": self.test_acc_cont, 
            "train_strat_acc_cont": self.train_strat_acc_cont, "test_strat_acc_cont": self.test_strat_acc_cont, 
        }
        with open(os.path.join(self.config.ckpt_path, "tensorboard.txt"), "w") as f:
            f.write(json.dumps(tbd) + "\n")

    def save_eval_records(self, eval_records, epc):
        with open(os.path.join(self.config.ckpt_path, f"eval-records-{epc}.jsonl"), "w") as f:
            pandas.DataFrame(eval_records).to_json(f, orient="records", lines=True)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if not os.path.exists(self.config.ckpt_path):
            os.makedirs(self.config.ckpt_path)
        torch.save(raw_model.state_dict(), os.path.join(self.config.ckpt_path, "checkpoint.ckpt"))

    def train(self, prt=True):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer, scheduler = raw_model.configure_optimizers(config)

        def run_epoch(split):
            self.metric.reset()
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            
            loader = DataLoader(data,  
                                pin_memory=False,
                                batch_size=config.batch_size,
                                num_workers=1,
                                collate_fn=lambda x: collate_fn(x, self.state_type, self.label_repr)
                                )
            losses = []
            hits_epoch = defaultdict(lambda : []) 
            if not is_train:
                eval_records = []
            pbar = tqdm(enumerate(loader), disable=not prt) if is_train else enumerate(loader)

            for it, pd in pbar:

                ctxt, options, labels = pd['llm_repr_ctxt'], pd['llm_repr_options'], pd['labels']
                with torch.set_grad_enabled(is_train):
                    preds, loss = model(
                        ctxt, options, labels,
                        do_sample=is_train
                        )
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                    self.metric.record(preds=preds, labels=labels)
     
                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    mean_loss = float(np.mean(losses))
                    mean_acc = {k: np.mean(v) for k, v in hits_epoch.items()}
                    lr = optimizer.param_groups[0]['lr']
                    prt_acc = self.metric.log()
                    pbar.set_description(
                        f"epoch {epoch+1}: loss {mean_loss:.2f}; lr {lr:.2e}; Acc [{prt_acc}]"
                    )

                if not is_train:
                    eval_records += pandas.concat([
                        pandas.DataFrame({
                            k+"_pred": v.cpu().tolist() for k, v in preds.items()
                        }),
                        pandas.DataFrame(pd['meta_info']),
                        pandas.DataFrame({
                            k+"_label": v.cpu().tolist() for k, v in labels.items()
                        })
                    ], axis=1).to_dict(orient='records')
            if is_train:
                self.train_loss_cont.append(mean_loss)
                self.train_acc_cont.append(mean_acc)

            if not is_train:
                test_loss = float(np.mean(losses))
                scheduler.step(test_loss)
                test_acc = {k: np.mean(v) for k, v in hits_epoch.items()}
                if prt: 
                    prt_acc = self.metric.log()
                    logger.info(f"loss {test_loss:.1f}; Acc [{prt_acc}]")
                self.test_loss_cont.append(test_loss)
                self.test_acc_cont.append(test_acc)
                return test_loss, eval_records

        self.tokens = 0  # counter used for learning rate decay

        for epoch in range(config.max_epochs):
            run_epoch('train')
            self.save_checkpoint()
            if self.test_dataset is not None:
                test_loss, eval_records = run_epoch('test')
                self.save_eval_records(eval_records, epoch)
