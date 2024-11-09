# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2023-present, Kenneth Li
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the othello-gpt (https://arxiv.org/pdf/2210.13382) implementation from https://github.com/likenneth/othello_world/blob/master/mingpt/probe_model.py by Kenneth Li which is licensed under the MIT License.
# You may obtain a copy of the License at 
# 
#   https://github.com/likenneth/othello_world/blob/master/LICENSE
# 
# 
#################################################################################### 

import torch
import torch.nn as nn
from torch.nn import functional as F

from task_info import _TASKS

_IGNORE_INDEX = -100


class Probe(nn.Module):
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # biases of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=0)
        return optimizer, scheduler


def map_predicate_to_task(predicate_key):
    if predicate_key.startswith("obj"):
        return 'obj'
    elif predicate_key.startswith("agent"):
        return 'agent'
    elif predicate_key.startswith("boxes"):
        return 'boxes'
    else:
        assert False


def calc_label_wght(labels, label_size):
    wght = [0. for _ in range(label_size)]
    total = labels.ne(_IGNORE_INDEX).float().sum()
    for y in range(label_size):
        wght[y] = 1. - (labels.eq(y).float().sum() / (total+1e-8))
    wght = torch.tensor(wght)
    ret =  wght / torch.sum(wght)
    return ret+1e-5


class BaseCLS(nn.Module):

    def __init__(self, nlayer, input_dim, output_dim, mid_dim=None, do_rwght_label=False):  # from 0 to 15
        super().__init__()
        self.input_dim = input_dim

        self.do_rwght_label = do_rwght_label
        if nlayer == 1:
            print("FORCING MID_DIM=OUT_DIM!")
            mid_dim = output_dim
        self.mid_dim = mid_dim

        proj_modules = [nn.Linear(self.input_dim, self.mid_dim, bias=True)]
        if nlayer > 2:
            for _ in range(nlayer-2):
                proj_modules.extend(
                    [nn.ReLU(True),
                    nn.Linear(self.mid_dim, self.mid_dim, bias=True),]    
                )
        if nlayer > 1:
            proj_modules.extend(
                [nn.ReLU(True),
                nn.Linear(self.mid_dim, output_dim, bias=True),]
            )
        self.nlabels = output_dim
        self.proj_modules = nn.Sequential(*proj_modules)

    def forward(self, ctxt, labels=None):
        '''
        ctxt: llm_repr for state_and_ops [bsz x dim]
        labels: [bsz]
        '''

        logits = self.proj_modules(ctxt.float().cuda())

        if labels is None:
            return logits.argmax(1), None
        else:
            labels = labels.cuda()
            if self.do_rwght_label:
                loss = F.cross_entropy(
                    logits, 
                    labels.cuda(), 
                    weight=calc_label_wght(labels, self.nlabels).cuda(),
                    ignore_index=_IGNORE_INDEX)

            else:
                loss = F.cross_entropy(
                    logits, 
                    labels.cuda(), 
                    ignore_index=_IGNORE_INDEX)
            return logits.argmax(1), loss


def make_encoder(nl, inp_dim, mid_dim, out_dim):
    if nl == 1:
        return nn.Sequential(
            nn.Linear(inp_dim, out_dim, bias=True),
        )

    layers = [nn.Linear(inp_dim, mid_dim, bias=True)]

    if nl > 2:
        for _ in range(nl-2):
            layers.extend([
                nn.ReLU(True),
                nn.Linear(mid_dim, mid_dim, bias=True),
            ])
    layers.extend([
        nn.ReLU(True),
        nn.Linear(mid_dim, out_dim, bias=True),
    ])
    return nn.Sequential(*layers)


class BaseRanker(nn.Module):

    def __init__(self, nlayer, encode_ctxt=False, dense_label=False, input_dim=5120, mid_dim=None):  # from 0 to 15
        super().__init__()
        self.dense_label = dense_label  # each cand has a binary label
        self.input_dim = input_dim*3

        if nlayer == 1:
            print("FORCING MID_DIM=OUT_DIM!")
            mid_dim = 1
        self.mid_dim = mid_dim

        if encode_ctxt:
            self.encoder = make_encoder(nlayer, 5120, 5120, 5120)
        else:
            self.encoder = None

        proj_modules = [nn.Linear(self.input_dim, self.mid_dim, bias=True)]
        if nlayer > 2:
            for _ in range(nlayer-2):
                proj_modules.extend(
                    [nn.ReLU(True),
                     nn.Linear(self.mid_dim, self.mid_dim, bias=True),]
                )
        if nlayer > 1:
            proj_modules.extend(
                [nn.ReLU(True),
                 nn.Linear(
                    self.mid_dim, 
                    1, 
                    bias=True),]
            )
        self.proj_modules = nn.Sequential(*proj_modules)

    def make_nn_inp(self, x1, x2):
        return torch.concat(
            [
              x1 * x2,
              x1 + x2,
              x1 - x2,
            ], dim=-1
        )

    def forward(self, ctxt, options, labels=None):
        '''
        ctxt: llm_repr for state_and_ops [bsz x dim]
        labels: [bsz]
        '''
        if self.encoder is not None:
            ctxt = self.encoder(ctxt.cuda().float())

        x = self.make_nn_inp(
            ctxt.float().unsqueeze(dim=1).repeat(1, options.size(1), 1).cuda(),
            options.float().cuda()
        )

        logits = self.proj_modules(x.float()).squeeze(2)  # [bsz, ncand, 1]
        if labels is None:
            if self.dense_label:
                return torch.where(
                    F.sigmoid(logits).ge(0.5), 1., 0.
                ), None
            else:
                return logits.argmax(1), None
        else:
            labels = labels.cuda()
            if self.dense_label:
                mask = labels.ne(_IGNORE_INDEX)
                # NOTE: risk of being divided by 0 if all labels are _IGNORE_INDEX
                pos_ratio = (
                        (1+labels.eq(1.).float().sum()) / (mask.float().sum()+1)
                    ).item()
                labels_slct = torch.masked_select(labels, mask)
                loss = F.binary_cross_entropy_with_logits(
                    torch.masked_select(logits, mask),
                    labels_slct,
                    weight=torch.where(
                        labels_slct.eq(1.),
                        1.-pos_ratio,
                        pos_ratio
                    )
                    )

                return torch.where(
                    F.sigmoid(logits).ge(0.5), 1., 0.
                ), loss
            else:
                loss = F.cross_entropy(logits, labels.cuda(), ignore_index=_IGNORE_INDEX)
                return logits.argmax(1), loss


class StateProbe(Probe):
    def __init__(self, state_type, nlayer, input_dim=5120, mid_dim=None):  # from 0 to 15
        super().__init__()
        self.input_dim = input_dim
        self.state_type = state_type

        proj = {}

        for k, v in _TASKS[state_type].items():
            print('v', v)
            if v['task_type'] == 'rank':
                proj[k] = BaseRanker(nlayer, dense_label=v['dense'], input_dim=input_dim, mid_dim=mid_dim)
            elif v['task_type'] == 'cls':
                proj[k] = BaseCLS(
                    nlayer, input_dim, v['label_size'], 
                    mid_dim=mid_dim, do_rwght_label=v['reweight']
                )
            else:
                assert False

        self.proj = nn.ModuleDict(proj)
        self.apply(self._init_weights)

    def forward(self, ctxt, predicates_options, labels=None, debug=False, **kwargs):
        '''
        ctxt: predicate_key: llm_repr for state_and_ops [bsz x dim]
        predicates_options: {predicate_key: [bsz x ncand x dim]}
        '''

        all_loss, all_logits = 0., {}

        for task, info in _TASKS[self.state_type].items():
            if task not in labels.keys():
                if debug:
                    print(f"NO {task} task in this batch, SKIPPING")
                continue

            if info['task_type'] == 'rank':

                logits, loss = self.proj[task](
                            ctxt, predicates_options[task],
                            labels=(labels[task] if labels is not None else None)
                        )
            else:
                logits, loss = self.proj[task](
                        ctxt, labels=(labels[task] if labels is not None else None)
                )
            all_loss += loss
            all_logits[task] = logits

        return all_logits, all_loss
