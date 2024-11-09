# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2023-present, Kenneth Li
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the othello-gpt (https://arxiv.org/pdf/2210.13382) implementation from https://github.com/likenneth/othello_world/blob/master/train_probe_othello.py by Kenneth Li which is licensed under the MIT License.
# You may obtain a copy of the License at 
# 
#   https://github.com/likenneth/othello_world/blob/master/LICENSE
# 
# 
#################################################################################### 

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

import time
import argparse
import torch
from probe_model import StateProbe
from trainer import Trainer, TrainerConfig
from trainer_dyna import Trainer as Trainer_dyna
from probe_dataset import ProbeDataset

parser = argparse.ArgumentParser(description='Train classification network')

parser.add_argument('--train_data_path',
                    required=True,
                    type=str)

parser.add_argument('--val_data_path',
                    required=True,
                    type=str)

parser.add_argument('--label_repr_path',
                    required=True,
                    type=str)

parser.add_argument('--state_type',
                    required=True,
                    type=str, 
                    choices=[
                        'raw', 'world', 'qval', 'world_dream', 'diff_world',
                        'policy', 'qstar', 'qabs', 'nopt', 'qopt', 'qopt_2', 'dist', 'dest', 'ndest',
                        'ablation', 'ablation_rel', 'legal'
                    ])

parser.add_argument('--test_data_path',
                    type=str)

parser.add_argument('--layer',
                    required=True,
                    default=-1,
                    type=int)

parser.add_argument('--seed',
                    default=-1,
                    type=int)

parser.add_argument('--z_nlayer',
                    default=-1,
                    type=int)

parser.add_argument('--epo',
                    default=16,
                    type=int)

parser.add_argument('--batch_size',
                    default=32,
                    type=int)

parser.add_argument('--z_dim',
                    default=5120,
                    type=int)

parser.add_argument('--input_dim',
                    default=5120,
                    type=int)


parser.add_argument('--mid_dim',
                    default=128,
                    type=int)

parser.add_argument('--sample_size',
                    default=5,
                    type=int)

parser.add_argument('--beta',
                    default=0.1,
                    type=float)

parser.add_argument('--lr',
                    default=1e-3,
                    type=float)

parser.add_argument('--exp',
                    default="state", 
                    type=str)

parser.add_argument('--output_path',
                    required=True,
                    type=str)

parser.add_argument("--use_VIB", 
                    action="store_true")

parser.add_argument("--for_nearby", 
                    action="store_true")

args, _ = parser.parse_known_args()

if args.seed >= 0:
    torch.manual_seed(args.seed)

train_dataset = ProbeDataset(args.train_data_path, args.label_repr_path, infinite=True, shuffle=True)

val_dataset = ProbeDataset(args.val_data_path, args.label_repr_path, infinite=False, shuffle=False)
assert isinstance(train_dataset, torch.utils.data.IterableDataset)

model = StateProbe(args.state_type, args.layer, input_dim=args.input_dim, mid_dim=args.mid_dim)

model.to("cuda")

sampler = None

max_epochs = args.epo
t_start = time.strftime("_%Y%m%d_%H%M%S")
tconf = TrainerConfig(
    max_epochs=max_epochs,
    batch_size=args.batch_size,
    learning_rate=args.lr,
    betas=(.9, .999),
    lr_decay=True,
    num_workers=4,
    weight_decay=0.,
    ckpt_path=os.path.join(args.output_path, f"layer{args.layer}_hid{args.mid_dim}/")
)

trainer_cls = Trainer_dyna if args.for_nearby else Trainer
trainer = trainer_cls(
    args.state_type,
    model,
    train_dataset,
    val_dataset,
    train_dataset.label_repr,
    tconf)

trainer.train(prt=True)
trainer.save_checkpoint()