# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

export TRAIN_DATA=''
export TEST_DATA=''

export MODEL_PATH=''
export OUTPUT_MODEL_PATH=''
python3 -u $HOME/llm_world_abs/cook/ft/sft.py \
        --dataset_path $TRAIN_DATA\
        --eval_dataset_path $TEST_DATA\
        --model_path $MODEL_PATH\
        --use_8bit_qunatization --use_peft_lora \
        --output_model_path $OUTPUT_MODEL_PATH --max_steps 1200