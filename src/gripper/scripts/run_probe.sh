# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

export LOG_DIR="$HOME/gripper_probe_logs"
export LLM="llama3" # LLM name, e.g. llama3
export LLM_MODEL="$HOME/scratch/probe_lm/models/Meta-Llama-3.1-8B-Instruct/" # {Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2, Llama-2-13b-chat-hf, Llama-2-7b-chat-hf, Phi-3-medium-4k-instruct, Phi-3-mini-4k-instruct}
export ADAPT_DIR="None" # path to fine-tuned LLM or None(for ICL)
export ICL_DATA="$HOME/gripper_dataset/val.pkl" # 
export PROMPT_METHOD='icl' #{'icl', 'inst-ft'} 'ft' for pythia
export DIM=8600
export input_dim=4096
export num_train=10000
export num_test=1000
export epoc=30
export layer_index=-6 # you need to adjust it accordinly if you use a shallow LM
export TRAIN_DATA="$HOME/gripper_dataset/train.pkl"
export TEST_DATA="$HOME/gripper_dataset/test.pkl"

mkdir $LOG_DIR
mkdir "${LOG_DIR}/${LLM}"
# box_name, box_content, agent_has_t, aloc, subgoal, relative_pos
export PROBE_DATA_TRAIN="$HOME/gripper_probe_dataset/probe_${LLM}_${PROMPT_METHOD}/train/"
export PROBE_DATA_TEST="$HOME/gripper_probe_dataset/probe_${LLM}_${PROMPT_METHOD}/test/"
export PROBE_DATA_LMR="$HOME/gripper_probe_dataset/probe_${LLM}_${PROMPT_METHOD}/label_embed.pkl"
# MAKE DATA
python3 -u $HOME/llm_world_abs/src/gripper/probe/make_probe_data.py \
        --dataset_path $TRAIN_DATA\
        --output_dataset_path $PROBE_DATA_TRAIN --n_layer $layer_index --num_neg 5  --icl_dataset_path $ICL_DATA\
        --ckpt_dir $LLM_MODEL --prompt_method $PROMPT_METHOD --n_data $num_train --num_neg 5 --adapter_dir $ADAPT_DIR

python3 -u $HOME/llm_world_abs/src/gripper/probe/make_probe_data.py \
        --dataset_path $TEST_DATA\
        --output_dataset_path $PROBE_DATA_TEST\
        --ckpt_dir $LLM_MODEL --prompt_method $PROMPT_METHOD --icl_dataset_path $ICL_DATA\
        --n_data $num_test --n_layer $layer_index --record_meta --num_neg 5

python3 -u $HOME/llm_world_abs/src/gripper/probe/make_label_lmr.py \
        --dataset_path $PROBE_DATA_TRAIN\
        --output_dataset_path $PROBE_DATA_LMR\
        --ckpt_dir $LLM_MODEL --use_embed

# TRAIN PROBE
for state_type in "raw" "world" "legal" "policy" "ndest"; do
    export out_model_path="$HOME/llm_world_abs/src/gripper/models_probe/${LLM}_${state_type}_probe_l2_dim${DIM}_last6/"
    echo "${LOG_DIR}/${LLM}_${state_type}_${DIM}.log"
    python3 -u $HOME/llm_world_abs/src/gripper/probe/train.py \
        --train_data_path $PROBE_DATA_TRAIN \
        --val_data_path $PROBE_DATA_TEST \
        --label_repr_path $PROBE_DATA_LMR\
        --output_path $out_model_path \
        --layer 2\
        --epo $epoc\
        --state_type $state_type\
        --batch_size 64 \
        --mid_dim $DIM \
        --input_dim $input_dim > "${LOG_DIR}/${LLM}/${state_type}_${DIM}_${PROMPT_METHOD}.log" 2>&1
done

# box_content_g, agent_has_g
export PROBE_DATA_TRAIN="$HOME/gripper_probe_dataset/probe_final_${LLM}_${PROMPT_METHOD}/train/"
export PROBE_DATA_TEST="$HOME/gripper_probe_dataset/probe_final_${LLM}_${PROMPT_METHOD}/test/"

# make goal data
python3 -u $HOME/llm_world_abs/src/gripper/probe/make_probe_data.py \
        --dataset_path $TRAIN_DATA\
        --output_dataset_path $PROBE_DATA_TRAIN\
        --ckpt_dir $LLM_MODEL --prompt_method $PROMPT_METHOD  --icl_dataset_path $ICL_DATA\
        --n_data $num_train --n_layer $layer_index --num_neg 5 --final_predicate

python3 -u $HOME/llm_world_abs/src/gripper/probe/make_probe_data.py \
        --dataset_path $TEST_DATA\
        --output_dataset_path $PROBE_DATA_TEST\
        --ckpt_dir $LLM_MODEL --prompt_method $PROMPT_METHOD --icl_dataset_path $ICL_DATA\
        --n_data $num_test --n_layer $layer_index --record_meta --num_neg 5 --final_predicate

# TRAIN goal PROBE
for state_type in "world_dream"; do
    export out_model_path="$HOME/llm_world_abs/src/gripper/models_final_probe/llm=${LLM}_${state_type}_probe_l2_dim${DIM}_last6/"
    python3 -u $HOME/llm_world_abs/src/gripper/probe/train.py \
        --train_data_path $PROBE_DATA_TRAIN \
        --val_data_path $PROBE_DATA_TEST \
        --label_repr_path $PROBE_DATA_LMR\
        --output_path $out_model_path \
        --layer 2\
        --epo $epoc\
        --state_type $state_type\
        --batch_size 64 \
        --mid_dim $DIM \
        --input_dim $input_dim > "${LOG_DIR}/${LLM}/${state_type}_${DIM}_${PROMPT_METHOD}_final.log" 2>&1
done
        
# nearby
export PROBE_DATA_TRAIN="$HOME/gripper_probe_dataset/probe_dyna_${LLM}_${PROMPT_METHOD}/train/"
export PROBE_DATA_TEST="$HOME/gripper_probe_dataset/probe_dyna_${LLM}_${PROMPT_METHOD}/test/"

# make nearby data
python3 -u $HOME/llm_world_abs/src/gripper/probe/make_probe_data_dyna.py \
        --dataset_path $TRAIN_DATA\
        --output_dataset_path $PROBE_DATA_TRAIN\
        --ckpt_dir $LLM_MODEL --prompt_method $PROMPT_METHOD  --icl_dataset_path $ICL_DATA\
        --n_data $num_train --n_layer $layer_index --num_neg 5 --tasks "legal" --obj_mention_method 'all'

python3 -u $HOME/llm_world_abs/src/gripper/probe/make_probe_data_dyna.py \
        --dataset_path $TEST_DATA\
        --output_dataset_path $PROBE_DATA_TEST\
        --ckpt_dir $LLM_MODEL --prompt_method $PROMPT_METHOD --icl_dataset_path $ICL_DATA\
        --record_meta --n_data $num_test --n_layer $layer_index --num_neg 5 --tasks "legal" --obj_mention_method 'all'

export state_type="legal"
export out_model_path="$HOME/models_dyna_probe/llm=${LLM}_${state_type}_probe_l2_dim${DIM}_last6/"
python3 -u $HOME/llm_world_abs/src/gripper/probe/train.py \
    --for_nearby \
    --train_data_path $PROBE_DATA_TRAIN \
    --val_data_path $PROBE_DATA_TEST \
    --label_repr_path $PROBE_DATA_LMR\
    --output_path $out_model_path \
    --layer 2\
    --epo $epoc\
    --state_type $state_type\
    --batch_size 64 \
    --mid_dim $DIM --input_dim $input_dim > "${LOG_DIR}/${LLM}/${state_type}_${DIM}_${PROMPT_METHOD}_dyna.log" 2>&1

# obj_ctrl
export PROBE_DATA_TRAIN="$HOME/gripper_probe_dataset/probe_objCtrl_${LLM}_${PROMPT_METHOD}/train/"
export PROBE_DATA_TEST="$HOME/gripper_probe_dataset/probe_objCtrl_${LLM}_${PROMPT_METHOD}/test/"

# make obj_ctrl data
python3 -u $HOME/llm_world_abs/src/gripper/probe/make_probe_data.py \
        --dataset_path $TRAIN_DATA\
        --output_dataset_path $PROBE_DATA_TRAIN --n_layer $layer_index --num_neg 5  --for_obj_ctrl --icl_dataset_path $ICL_DATA\
        --ckpt_dir $LLM_MODEL --prompt_method $PROMPT_METHOD --n_data $num_train --num_neg 5

python3 -u $HOME/llm_world_abs/src/gripper/probe/make_probe_data.py \
        --dataset_path $TEST_DATA\
        --output_dataset_path $PROBE_DATA_TEST\
        --ckpt_dir $LLM_MODEL --prompt_method $PROMPT_METHOD --for_obj_ctrl --icl_dataset_path $ICL_DATA\
        --n_data $num_test --n_layer $layer_index --record_meta --num_neg 5

export state_type="policy"
export out_model_path="$HOME/models_objCtrl/llm=${LLM}_${state_type}_probe_l2_dim${DIM}_last6/"
python3 -u $HOME/llm_world_abs/src/gripper/probe/train.py \
    --train_data_path $PROBE_DATA_TRAIN \
    --val_data_path $PROBE_DATA_TEST \
    --label_repr_path $PROBE_DATA_LMR\
    --output_path $out_model_path \
    --layer 2\
    --epo $epoc\
    --state_type $state_type\
    --batch_size 64 \
    --mid_dim $DIM --input_dim $input_dim > "${LOG_DIR}/${LLM}/${state_type}_${DIM}_${PROMPT_METHOD}_objCtrl.log" 2>&1