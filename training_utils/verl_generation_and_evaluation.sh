#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1

eval_dataset_path="/home/nfs06/chenyq/data/datasets/gsm8k/test.parquet"

N_SAMPLES=1
nnodes=1
n_gpus_per_node=1

CHECKPOINT="global_step_435" 
TRAINING_SETTING="verl_gsm8k_init_training_v2" # verl_gsm8k_init_training
SAVE_SUFFIX=gsm8k_responses_ppo_v2
CHECKPOINT_PATH="/home/nfs06/chenyq/Code-Utils/training_utils/checkpoints/init_training/$TRAINING_SETTING/$CHECKPOINT"
MODEL_PATH="$CHECKPOINT_PATH/actor/huggingface"
# MODEL_PATH="/home/nfs05/model/Qwen2.5-0.5B-Instruct"

WORK_DIR="/home/nfs06/chenyq/Code-Utils/training_utils"
GENERATION_RESULT_DIR="$WORK_DIR/evaluation_results/generation_res"
EVAL_RESULT_DIR="$WORK_DIR/evaluation_results/generation_res"
RESULT_DIR="$WORK_DIR/evaluation_results/logs"
mkdir -p $GENERATION_RESULT_DIR
mkdir -p $EVAL_RESULT_DIR
mkdir -p $RESULT_DIR

GENERATION_RESPONSES=$GENERATION_RESULT_DIR/$SAVE_SUFFIX.parquet

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_PREFIX="${SAVE_SUFFIX}_n${N_SAMPLES}_${TIMESTAMP}"
LOG_FILE="$RESULT_DIR/${OUTPUT_PREFIX}.log"
RUN_DIR="$RESULT_DIR/${OUTPUT_PREFIX}_run"

# Generation and then evaluation.
python -m verl.trainer.main_generation\
  model.path=$MODEL_PATH \
  data.path=$eval_dataset_path \
  data.output_path=$GENERATION_RESPONSES \
  data.prompt_key=prompt \
  data.n_samples=$N_SAMPLES \
  trainer.n_gpus_per_node=$n_gpus_per_node \
  trainer.nnodes=$nnodes \
  2>&1 | tee -a $LOG_FILE

python evaluation.py \
    data.path=$GENERATION_RESPONSES \
    data.prompt_key=prompt \
    data.response_key=responses \
    data.data_source_key=data_source \
    data.reward_model_key=reward_model \
    custom_reward_function.task_name=gsm8k \
    ray_init.num_cpus=8 \
    2>&1 | tee $LOG_FILE