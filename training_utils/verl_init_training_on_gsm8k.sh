# model_path="/home/nfs05/model/Qwen2.5-Math-7B-Instruct/"
miner_run_model_path="/home/nfs05/model/Qwen2.5-0.5B-Instruct"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export YOUR_RUN_NAME="verl_gsm8k_init_training_v2"
export YOUR_PROJECT_NAME="init_training"
export DATASET_DOWNLOAD_PATH="/home/nfs06/chenyq/data/datasets"
export DATASET_CACHE_PATH="/home/nfs06/chenyq/data/cache"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$DATASET_CACHE_PATH"
export HF_DATASETS_CACHE="$DATASET_DOWNLOAD_PATH"

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
fi

echo "检测到 $GPU_COUNT 张GPU，设置 trainer.n_gpus_per_node=$GPU_COUNT"

#    <7B: tensor_model_parallel_size=1
# 7B-30B: tensor_model_parallel_size=2-4 
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATASET_DOWNLOAD_PATH/gsm8k/train.parquet \
    data.val_files=$DATASET_DOWNLOAD_PATH/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=${miner_run_model_path} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    critic.optim.lr=1e-5 \
    critic.shuffle=True \
    critic.model.path=${miner_run_model_path} \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$GPU_COUNT \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$YOUR_PROJECT_NAME \
    trainer.experiment_name=$YOUR_RUN_NAME \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log