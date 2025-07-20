# model_path="/home/nfs05/model/Qwen2.5-Math-7B-Instruct/"
miner_run_model_path="/home/nfs05/model/Qwen2.5-0.5B-Instruct"
export CUDA_VISIBLE_DEVICES=1,3
export YOUR_RUN_NAME="verl_gsm8k_init_training"
export YOUR_PROJECT_NAME="init_training"
export DATASET_DOWNLOAD_PATH="/home/nfs06/chenyq/data/datasets"
export DATASET_CACHE_PATH="/home/nfs06/chenyq/data/cache"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$DATASET_CACHE_PATH"
export HF_DATASETS_CACHE="$DATASET_DOWNLOAD_PATH"

# 自动计算GPU数量
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # 计算逗号分隔的GPU数量
    GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    # 如果没有设置CUDA_VISIBLE_DEVICES，使用所有可用GPU
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
fi

echo "检测到 $GPU_COUNT 张GPU，设置 trainer.n_gpus_per_node=$GPU_COUNT"

# 运行VERL训练
#    <7B: tensor_model_parallel_size=1
# 7B-30B: tensor_model_parallel_size=2-4
#  critic.model.fsdp_config.param_offload=True \
#  critic.model.fsdp_config.optimizer_offload=True \
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$DATASET_DOWNLOAD_PATH/gsm8k/train.parquet \
 data.val_files=$DATASET_DOWNLOAD_PATH/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=${miner_run_model_path} \
 actor_rollout_ref.rollout.name=sglang \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=${miner_run_model_path} \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=$GPU_COUNT \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.logger='["console","wandb"]' \
 trainer.project_name=$YOUR_PROJECT_NAME \
 trainer.experiment_name=$YOUR_RUN_NAME \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log