export YOUR_RUN_NAME="verl_gsm8k_init_training_v2"
export YOUR_PROJECT_NAME="init_training"

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/${YOUR_PROJECT_NAME}/${YOUR_RUN_NAME}/global_step_435/actor \
    --target_dir checkpoints/${YOUR_PROJECT_NAME}/${YOUR_RUN_NAME}/global_step_435/actor/huggingface