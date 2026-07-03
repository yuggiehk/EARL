#!/bin/bash
# 文件位置：/root/VLM-R1/run_scripts/run_grpo_ego_irg.sh

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"

# Ego-IRGBench数据路径配置
data_paths="/root/autodl-tmp/Ego-IRGBench_dataset/converted_ego_irgbench/ego_irgbench_train.jsonl" 
image_folders="/root/autodl-tmp/Ego-IRGBench_dataset"  # RGB和mask的父目录
model_path="/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"  # 根据您的模型路径调整
reward_method="ego_irg_sam2"  # 使用我们的SAM2集成奖励方法
is_reward_customized_from_vlm_module=False  # 使用我们自定义的奖励函数

echo "data_paths: $data_paths"
echo "image_folders: $image_folders"
echo "reward_method: $reward_method"

export EXP_NAME="Qwen2.5-VL-7B-Instruct-ego-irg-sam2-lora"
TASK_TYPE="ego_irg"  # 自定义任务类型
cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

# 4张A800 80G GPU配置
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir /root/autodl-tmp/GRPO_Model \
    --resume_from_checkpoint False \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --reward_method $reward_method \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --logging_steps 10 \
    --num_train_epochs 5 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 250 \
    --eval_steps 500 \
    --num_generations 4 \
    --max_completion_length 256 \
    --reward_funcs ego_irg_sam2 \
    --beta 0.04 \
    --dataset-name ego-irgbench \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero2.json \
    --learning_rate 5e-7 \
    --warmup_ratio 0.1 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --max_pixels 12845056 \
    --min_pixels 3136 \
    --val_split_ratio 0.1

echo "Training completed for ${EXP_NAME}"