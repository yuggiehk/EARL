#!/bin/bash
# 文件位置：run_scripts/run_grpo_ego_irgbench.sh

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"
export WANDB_DISABLED=true
# Ego-IRGBench数据集路径配置
data_paths="/root/autodl-tmp/Ego-IRGBench_dataset/converted_ego_irgbench/ego_irgbench_train_modified_v2_cleaned.jsonl"
image_folders="/root/autodl-tmp/Ego-IRGBench_dataset"  # 包含RGB和mask的根目录
model_path="/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"  # 请根据您的实际模型路径调整

# 使用自定义的奖励函数（ego_irg_sam2）
is_reward_customized_from_vlm_module=False
reward_method="ego_irg_sam2"

echo "data_paths: $data_paths"
echo "image_folders: $image_folders"
echo "reward_method: $reward_method"

export EXP_NAME="Qwen2.5-VL-7B-Instruct-ego-irgbench-lora"
TASK_TYPE="ego_irg"

cd ${REPO_HOME}/src/open-r1-multimodal

# 确保能找到 open_r1 包与脚本
export PYTHONPATH="${REPO_HOME}/src/open-r1-multimodal/src:${PYTHONPATH}"

export DEBUG_MODE="true"
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

# 基本存在性检查（避免跑到一半才报错）
[ -f "${data_paths}" ] || { echo "Data file not found: ${data_paths}"; exit 1; }
[ -d "${image_folders}" ] || { echo "Image folder not found: ${image_folders}"; exit 1; }


torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
  ${REPO_HOME}/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir /root/autodl-tmp/GRPO_Model/new \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --reward_method $reward_method \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 2 \
    --num_train_epochs 2 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 250 \
    --eval_steps 500 \
    --num_generations 4 \
    --max_completion_length 96 \
    --reward_funcs format ego_answer ego_grounding \
    --beta 0.01 \
    --dataset-name ego_irgbench \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero2.json \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --max_pixels 12845056 \
    --min_pixels 3136 \
    --val_split_ratio 0.1 \
    --goal_embedding_path /root/ego_caption_train_features_full.npz 
# echo "Training completed for ${EXP_NAME}"



