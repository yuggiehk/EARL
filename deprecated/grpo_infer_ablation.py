import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from peft import PeftModel
from transformers import AutoTokenizer, AutoProcessor
from modelscope import Qwen2_5_VLForConditionalGeneration

# ==========================
# 手动路径配置（可直接修改）
# ==========================
BASE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
LORA_ADAPTER_PATH = "/root/autodl-tmp/GRPO_Model/half_format_checkpoint-1000"
TEST_DATA_PATH = "/root/autodl-tmp/Ego-IRGBench_dataset/converted_ego_irgbench/ego_irgbench_test_modified_v2_cleaned.jsonl"
OUTPUT_PATH = "/root/VLM-R1/ablation_half_format_test.json"
SAVE_INTERVAL = 10  # 每多少条样本保存一次结果


def save_results(results, output_path):
    """保存结果到 JSON 文件（原子写入）。"""
    temp_path = output_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    os.replace(temp_path, output_path)


def load_model_and_processor(base_model_path, lora_adapter_path):
    """加载基础模型 + LoRA 适配器，并融合权重。"""
    print(f"1️⃣ 从 '{base_model_path}' 加载基础模型和处理器...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"2️⃣ 加载 LoRA 适配器 '{lora_adapter_path}'...")
    model = PeftModel.from_pretrained(model, lora_adapter_path)

    print("3️⃣ 合并 LoRA 权重到主模型中…")
    model = model.merge_and_unload()
    model.eval()

    print("✅ 模型加载完成。")
    return model, processor, tokenizer


def run_inference():
    """主推理流程。"""
    model, processor, tokenizer = load_model_and_processor(BASE_MODEL_PATH, LORA_ADAPTER_PATH)

    print(f"4️⃣ 读取测试集: {TEST_DATA_PATH}")
    results = []
    processed = set()

    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed = {item["image_path"] for item in results if "image_path" in item}
            print(f"发现已有结果文件，共 {len(processed)} 条。将在此基础上断点续跑。")
        except Exception as e:
            print(f"⚠️ 读取已有结果失败 ({e})，重新运行。")
            results = []

    # 读取测试数据
    all_data = []
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            all_data.append(json.loads(line))

    test_data = [d for d in all_data if d.get("image") not in processed]
    print(f"总测试样本: {len(all_data)}，已完成: {len(processed)}，待处理: {len(test_data)}")

    if not test_data:
        print("🎯 所有样本已处理完毕！")
        return

    print("5️⃣ 开始推理...")
    with tqdm(total=len(test_data), desc="推理中") as pbar:
        for i, item in enumerate(test_data):
            image_path = item.get("image")
            query_text = item["conversations"][0]["value"]
            ground_truth = item["conversations"][1]["value"]

            if not os.path.exists(image_path):
                tqdm.write(f"⚠️ 文件缺失: {image_path}")
                results.append({
                    "image_path": image_path,
                    "query": query_text,
                    "ground_truth": ground_truth,
                    "prediction": "ERROR: Image not found"
                })
                pbar.update(1)
                continue

            try:
                image = Image.open(image_path).convert("RGB")

                # 构建输入消息
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": query_text}
                    ]}
                ]

                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    gen_kwargs = {"max_new_tokens": 88, "do_sample": False}
                    generated_ids = model.generate(**inputs, **gen_kwargs)

                gen_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
                prediction = processor.decode(gen_ids[0], skip_special_tokens=True).strip()

                results.append({
                    "image_path": image_path,
                    "query": query_text,
                    "ground_truth": ground_truth,
                    "prediction": prediction
                })

            except Exception as e:
                tqdm.write(f"❌ 推理错误 {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "query": query_text,
                    "ground_truth": ground_truth,
                    "prediction": f"ERROR: {str(e)}"
                })

            pbar.update(1)

            # 定期保存进度
            if (i + 1) % SAVE_INTERVAL == 0 or (i + 1) == len(test_data):
                tqdm.write(f"💾 保存进度: {len(results)} / {len(all_data)}")
                save_results(results, OUTPUT_PATH)

    print("✅ 推理完成，所有结果已保存至：", OUTPUT_PATH)


if __name__ == "__main__":
    run_inference()