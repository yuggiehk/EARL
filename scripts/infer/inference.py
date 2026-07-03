#!/usr/bin/env python3
"""
EARL Inference Script — Qwen2.5-VL-7B + LoRA Adapter for Ego-IRGBench.

Usage:
    python inference.py \
        --base_model /path/to/Qwen2.5-VL-7B-Instruct \
        --lora_adapter /path/to/checkpoint-1000 \
        --test_data /path/to/test.jsonl \
        --output /path/to/results.json \
        --max_new_tokens 88 \
        --save_interval 10
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from peft import PeftModel
from transformers import AutoTokenizer, AutoProcessor
from modelscope import Qwen2_5_VLForConditionalGeneration


def save_results(results, output_path):
    """Atomic save to JSON file."""
    temp_path = output_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    os.replace(temp_path, output_path)


def load_model_and_processor(base_model_path, lora_adapter_path):
    """Load base model + LoRA adapter, merge weights."""
    print(f"[1/3] Loading base model from '{base_model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[2/3] Loading LoRA adapter from '{lora_adapter_path}'...")
    model = PeftModel.from_pretrained(model, lora_adapter_path)

    print("[3/3] Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    model.eval()

    print("Model ready.")
    return model, processor, tokenizer


def run_inference(args):
    """Main inference loop with checkpoint resumption."""
    model, processor, tokenizer = load_model_and_processor(
        args.base_model, args.lora_adapter
    )

    print(f"Reading test data: {args.test_data}")
    results = []
    processed = set()

    # Resume from existing output if present
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed = {item["image_path"] for item in results if "image_path" in item}
            print(f"Found existing results: {len(processed)} samples. Resuming.")
        except Exception as e:
            print(f"Warning: Could not read existing results ({e}). Starting fresh.")
            results = []

    # Load all test data
    all_data = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for line in f:
            all_data.append(json.loads(line))

    test_data = [d for d in all_data if d.get("image") not in processed]
    print(f"Total: {len(all_data)} | Done: {len(processed)} | Remaining: {len(test_data)}")

    if not test_data:
        print("All samples already processed!")
        return

    print("Starting inference...")
    with tqdm(total=len(test_data), desc="Inference") as pbar:
        for i, item in enumerate(test_data):
            image_path = item.get("image")
            query_text = item["conversations"][0]["value"]
            ground_truth = item["conversations"][1]["value"]

            if not os.path.exists(image_path):
                tqdm.write(f"Missing file: {image_path}")
                results.append({
                    "image_path": image_path,
                    "query": query_text,
                    "ground_truth": ground_truth,
                    "prediction": "ERROR: Image not found",
                })
                pbar.update(1)
                continue

            try:
                image = Image.open(image_path).convert("RGB")

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query_text},
                    ],
                }]

                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(text=[text], images=[image], return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    gen_kwargs = {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": False,
                    }
                    generated_ids = model.generate(**inputs, **gen_kwargs)

                gen_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
                prediction = processor.decode(
                    gen_ids[0], skip_special_tokens=True
                ).strip()

                results.append({
                    "image_path": image_path,
                    "query": query_text,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                })

            except Exception as e:
                tqdm.write(f"Error on {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "query": query_text,
                    "ground_truth": ground_truth,
                    "prediction": f"ERROR: {str(e)}",
                })

            pbar.update(1)

            # Periodic save
            if (i + 1) % args.save_interval == 0 or (i + 1) == len(test_data):
                tqdm.write(f"Saving progress: {len(results)} / {len(all_data)}")
                save_results(results, args.output)

    print(f"Done! Results saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="EARL Inference — Qwen2.5-VL-7B + LoRA on Ego-IRGBench"
    )
    parser.add_argument(
        "--base_model", type=str,
        default="/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct",
        help="Path to base Qwen2.5-VL model",
    )
    parser.add_argument(
        "--lora_adapter", type=str,
        default="/root/autodl-tmp/GRPO_Model/new/checkpoint-1000",
        help="Path to LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--test_data", type=str,
        default="/root/autodl-tmp/Ego-IRGBench_dataset/converted_ego_irgbench/ego_irgbench_test_modified_v2_cleaned.jsonl",
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--output", type=str,
        default="/root/VLM-R1/outputs/inference_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=88,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--save_interval", type=int, default=10,
        help="Save results every N samples",
    )
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
