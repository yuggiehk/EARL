"""
[DEPRECATED] The first 919 lines of this file contained an earlier
version of the GRPO training script with Ego-IRGBench reward functions, SAM2 integration, and goal embedding support. They have been removed for clarity.
See the git history or deprecated/ folder for the original code.
"""

#         trainer.push_to_hub()


# if __name__ == "__main__":
#     parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
#     script_args, training_args, model_args = parser.parse_args_and_config()
#     if training_args.deepspeed and "zero3" in training_args.deepspeed:
#         print("zero3 is used, qwen2_5vl forward monkey patch is applied")
#         monkey_patch_qwen2_5vl_forward()
#     main(script_args, training_args, model_args)






import os
import re
import pathlib
import cv2
import torch
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from babel.numbers import parse_decimal
from utils.math import compute_score
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import PIL
from Levenshtein import ratio
from open_r1.utils.pycocotools.coco import COCO
from open_r1.utils.pycocotools.cocoeval import COCOeval
import json
import math
from json_repair import repair_json
from pycocoevalcap.cider.cider import Cider
from PIL import Image

import ast
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize


for pkg in ["punkt", "punkt_tab", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}") if "punkt" in pkg else nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        print(f"NLTK resource '{pkg}' not found, downloading...")
        nltk.download(pkg)


from open_r1.vlm_modules import *

from transformers.utils import logging
from transformers import AutoProcessor, AutoTokenizer

from openai import OpenAI


def sigmoid_normalize(score: float, k: float = 1.0, x0: float = 1.5) -> float:
    return 1 / (1 + math.exp(-k * (score - x0)))


logger = logging.get_logger(__name__)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-proj-1234567890"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
)

from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward, monkey_patch_torch_load
monkey_patch_qwen2_5vl_flash_attn()    
monkey_patch_torch_load()

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    _sam2_predictor = None
    
    def get_sam2_predictor(
        checkpoint_path: str = "/root/sam2/checkpoints/sam2_hiera_large.pt", 
        model_cfg_path: str = "sam2_hiera_l.yaml"
    ): 
        global _sam2_predictor
        if _sam2_predictor is None:
            logger.info("Initializing SAM2 predictor for the first time...")
            
            if not os.path.exists(checkpoint_path):
                logger.error(f"SAM2 checkpoint not found: {checkpoint_path}")
                return None

            try:
                logger.info(f"Loading SAM2 model with config: {model_cfg_path}")
                sam2_model = build_sam2(model_cfg_path, checkpoint_path)
                _sam2_predictor = SAM2ImagePredictor(sam2_model)
                logger.info("SAM2 model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load SAM2 model: {e}")
                _sam2_predictor = None
                
        return _sam2_predictor
    
except ImportError:
    print("Warning: SAM2 not available. Using placeholder functions.")
    _sam2_predictor = None
    
    def get_sam2_predictor(checkpoint_path: str = "", model_cfg: str = ""):
        return None

tokenizer = None

def initialize_tokenizer(model_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer

@dataclass
class GRPOScriptArguments(ScriptArguments):
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    task_type: Optional[str] = field(
        default=None,
        metadata={"help": "Choose task type: 'default', 'gui', ..."},
    )
    is_reward_customized_from_vlm_module: bool = field(
        default=False,
        metadata={"help": "Whether to use a customized reward from vlm module"},
    )
    goal_embedding_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-computed goal caption embeddings (.npz file)."}
    )

def extract_three_stage_content(text: str) -> Tuple[str, str]:
    """Extracts content from <answer> and <bbox> tags using a loose search."""
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    bbox_match = re.search(r'<bbox>(.*?)</bbox>', text, re.DOTALL)
    
    answer = answer_match.group(1).strip() if answer_match else ""
    bbox = bbox_match.group(1).strip() if bbox_match else ""
    
    return answer, bbox

def count_correct_tags(text: str) -> int:
    patterns = [
        r'<answer>.*?</answer>',
        r'<bbox>.*?</bbox>'
    ]
    
    correct_count = 0
    for pattern in patterns:
        if re.search(pattern, text, re.DOTALL):
            correct_count += 1
    
    return correct_count

def parse_answer_list(answer_str: str) -> Optional[List[str]]:
    try:
        parsed_list = ast.literal_eval(answer_str)
        if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
            return parsed_list
        return None
    except (ValueError, SyntaxError, TypeError):
        return None

def calculate_meteor_score(candidate: str, reference: str) -> float:
    if not candidate.strip() or not reference.strip():
        return 0.0
    
    candidate_tokens = word_tokenize(candidate.lower())
    reference_tokens = word_tokenize(reference.lower())
    
    return meteor_score([reference_tokens], candidate_tokens)

def calculate_cider_score(candidate: str, reference: str) -> float:
    if not candidate.strip() or not reference.strip():
        return 0.0
    
    try:
        cider = Cider()
        candidates = {'0': [candidate.strip()]}
        references = {'0': [reference.strip()]}
        score, _ = cider.compute_score(references, candidates)
        return max(0.0, float(score))
    except Exception:
        return 0.0

def calculate_hybrid_text_score(candidate: str, reference: str) -> float:
    meteor = calculate_meteor_score(candidate, reference)
    cider = calculate_cider_score(candidate, reference)
    cider_norm = sigmoid_normalize(cider, k=1.0, x0=1.3)

    return meteor 

def parse_bbox_content_flexible(bbox_text: str) -> List[Dict[str, Any]]:
    if bbox_text.strip().lower() == "none":
        return []
    
    parsed_items = []
    lines = [line.strip() for line in bbox_text.split('\n') if line.strip()]
    
    for line in lines:
        try:
            if ':' in line:
                parts = line.split(':', 1)
                name_part = parts[0].strip().strip('"\'')
                coords_part = parts[1].strip()
                
                coords = ast.literal_eval(coords_part)
                if isinstance(coords, list) and len(coords) == 4:
                    parsed_items.append({"name": name_part, "coords": coords})
            else:
                coords = ast.literal_eval(line)
                if isinstance(coords, list) and len(coords) == 4:
                    parsed_items.append({"name": "", "coords": coords})
                    
        except (ValueError, SyntaxError):
            continue
    
    return parsed_items

def match_predicted_names_to_gt(bbox_items: List[Dict], answer_entities: List[str], gt_entities: List[str]) -> List[int]:
    if not bbox_items or not gt_entities:
        return []
    
    matches = []
    
    for item in bbox_items:
        pred_name = item["name"].lower().strip()
        best_match_idx = -1
        best_score = 0.0
        
        if pred_name:
            for i, gt_entity in enumerate(gt_entities):
                gt_name = gt_entity.lower().strip()
                
                if pred_name in gt_name or gt_name in pred_name:
                    score = 1.0
                else:
                    score = ratio(pred_name, gt_name)
                
                if score > best_score and score > 0.5:
                    best_score = score
                    best_match_idx = i
        
        elif answer_entities and len(matches) < len(answer_entities):
            answer_idx = len(matches)
            if answer_idx < len(answer_entities):
                answer_name = answer_entities[answer_idx].lower().strip()
                
                for i, gt_entity in enumerate(gt_entities):
                    gt_name = gt_entity.lower().strip()
                    
                    if answer_name in gt_name or gt_name in answer_name:
                        score = 1.0
                    else:
                        score = ratio(answer_name, gt_name)
                    
                    if score > best_score and score > 0.5:
                        best_score = score
                        best_match_idx = i
        
        matches.append(best_match_idx)
    
    return matches

def sam2_generate_masks(coordinates: List[Tuple[int, int, int, int]], 
                       rgb_image_path: str) -> List[np.ndarray]:
    if not coordinates or not os.path.exists(rgb_image_path):
        return []
    
    try:
        predictor = get_sam2_predictor()
        if predictor is None:
            print("SAM2 not available, using placeholder masks")
            return [np.zeros((480, 640), dtype=np.uint8) for _ in coordinates]
        
        image = cv2.imread(rgb_image_path)
        if image is None: return []
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        
        masks = []
        for (x1, y1, x2, y2) in coordinates:
            input_box = np.array([x1, y1, x2, y2])
            mask_output, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
            masks.append(mask_output[0].astype(np.uint8))
        
        return masks
        
    except Exception as e:
        print(f"Error in SAM2 mask generation: {e}")
        return []

def load_gt_mask_image(mask_path: str) -> np.ndarray:
    try:
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask_img if mask_img is not None else np.array([])
    except Exception:
        return np.array([])

def extract_gt_masks_from_image(mask_image: np.ndarray, masks_idx: List[int]) -> List[np.ndarray]:
    if mask_image.size == 0:
        return []
    
    gt_masks = []
    valid_indices = [idx for idx in masks_idx if idx > 0]
    for idx in valid_indices:
        mask = (mask_image == idx).astype(np.uint8)
        gt_masks.append(mask)
    
    return gt_masks

def calculate_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    if pred_mask.shape != gt_mask.shape:
        h, w = gt_mask.shape
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    return float(intersection / union) if union > 0 else 0.0

def ego_answer_rewards(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for i, (content, sol) in enumerate(zip(contents, solution)):
        sample_metadata = {}
        for key in ['metadata', 'n_gt', 'entities']:
            if key in kwargs:
                if isinstance(kwargs[key], list) and len(kwargs[key]) > i:
                    sample_metadata[key] = kwargs[key][i]
                else:
                    sample_metadata[key] = kwargs[key]
        
        # Integrated logic from the former ego_answer_reward function
        metadata = sample_metadata.get('metadata', {})
        n_gt = metadata.get('n_gt', 0)
        entities_gt = metadata.get('entities', [])
        
        answer_gen, _ = extract_three_stage_content(content)
        
        reward = 0.0
        if not answer_gen:
            reward = 0.0
        elif n_gt == 0:
            if "no suitable referring result" in answer_gen.lower():
                reward = 1.0
            else:
                reward = 0.0
        else:
            pred_entities = parse_answer_list(answer_gen)
            
            if pred_entities is None:
                reward = 0.0
            elif len(pred_entities) != len(entities_gt):
                reward = 0.1
            else:
                total_score = 0.0
                for i_entity in range(len(entities_gt)):
                    pred_entity = pred_entities[i_entity].lower().strip()
                    gt_entity = entities_gt[i_entity].lower().strip()
                    
                    if pred_entity in gt_entity or gt_entity in pred_entity:
                        score = 1.0
                    else:
                        score = ratio(pred_entity, gt_entity)
                    
                    total_score += score
                
                reward = total_score / len(entities_gt) if entities_gt else 0.0
        
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "./debug.log")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path + "_answer.txt", "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Answer Reward: {reward:.3f} -------------\n")
                f.write(f"Content: {content[:200]}...\n\n")
    
    return rewards

def ego_grounding_rewards(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for i, (content, sol) in enumerate(zip(contents, solution)):
        sample_metadata = {}
        for key in ['metadata', 'n_gt', 'entities', 'rgb_path', 'mask_path', 'masks_idx']:
            if key in kwargs:
                if isinstance(kwargs[key], list) and len(kwargs[key]) > i:
                    sample_metadata[key] = kwargs[key][i]
                else:
                    sample_metadata[key] = kwargs[key]
        
        # Integrated logic from the former ego_grounding_reward function
        metadata = sample_metadata.get('metadata', {})
        n_gt = metadata.get('n_gt', 0)
        entities_gt = metadata.get('entities', [])
        rgb_path = metadata.get('rgb_path', '')
        mask_path = metadata.get('mask_path', '')
        masks_idx = metadata.get('masks_idx', [])
        
        answer_gen, bbox_gen = extract_three_stage_content(content)
        
        reward = 0.0
        if not bbox_gen:
            reward = 0.0
        elif n_gt == 0:
            if bbox_gen.strip().lower() == "none":
                reward = 1.0
            else:
                reward = 0.0
        else:
            bbox_items = parse_bbox_content_flexible(bbox_gen)
            
            if not bbox_items:
                reward = 0.0
            else:
                valid_coords = []
                for item in bbox_items:
                    coords = item["coords"]
                    try:
                        x1, y1, x2, y2 = coords
                        if isinstance(x1, int) and isinstance(y1, int) and isinstance(x2, int) and isinstance(y2, int):
                            if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0:
                                valid_coords.append((x1, y1, x2, y2))
                    except (ValueError, TypeError):
                        continue
                
                if not valid_coords:
                    reward = 0.0
                elif not os.path.exists(rgb_path) or not os.path.exists(mask_path):
                    reward = 0.2
                else:
                    try:
                        answer_entities = parse_answer_list(answer_gen) if answer_gen else []
                        
                        matches = match_predicted_names_to_gt(bbox_items[:len(valid_coords)], 
                                                            answer_entities, entities_gt)
                        
                        pred_masks = sam2_generate_masks(valid_coords, rgb_path)
                        gt_masks = extract_gt_masks_from_image(load_gt_mask_image(mask_path), masks_idx)
                        
                        if not pred_masks or not gt_masks:
                            reward = 0.2
                        else:
                            total_iou = 0.0
                            valid_matches = 0
                            
                            for i_match, match_idx in enumerate(matches):
                                if match_idx >= 0 and i_match < len(pred_masks) and match_idx < len(gt_masks):
                                    iou = calculate_mask_iou(pred_masks[i_match], gt_masks[match_idx])
                                    total_iou += iou
                                    valid_matches += 1
                            
                            if valid_matches > 0:
                                reward = total_iou / valid_matches
                            else:
                                reward = 0.2
                                
                    except Exception as e:
                        print(f"Error in grounding calculation: {e}")
                        reward = 0.0
        
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "./debug.log")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path + "_grounding.txt", "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Grounding Reward: {reward:.3f} -------------\n")
                f.write(f"Content: {content[:200]}...\n")
                f.write(f"RGB: {sample_metadata.get('rgb_path', '')}\n")
                f.write(f"Mask: {sample_metadata.get('mask_path', '')}\n\n")
    
    return rewards

def extract_choice(text):
    text = text.upper()
    text = re.sub(r'\s+', ' ', text)
    choices = re.findall(r'(?<![A-Z])([A-Z])(?=[\.\,\?\!\:\;]|$)', text)
    if not choices:
        return None
    if len(choices) == 1:
        return choices[0]
    choice_scores = {choice: 0 for choice in choices}
    keywords = [
        '答案', '选择', '正确', '是', '对',
        'answer', 'correct', 'choose', 'select', 'right',
        '认为', '应该', '觉得', 'think', 'believe', 'should'
    ]
    for choice in choices:
        pos = text.find(choice)
        context = text[max(0, pos-20):min(len(text), pos+20)]
        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1
        if pos > len(text) * 0.7:
            choice_scores[choice] += 2
        if pos < len(text) - 1 and text[pos+1] in '。.!！,，':
            choice_scores[choice] += 1
    return max(choice_scores.items(), key=lambda x: x[1])[0]

def clean_text(text, exclue_chars=['\n', '\r']):
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        text = answer_matches[-1]
    
    for char in exclue_chars:
        if char in ['\n', '\r']:
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')
    
    return text.strip().rstrip('.').lower()

def default_accuracy_reward(content, sol, **kwargs):
    reward = 0.0
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()
    
    try:
        answer = parse(student_answer)
        if float(verify(answer, parse(ground_truth))) > 0:
            reward = 1.0
    except Exception:
        pass

    if reward == 0.0:
        try: 
            has_numbers = bool(re.search(r'\d', ground_truth))
            has_choices = extract_choice(ground_truth)
            
            if has_numbers:
                try:
                    content_clean, sol_clean = float(clean_text(student_answer)), float(clean_text(ground_truth))
                    reward = 1.0 if content_clean == sol_clean else 0.0
                except:
                    reward = ratio(clean_text(student_answer), clean_text(ground_truth))
            elif has_choices:
                correct_choice = has_choices.upper()
                student_choice = extract_choice(student_answer)
                if student_choice:
                    reward = 1.0 if student_choice == correct_choice else 0.0
            else:
                reward = ratio(clean_text(student_answer), clean_text(ground_truth))
        except Exception:
            pass

    return reward

def accuracy_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    accu_reward_methods = kwargs.get("accu_reward_method", ["default"] * len(contents))
    
    metadata_list = kwargs.get('metadata', [{} for _ in contents])

    for i, (content, sol, accu_reward_method) in enumerate(zip(contents, solution, accu_reward_methods)):
        sample_kwargs = {'metadata': metadata_list[i]}

        if accu_reward_method == "default":
            reward = default_accuracy_reward(content, sol)
        else:
            reward = default_accuracy_reward(content, sol)
            
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "./debug.log")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"accu_reward_method: {accu_reward_method}\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
        
    return rewards

def format_reward(completions, **kwargs):
    rewards = []
    completion_contents = [completion[0]["content"] for completion in completions]
    
    answer_pattern = r"<answer>.*?</answer>"
    bbox_pattern = r"<bbox>.*?</bbox>"

    for content in completion_contents:
        has_answer = re.search(answer_pattern, content, re.DOTALL) is not None
        has_bbox = re.search(bbox_pattern, content, re.DOTALL) is not None
        
        # 更宽松的格式奖励
        if has_answer and has_bbox:
            reward = 1.0
        elif has_answer:
            reward = 0.5  
        elif has_bbox:
            reward = 0.5  
        else:
            reward = 0.0
        rewards.append(reward)

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH", "./debug.log")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for i, content in enumerate(completion_contents):
                f.write(f"Content: {content}\n")
                f.write(f"Reward: {rewards[i]}\n")

    return rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "ego_answer": ego_answer_rewards, 
    "ego_grounding": ego_grounding_rewards,
}

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

SYSTEM_PROMPT = (
    "You are a multimodal assistant trained for referring image grounding. "
    "Follow the explicit output format `<answer>` and `<bbox>` as given in the dataset instructions. "
)

def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)
    question_prompt = vlm_module_cls.get_question_template(task_type=script_args.task_type)

    if script_args.is_reward_customized_from_vlm_module:
        reward_funcs = [vlm_module_cls.select_reward_func(func, script_args.task_type) for func in script_args.reward_funcs]
    else:
        reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    goal_embeddings = None
    if script_args.goal_embedding_path:
        if os.path.exists(script_args.goal_embedding_path):
            print(f"Loading goal embeddings from {script_args.goal_embedding_path}...")
            # goal_embeddings = np.load(script_args.goal_embedding_path)
            with np.load(script_args.goal_embedding_path) as data:
                goal_embeddings = {key: data[key] for key in data.files}
            print(f"Goal embeddings loaded successfully. Found {len(goal_embeddings)} entries.")
        else:
            print(f"Warning: Goal embedding file not found at {script_args.goal_embedding_path}")

    import json
    from datasets import Dataset
    
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")

    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    if script_args.reward_method is None:
        accu_reward_methods = ["default"] * len(data_files)
    else:
        accu_reward_methods = script_args.reward_method.split(":")
        assert len(accu_reward_methods) == len(data_files), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"

    all_data = []
    for data_file, image_folder, accu_reward_method in zip(data_files, image_folders, accu_reward_methods):
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                
                if 'image' in item:
                    if isinstance(item['image'], str):
                        item['image_path'] = item['image']
                    else:
                        raise ValueError(f"Unsupported image type: {type(item['image'])}")
                
                item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
                item['solution'] = item['conversations'][1]['value']
                
                if 'metadata' in item:
                    item.update(item['metadata'])
                
                item['accu_reward_method'] = item.get('accu_reward_method', accu_reward_method)
                all_data.append(item)

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        if 'image_path' in example and example['image_path'] is not None:
            full_image_path = example['image_path']
            if not os.path.exists(full_image_path):
                print(f"Warning: Image path does not exist: {full_image_path}")
            
            result_dict = {
                'image_path': full_image_path,
                'problem': example['problem'],
                'solution': example['solution'],
                'accu_reward_method': example['accu_reward_method'],
                
                'metadata': {
                    'n_gt': example.get('n_gt', 0),
                    'entities': example.get('entities', []),
                    'caption': example.get('caption', ''),
                    'rgb_path': example.get('rgb_path', ''),
                    'mask_path': example.get('mask_path', ''),
                    'masks_idx': example.get('masks_idx', []),
                },
                
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'text': None},
                        {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                    ]
                }]
            }

            if goal_embeddings is not None:
                image_id = os.path.basename(example['image_path'])
                if image_id in goal_embeddings:
                    embedding_vector = goal_embeddings[image_id]
                    result_dict['goal_caption_embedding'] = torch.from_numpy(embedding_vector.astype(np.float32))
                else:
                    print(f"Warning: Embedding for image ID '{image_id}' not found. Using zero vector.")
                    result_dict['goal_caption_embedding'] = torch.zeros(2048, dtype=torch.float32)
            else:
                result_dict['goal_caption_embedding'] = torch.zeros(2048, dtype=torch.float32)

            return result_dict

        else:
            return {
                'problem': example['problem'],
                'solution': example['solution'],
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                    ]
                }],
                'goal_caption_embedding': torch.zeros(2048, dtype=torch.float32)
            }

    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)

    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

    trainer_cls = VLMGRPOTrainer
    print("using trainer:", trainer_cls.__name__)
    initialize_tokenizer(model_args.model_name_or_path)
    
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


# if __name__ == "__main__":
#     parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
#     script_args, training_args, model_args = parser.parse_args_and_config()
#     if training_args.deepspeed and "zero3" in training_args.deepspeed:
#         print("zero3 is used, qwen2_5vl forward monkey patch is applied")
#         monkey_patch_qwen2_5vl_forward()
#     main(script_args, training_args, model_args)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    # 检查是否使用 zero2
    is_zero2 = training_args.deepspeed and "zero2" in training_args.deepspeed
    is_zero3 = training_args.deepspeed and "zero3" in training_args.deepspeed
    
    if is_zero3:
        print("Zero3 is used, applying zero3-specific monkey patch")
        monkey_patch_qwen2_5vl_forward()
    elif is_zero2:
        print("Zero2 is used, applying zero2-specific monkey patch")
        monkey_patch_qwen2_5vl_forward()  # 使用修改后的版本
    else:
        print("No DeepSpeed zero3/zero2 detected, applying standard monkey patch")
        monkey_patch_qwen2_5vl_forward()
    
    main(script_args, training_args, model_args)