# import os
# import json
# import re
# import argparse
# import numpy as np
# import ast
# import cv2
# from tqdm import tqdm
# from typing import List, Tuple, Dict

# # 文本评估导入
# import nltk
# from nltk.translate.meteor_score import meteor_score
# from nltk.tokenize import word_tokenize
# from pycocoevalcap.cider.cider import Cider

# # 自动下载NLTK资源
# print("正在检查NLTK资源...")
# for pkg in ["punkt", "wordnet"]:
#     try:
#         if pkg == "punkt": nltk.data.find(f"tokenizers/{pkg}")
#         else: nltk.data.find(f"corpora/{pkg}")
#     except LookupError:
#         print(f"未找到NLTK资源 '{pkg}'，正在下载...")
#         nltk.download(pkg)
# print("NLTK资源检查完毕。")

# # --- SAM2 集成 ---
# try:
#     from sam2.build_sam import build_sam2
#     from sam2.sam2_image_predictor import SAM2ImagePredictor
#     SAM2_AVAILABLE = True
# except ImportError:
#     print("警告: 未找到 'sam2' 库。掩码评估部分将被跳过。")
#     SAM2_AVAILABLE = False

# _sam2_predictor = None

# def get_sam2_predictor(checkpoint_path: str, model_cfg_name: str):
#     global _sam2_predictor
#     if not SAM2_AVAILABLE: return None
#     if _sam2_predictor is None:
#         print("首次初始化SAM2预测器...")
#         if not os.path.exists(checkpoint_path):
#             print(f"错误: 找不到SAM2检查点: {checkpoint_path}")
#             return None
#         try:
#             print(f"正在使用配置名 '{model_cfg_name}' 加载SAM2模型...")
#             sam2_model = build_sam2(model_cfg_name, checkpoint_path)
#             _sam2_predictor = SAM2ImagePredictor(sam2_model)
#             print("SAM2模型加载成功。")
#         except Exception as e:
#             print(f"加载SAM2模型失败: {e}")
#             _sam2_predictor = "failed"
#     return _sam2_predictor if _sam2_predictor != "failed" else None

# def sam2_generate_masks(coordinates: List[Tuple], rgb_image_path: str, predictor) -> List[np.ndarray]:
#     if not coordinates or not os.path.exists(rgb_image_path) or predictor is None: return []
#     try:
#         image = cv2.imread(rgb_image_path)
#         if image is None: return []
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         predictor.set_image(image_rgb)
#         masks = []
#         for (x1, y1, x2, y2) in coordinates:
#             input_box = np.array([x1, y1, x2, y2])
#             mask_output, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
#             masks.append(mask_output[0].astype(np.uint8))
#         return masks
#     except Exception as e:
#         tqdm.write(f"SAM2生成掩码时出错: {e}")
#         return []

# # --- 工具函数 ---
# def load_gt_mask_image(mask_path: str) -> np.ndarray:
#     return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# def extract_gt_masks_from_image(mask_image: np.ndarray, masks_idx: List[int]) -> List[np.ndarray]:
#     if mask_image is None or mask_image.size == 0: return []
#     valid_indices = [idx for idx in masks_idx if idx > 0]
#     return [(mask_image == idx).astype(np.uint8) for idx in valid_indices]

# def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
#     # mask1和mask2现在是布尔类型或0/1的numpy数组
#     intersection = np.logical_and(mask1, mask2).sum()
#     union = np.logical_or(mask1, mask2).sum()
#     return float(intersection) / float(union) if union > 0 else 0.0

# def parse_xml_tags(text: str) -> dict:
#     tags = {}
#     tags['analyzing'] = (m.group(1).strip() if (m := re.search(r'<analyzing>(.*?)</analyzing>', text, re.DOTALL)) else "")
#     tags['answer'] = (m.group(1).strip() if (m := re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)) else "")
#     tags['bbox'] = (m.group(1).strip() if (m := re.search(r'<bbox>(.*?)</bbox>', text, re.DOTALL)) else "")
#     return tags

# def extract_bbox_with_names(bbox_text: str) -> Dict[str, Tuple[int, int, int, int]]:
#     """
#     从bbox文本中提取物体名称和对应的坐标
#     输入格式: "the right hand": [189, 163, 274, 447]\n"the active kettle": [250, 161, 329, 270]
#     返回: {"the right hand": (189, 163, 274, 447), "the active kettle": (250, 161, 329, 270)}
#     """
#     if not bbox_text or bbox_text.strip().lower() == "none": 
#         return {}
    
#     objects_dict = {}
#     for line in bbox_text.split('\n'):
#         line = line.strip()
#         if not line: continue
        
#         try:
#             # 使用正则表达式匹配 "object name": [x1, y1, x2, y2] 格式
#             match = re.match(r'"([^"]+)":\s*\[([^\]]+)\]', line)
#             if match:
#                 object_name = match.group(1).strip()
#                 coords_str = match.group(2)
#                 coords = [int(float(c.strip())) for c in coords_str.split(',')]
#                 if len(coords) == 4:
#                     objects_dict[object_name] = tuple(coords)
#         except (ValueError, IndexError) as e:
#             tqdm.write(f"解析bbox行出错: {line}, 错误: {e}")
#             continue
    
#     return objects_dict

# def normalize_object_name(name: str) -> str:
#     """
#     标准化物体名称，用于匹配
#     """
#     name = name.lower().strip()
#     # 移除常见的修饰词
#     name = re.sub(r'\b(the|a|an)\b', '', name).strip()
#     name = re.sub(r'\b(active|interacting with right hand)\b', '', name).strip()
#     name = re.sub(r'\s+', ' ', name).strip()
#     return name

# def match_objects_by_name(pred_objects: Dict[str, Tuple], gt_entities: List[str]) -> Dict[str, str]:
#     """
#     根据名称匹配预测物体和GT物体
#     返回: {pred_name: gt_name} 的映射
#     """
#     matches = {}
    
#     # 标准化GT物体名称
#     normalized_gt = {normalize_object_name(gt): gt for gt in gt_entities}
    
#     for pred_name in pred_objects.keys():
#         normalized_pred = normalize_object_name(pred_name)
        
#         # 直接匹配
#         if normalized_pred in normalized_gt:
#             matches[pred_name] = normalized_gt[normalized_pred]
#             continue
        
#         # 模糊匹配：检查是否包含关键词
#         best_match = None
#         for norm_gt, orig_gt in normalized_gt.items():
#             if normalized_pred in norm_gt or norm_gt in normalized_pred:
#                 # 优先匹配更相似的
#                 if best_match is None or len(norm_gt) < len(normalize_object_name(best_match)):
#                     best_match = orig_gt
        
#         if best_match:
#             matches[pred_name] = best_match
    
#     return matches

# def evaluate_grounding_per_category(pred_objects: Dict[str, Tuple], 
#                                    gt_mask_image: np.ndarray, 
#                                    masks_idx: List[int], 
#                                    gt_entities: List[str],
#                                    sam_predictor, 
#                                    image_path: str) -> Tuple[List[float], float]:
#     """
#     按类别匹配并计算IoU，返回每个类别的IoU和平均IoU
#     """
#     if not pred_objects or gt_mask_image is None:
#         # 如果没有预测物体但有GT物体，所有类别IoU为0
#         num_gt_objects = len([idx for idx in masks_idx if idx > 0])
#         return [0.0] * num_gt_objects, 0.0
    
#     # 获取GT中的物体类别和对应的mask
#     valid_masks_idx = [idx for idx in masks_idx if idx > 0]
#     if len(valid_masks_idx) != len(gt_entities):
#         tqdm.write(f"警告: masks_idx长度({len(valid_masks_idx)})与entities长度({len(gt_entities)})不匹配")
    
#     gt_masks = {}  # {entity_name: mask_array}
#     for i, (idx, entity) in enumerate(zip(valid_masks_idx, gt_entities)):
#         gt_mask = (gt_mask_image == idx).astype(np.uint8)
#         gt_masks[entity] = gt_mask
    
#     # 匹配预测物体和GT物体
#     object_matches = match_objects_by_name(pred_objects, gt_entities)
    
#     # 逐类别计算IoU
#     category_ious = []
    
#     for gt_entity in gt_entities:
#         best_iou = 0.0
        
#         # 找到匹配的预测物体
#         matched_pred_names = [pred_name for pred_name, gt_name in object_matches.items() if gt_name == gt_entity]
        
#         if matched_pred_names:
#             gt_mask = gt_masks[gt_entity]
            
#             for pred_name in matched_pred_names:
#                 pred_bbox = pred_objects[pred_name]
#                 # 使用SAM2生成预测mask
#                 pred_masks = sam2_generate_masks([pred_bbox], image_path, sam_predictor)
                
#                 if pred_masks:
#                     pred_mask = pred_masks[0]
#                     iou = calculate_mask_iou(pred_mask, gt_mask)
#                     best_iou = max(best_iou, iou)
        
#         category_ious.append(best_iou)
    
#     avg_iou = np.mean(category_ious) if category_ious else 0.0
#     return category_ious, avg_iou

# def format_answer_sentence(answer_str: str) -> str:
#     if "no suitable referring result" in answer_str.lower(): return answer_str
#     try:
#         answer_list = ast.literal_eval(answer_str)
#         if not isinstance(answer_list, list): return answer_str
#         if not answer_list: return "There is no suitable referring result."
#         parts = []
#         for i, item in enumerate(answer_list):
#             clean_item = item.replace("the ", "").replace("active ", "").strip()
#             if len(answer_list) == 1: return f"The mask of the active {clean_item} interacting with right hand is <MSK_0>."
#             else:
#                 if i == 0: parts.append(f"the mask of the {clean_item} is <MSK_{i}>")
#                 else: parts.append(f"the mask of the active {clean_item} interacting with right hand is <MSK_{i}>")
#         return ", and ".join(parts) + "."
#     except (ValueError, SyntaxError):
#         return answer_str

# def calculate_meteor_score(candidate: str, reference: str) -> float:
#     if not candidate.strip() or not reference.strip(): return 0.0
#     candidate_tokens = word_tokenize(candidate.lower())
#     reference_tokens = word_tokenize(reference.lower())
#     return meteor_score([reference_tokens], candidate_tokens)

# # --- 主评估函数 ---
# def evaluate(args):
#     with open(args.results_path, 'r', encoding='utf-8') as f:
#         results = json.load(f)

#     cider_scorer = Cider()
#     analyze_preds_cider, analyze_gts_cider = {}, {}
#     answer_preds_cider, answer_gts_cider = {}, {}
#     analyze_meteor_scores, answer_meteor_scores = [], []
#     category_ious_all = []  # 存储所有样本的每类别IoU
#     avg_ious_all = []       # 存储所有样本的平均IoU

#     sam_predictor = get_sam2_predictor(args.sam_checkpoint, args.sam_config)

#     print("开始评估...")
#     for i, res in enumerate(tqdm(results, desc="评估进度")):
#         pred_text, gt_text = res.get("prediction", ""), res.get("ground_truth", "")
#         image_path = res.get("image_path")
#         metadata = res.get("metadata", {})
#         sample_id = str(i)
#         pred_tags, gt_tags = parse_xml_tags(pred_text), parse_xml_tags(gt_text)

#         # 1. 文本评估
#         if pred_tags['analyzing'] and gt_tags['analyzing']:
#             analyze_preds_cider[sample_id] = [pred_tags['analyzing']]
#             analyze_gts_cider[sample_id] = [gt_tags['analyzing']]
#             analyze_meteor_scores.append(calculate_meteor_score(pred_tags['analyzing'], gt_tags['analyzing']))
#         if pred_tags['answer'] and gt_tags['answer']:
#             formatted_pred_answer = format_answer_sentence(pred_tags['answer'])
#             answer_preds_cider[sample_id] = [formatted_pred_answer]
#             answer_gts_cider[sample_id] = [gt_tags['answer']]
#             answer_meteor_scores.append(calculate_meteor_score(formatted_pred_answer, gt_tags['answer']))

#         # 2. 分割评估 (按类别计算IoU)
#         pred_objects = extract_bbox_with_names(pred_tags['bbox'])
        
#         # 获取GT信息
#         masks_idx = metadata.get('masks_idx', [])
#         gt_entities = metadata.get('entities', [])
        
#         if not image_path or not os.path.exists(image_path):
#             # 图像路径无效，该样本所有类别IoU为0
#             num_gt_objects = len([idx for idx in masks_idx if idx > 0])
#             category_ious_all.append([0.0] * num_gt_objects)
#             avg_ious_all.append(0.0)
#             continue
        
#         # 加载GT mask图像
#         gt_mask_path = image_path.replace('/RGB/', '/mask/').replace('.jpg', '.png')
#         gt_mask_image = None
#         if os.path.exists(gt_mask_path):
#             gt_mask_image = load_gt_mask_image(gt_mask_path)
        
#         # 计算每个类别的IoU
#         category_ious, avg_iou = evaluate_grounding_per_category(
#             pred_objects, gt_mask_image, masks_idx, gt_entities, sam_predictor, image_path
#         )
        
#         category_ious_all.append(category_ious)
#         avg_ious_all.append(avg_iou)

#     # --- 计算最终的grounding指标 ---
#     # 方法1: 先计算每个样本的平均IoU，再对所有样本求平均（与原代码一致）
#     overall_avg_iou = np.mean(avg_ious_all) if avg_ious_all else 0.0
    
#     # 方法2: 按类别统计（可选，用于更详细的分析）
#     if category_ious_all:
#         # 将所有样本的每类别IoU合并
#         all_category_ious = []
#         for sample_ious in category_ious_all:
#             all_category_ious.extend(sample_ious)
#         per_category_avg_iou = np.mean(all_category_ious) if all_category_ious else 0.0
#     else:
#         per_category_avg_iou = 0.0

#     # --- 打印最终平均分数 ---
#     print("\n--- 评估结果 ---\n")
#     if analyze_preds_cider:
#         analyze_cider_score, _ = cider_scorer.compute_score(analyze_gts_cider, analyze_preds_cider)
#         print(f"【<analyzing> 内容评估】")
#         print(f"  - 平均 CIDEr:  {analyze_cider_score:.4f}")
#         print(f"  - 平均 METEOR: {np.mean(analyze_meteor_scores):.4f}\n")
#     if answer_preds_cider:
#         answer_cider_score, _ = cider_scorer.compute_score(answer_gts_cider, answer_preds_cider)
#         print(f"【<answer> 内容评估 (格式化后)】")
#         print(f"  - 平均 CIDEr:  {answer_cider_score:.4f}")
#         print(f"  - 平均 METEOR: {np.mean(answer_meteor_scores):.4f}\n")
#     if avg_ious_all:
#         print(f"【分割评估 (按类别IoU via SAM2)】")
#         print(f"  - 样本平均IoU: {overall_avg_iou:.4f}")
#         print(f"  - 类别平均IoU: {per_category_avg_iou:.4f}")
#         print(f"  - (基于 {len(avg_ious_all)} 个样本的类别匹配IoU计算)\n")
#     else:
#         print("【分割评估 (按类别IoU via SAM2)】")
#         print("  - 未计算任何IoU分数。请检查数据和路径。\n")
#     print("🎉 评估完成！")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="评估VLM推理结果，包含NLTK文本指标和SAM2按类别IoU。")
#     parser.add_argument("results_path", type=str, help="指向推理结果JSON文件的路径。")
#     parser.add_argument("--sam_checkpoint", type=str, default="/root/sam2/checkpoints/sam2_hiera_large.pt", help="SAM2模型检查点路径。")
#     parser.add_argument("--sam_config", type=str, default="sam2_hiera_l.yaml", help="SAM2模型配置名称。")
#     args = parser.parse_args()
#     evaluate(args)





# import os
# import json
# import re
# import argparse
# import numpy as np
# import ast
# import cv2
# from tqdm import tqdm
# from typing import List, Tuple, Dict

# # 文本评估导入
# import nltk
# from nltk.translate.meteor_score import meteor_score
# from nltk.tokenize import word_tokenize
# from pycocoevalcap.cider.cider import Cider

# # 自动下载NLTK资源
# print("正在检查NLTK资源...")
# for pkg in ["punkt", "wordnet"]:
#     try:
#         if pkg == "punkt": nltk.data.find(f"tokenizers/{pkg}")
#         else: nltk.data.find(f"corpora/{pkg}")
#     except LookupError:
#         print(f"未找到NLTK资源 '{pkg}'，正在下载...")
#         nltk.download(pkg)
# print("NLTK资源检查完毕。")

# # --- SAM2 集成 ---
# try:
#     from sam2.build_sam import build_sam2
#     from sam2.sam2_image_predictor import SAM2ImagePredictor
#     SAM2_AVAILABLE = True
# except ImportError:
#     print("警告: 未找到 'sam2' 库。掩码评估部分将被跳过。")
#     SAM2_AVAILABLE = False

# _sam2_predictor = None

# def get_sam2_predictor(checkpoint_path: str, model_cfg_name: str):
#     global _sam2_predictor
#     if not SAM2_AVAILABLE: return None
#     if _sam2_predictor is None:
#         print("首次初始化SAM2预测器...")
#         if not os.path.exists(checkpoint_path):
#             print(f"错误: 找不到SAM2检查点: {checkpoint_path}")
#             return None
#         try:
#             print(f"正在使用配置名 '{model_cfg_name}' 加载SAM2模型...")
#             sam2_model = build_sam2(model_cfg_name, checkpoint_path)
#             _sam2_predictor = SAM2ImagePredictor(sam2_model)
#             print("SAM2模型加载成功。")
#         except Exception as e:
#             print(f"加载SAM2模型失败: {e}")
#             _sam2_predictor = "failed"
#     return _sam2_predictor if _sam2_predictor != "failed" else None

# def sam2_generate_masks(coordinates: List[Tuple], rgb_image_path: str, predictor) -> List[np.ndarray]:
#     if not coordinates or not os.path.exists(rgb_image_path) or predictor is None: return []
#     try:
#         image = cv2.imread(rgb_image_path)
#         if image is None: return []
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         predictor.set_image(image_rgb)
#         masks = []
#         for (x1, y1, x2, y2) in coordinates:
#             input_box = np.array([x1, y1, x2, y2])
#             mask_output, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
#             masks.append(mask_output[0].astype(np.uint8))
#         return masks
#     except Exception as e:
#         tqdm.write(f"SAM2生成掩码时出错: {e}")
#         return []

# # --- 工具函数 ---
# def load_gt_mask_image(mask_path: str) -> np.ndarray:
#     return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# def extract_gt_masks_from_image(mask_image: np.ndarray, masks_idx: List[int]) -> List[np.ndarray]:
#     if mask_image is None or mask_image.size == 0: return []
#     valid_indices = [idx for idx in masks_idx if idx > 0]
#     return [(mask_image == idx).astype(np.uint8) for idx in valid_indices]

# def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
#     # mask1和mask2现在是布尔类型或0/1的numpy数组
#     intersection = np.logical_and(mask1, mask2).sum()
#     union = np.logical_or(mask1, mask2).sum()
#     return float(intersection) / float(union) if union > 0 else 0.0

# def parse_xml_tags(text: str) -> dict:
#     tags = {}
#     tags['analyzing'] = (m.group(1).strip() if (m := re.search(r'<analyzing>(.*?)</analyzing>', text, re.DOTALL)) else "")
#     tags['answer'] = (m.group(1).strip() if (m := re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)) else "")
#     tags['bbox'] = (m.group(1).strip() if (m := re.search(r'<bbox>(.*?)</bbox>', text, re.DOTALL)) else "")
#     return tags

# def is_no_object_answer(answer_text: str) -> bool:
#     """
#     判断answer是否表示"没有物体需要分割"
#     """
#     if not answer_text:
#         return False
    
#     answer_lower = answer_text.lower().strip()
#     no_object_patterns = [
#         "there is no suitable referring result",
#         "no suitable referring result",
#         "there are no suitable referring results",
#         "no suitable referring results",
#         "there is no object",
#         "no object",
#         "there is nothing",
#         "nothing",
#         "none"
#     ]
    
#     return any(pattern in answer_lower for pattern in no_object_patterns)

# def is_no_object_bbox(bbox_text: str) -> bool:
#     """
#     判断bbox是否表示"没有物体"
#     """
#     if not bbox_text:
#         return True
    
#     bbox_lower = bbox_text.lower().strip()
#     return bbox_lower in ["none", "null", ""]

# def extract_bbox_with_names(bbox_text: str) -> Dict[str, Tuple[int, int, int, int]]:
#     """
#     从bbox文本中提取物体名称和对应的坐标
#     输入格式: "the right hand": [189, 163, 274, 447]\n"the active kettle": [250, 161, 329, 270]
#     返回: {"the right hand": (189, 163, 274, 447), "the active kettle": (250, 161, 329, 270)}
#     """
#     if not bbox_text or bbox_text.strip().lower() == "none": 
#         return {}
    
#     objects_dict = {}
#     for line in bbox_text.split('\n'):
#         line = line.strip()
#         if not line: continue
        
#         try:
#             # 使用正则表达式匹配 "object name": [x1, y1, x2, y2] 格式
#             match = re.match(r'"([^"]+)":\s*\[([^\]]+)\]', line)
#             if match:
#                 object_name = match.group(1).strip()
#                 coords_str = match.group(2)
#                 coords = [int(float(c.strip())) for c in coords_str.split(',')]
#                 if len(coords) == 4:
#                     objects_dict[object_name] = tuple(coords)
#         except (ValueError, IndexError) as e:
#             tqdm.write(f"解析bbox行出错: {line}, 错误: {e}")
#             continue
    
#     return objects_dict

# def normalize_object_name(name: str) -> str:
#     """
#     标准化物体名称，用于匹配
#     """
#     name = name.lower().strip()
#     # 移除常见的修饰词
#     name = re.sub(r'\b(the|a|an)\b', '', name).strip()
#     name = re.sub(r'\b(active|interacting with right hand)\b', '', name).strip()
#     name = re.sub(r'\s+', ' ', name).strip()
#     return name

# def match_objects_by_name(pred_objects: Dict[str, Tuple], gt_entities: List[str]) -> Dict[str, str]:
#     """
#     根据名称匹配预测物体和GT物体
#     返回: {pred_name: gt_name} 的映射
#     """
#     matches = {}
    
#     # 标准化GT物体名称
#     normalized_gt = {normalize_object_name(gt): gt for gt in gt_entities}
    
#     for pred_name in pred_objects.keys():
#         normalized_pred = normalize_object_name(pred_name)
        
#         # 直接匹配
#         if normalized_pred in normalized_gt:
#             matches[pred_name] = normalized_gt[normalized_pred]
#             continue
        
#         # 模糊匹配：检查是否包含关键词
#         best_match = None
#         for norm_gt, orig_gt in normalized_gt.items():
#             if normalized_pred in norm_gt or norm_gt in normalized_pred:
#                 # 优先匹配更相似的
#                 if best_match is None or len(norm_gt) < len(normalize_object_name(best_match)):
#                     best_match = orig_gt
        
#         if best_match:
#             matches[pred_name] = best_match
    
#     return matches

# def evaluate_grounding_per_category(pred_objects: Dict[str, Tuple], 
#                                    gt_mask_image: np.ndarray, 
#                                    masks_idx: List[int], 
#                                    gt_entities: List[str],
#                                    sam_predictor, 
#                                    image_path: str,
#                                    gt_has_no_objects: bool = False,
#                                    pred_has_no_objects: bool = False) -> Tuple[List[float], float]:
#     """
#     按类别匹配并计算IoU，返回每个类别的IoU和平均IoU
    
#     Args:
#         pred_objects: 预测的物体字典
#         gt_mask_image: GT mask图像
#         masks_idx: GT mask索引
#         gt_entities: GT物体列表
#         sam_predictor: SAM2预测器
#         image_path: 图像路径
#         gt_has_no_objects: GT是否表示没有物体
#         pred_has_no_objects: 预测是否表示没有物体
#     """
    
#     # 特殊情况1: GT和预测都表示没有物体 -> 完美匹配，给满分
#     if gt_has_no_objects and pred_has_no_objects:
#         return [1.0], 1.0
    
#     # 特殊情况2: GT表示没有物体，但预测有物体 -> 错误，给0分
#     if gt_has_no_objects and not pred_has_no_objects:
#         return [0.0], 0.0
    
#     # 特殊情况3: GT有物体，但预测表示没有物体 -> 错误，每个GT物体都是0分
#     if not gt_has_no_objects and pred_has_no_objects:
#         num_gt_objects = len([idx for idx in masks_idx if idx > 0])
#         return [0.0] * num_gt_objects, 0.0
    
#     # 正常情况: GT和预测都有物体，按原逻辑计算IoU
#     if not pred_objects or gt_mask_image is None:
#         # 如果没有预测物体但有GT物体，所有类别IoU为0
#         num_gt_objects = len([idx for idx in masks_idx if idx > 0])
#         return [0.0] * num_gt_objects, 0.0
    
#     # 获取GT中的物体类别和对应的mask
#     valid_masks_idx = [idx for idx in masks_idx if idx > 0]
#     if len(valid_masks_idx) != len(gt_entities):
#         tqdm.write(f"警告: masks_idx长度({len(valid_masks_idx)})与entities长度({len(gt_entities)})不匹配")
    
#     gt_masks = {}  # {entity_name: mask_array}
#     for i, (idx, entity) in enumerate(zip(valid_masks_idx, gt_entities)):
#         gt_mask = (gt_mask_image == idx).astype(np.uint8)
#         gt_masks[entity] = gt_mask
    
#     # 匹配预测物体和GT物体
#     object_matches = match_objects_by_name(pred_objects, gt_entities)
    
#     # 逐类别计算IoU
#     category_ious = []
    
#     for gt_entity in gt_entities:
#         best_iou = 0.0
        
#         # 找到匹配的预测物体
#         matched_pred_names = [pred_name for pred_name, gt_name in object_matches.items() if gt_name == gt_entity]
        
#         if matched_pred_names:
#             gt_mask = gt_masks[gt_entity]
            
#             for pred_name in matched_pred_names:
#                 pred_bbox = pred_objects[pred_name]
#                 # 使用SAM2生成预测mask
#                 pred_masks = sam2_generate_masks([pred_bbox], image_path, sam_predictor)
                
#                 if pred_masks:
#                     pred_mask = pred_masks[0]
#                     iou = calculate_mask_iou(pred_mask, gt_mask)
#                     best_iou = max(best_iou, iou)
        
#         category_ious.append(best_iou)
    
#     avg_iou = np.mean(category_ious) if category_ious else 0.0
#     return category_ious, avg_iou

# def format_answer_sentence(answer_str: str) -> str:
#     if "no suitable referring result" in answer_str.lower(): return answer_str
#     try:
#         answer_list = ast.literal_eval(answer_str)
#         if not isinstance(answer_list, list): return answer_str
#         if not answer_list: return "There is no suitable referring result."
#         parts = []
#         for i, item in enumerate(answer_list):
#             clean_item = item.replace("the ", "").replace("active ", "").strip()
#             if len(answer_list) == 1: return f"The mask of the active {clean_item} interacting with right hand is <MSK_0>."
#             else:
#                 if i == 0: parts.append(f"the mask of the {clean_item} is <MSK_{i}>")
#                 else: parts.append(f"the mask of the active {clean_item} interacting with right hand is <MSK_{i}>")
#         return ", and ".join(parts) + "."
#     except (ValueError, SyntaxError):
#         return answer_str

# def calculate_meteor_score(candidate: str, reference: str) -> float:
#     if not candidate.strip() or not reference.strip(): return 0.0
#     candidate_tokens = word_tokenize(candidate.lower())
#     reference_tokens = word_tokenize(reference.lower())
#     return meteor_score([reference_tokens], candidate_tokens)

# # --- 主评估函数 ---
# def evaluate(args):
#     with open(args.results_path, 'r', encoding='utf-8') as f:
#         results = json.load(f)

#     cider_scorer = Cider()
#     analyze_preds_cider, analyze_gts_cider = {}, {}
#     answer_preds_cider, answer_gts_cider = {}, {}
#     analyze_meteor_scores, answer_meteor_scores = [], []
#     category_ious_all = []  # 存储所有样本的每类别IoU
#     avg_ious_all = []       # 存储所有样本的平均IoU
    
#     # 统计特殊情况
#     perfect_no_object_matches = 0  # GT和预测都正确表示没有物体
#     false_positive_objects = 0     # GT没有物体但预测有物体
#     false_negative_objects = 0     # GT有物体但预测说没有物体

#     sam_predictor = get_sam2_predictor(args.sam_checkpoint, args.sam_config)

#     print("开始评估...")
#     for i, res in enumerate(tqdm(results, desc="评估进度")):
#         pred_text, gt_text = res.get("prediction", ""), res.get("ground_truth", "")
#         image_path = res.get("image_path")
#         metadata = res.get("metadata", {})
#         sample_id = str(i)
#         pred_tags, gt_tags = parse_xml_tags(pred_text), parse_xml_tags(gt_text)

#         # 1. 文本评估
#         if pred_tags['analyzing'] and gt_tags['analyzing']:
#             analyze_preds_cider[sample_id] = [pred_tags['analyzing']]
#             analyze_gts_cider[sample_id] = [gt_tags['analyzing']]
#             analyze_meteor_scores.append(calculate_meteor_score(pred_tags['analyzing'], gt_tags['analyzing']))
#         if pred_tags['answer'] and gt_tags['answer']:
#             formatted_pred_answer = format_answer_sentence(pred_tags['answer'])
#             answer_preds_cider[sample_id] = [formatted_pred_answer]
#             answer_gts_cider[sample_id] = [gt_tags['answer']]
#             answer_meteor_scores.append(calculate_meteor_score(formatted_pred_answer, gt_tags['answer']))

#         # 2. 分割评估 (按类别计算IoU)
#         pred_objects = extract_bbox_with_names(pred_tags['bbox'])
        
#         # 判断是否为"没有物体"的情况
#         gt_has_no_objects = is_no_object_answer(gt_tags['answer']) or is_no_object_bbox(gt_tags['bbox'])
#         pred_has_no_objects = is_no_object_answer(pred_tags['answer']) or is_no_object_bbox(pred_tags['bbox']) or len(pred_objects) == 0
        
#         # 统计特殊情况
#         if gt_has_no_objects and pred_has_no_objects:
#             perfect_no_object_matches += 1
#         elif gt_has_no_objects and not pred_has_no_objects:
#             false_positive_objects += 1
#         elif not gt_has_no_objects and pred_has_no_objects:
#             false_negative_objects += 1
        
#         # 获取GT信息
#         masks_idx = metadata.get('masks_idx', [])
#         gt_entities = metadata.get('entities', [])
        
#         if not image_path or not os.path.exists(image_path):
#             # 图像路径无效的处理
#             if gt_has_no_objects and pred_has_no_objects:
#                 category_ious_all.append([1.0])
#                 avg_ious_all.append(1.0)
#             else:
#                 num_gt_objects = len([idx for idx in masks_idx if idx > 0]) if not gt_has_no_objects else 1
#                 category_ious_all.append([0.0] * num_gt_objects)
#                 avg_ious_all.append(0.0)
#             continue
        
#         # 加载GT mask图像
#         gt_mask_path = image_path.replace('/RGB/', '/mask/').replace('.jpg', '.png')
#         gt_mask_image = None
#         if os.path.exists(gt_mask_path):
#             gt_mask_image = load_gt_mask_image(gt_mask_path)
        
#         # 计算每个类别的IoU（考虑无物体情况）
#         category_ious, avg_iou = evaluate_grounding_per_category(
#             pred_objects, gt_mask_image, masks_idx, gt_entities, sam_predictor, image_path,
#             gt_has_no_objects, pred_has_no_objects
#         )
        
#         category_ious_all.append(category_ious)
#         avg_ious_all.append(avg_iou)

#     # --- 计算最终的grounding指标 ---
#     # 方法1: 先计算每个样本的平均IoU，再对所有样本求平均（与原代码一致）
#     overall_avg_iou = np.mean(avg_ious_all) if avg_ious_all else 0.0
    
#     # 方法2: 按类别统计（可选，用于更详细的分析）
#     if category_ious_all:
#         # 将所有样本的每类别IoU合并
#         all_category_ious = []
#         for sample_ious in category_ious_all:
#             all_category_ious.extend(sample_ious)
#         per_category_avg_iou = np.mean(all_category_ious) if all_category_ious else 0.0
#     else:
#         per_category_avg_iou = 0.0

#     # --- 打印最终平均分数 ---
#     print("\n--- 评估结果 ---\n")
#     if analyze_preds_cider:
#         analyze_cider_score, _ = cider_scorer.compute_score(analyze_gts_cider, analyze_preds_cider)
#         print(f"【<analyzing> 内容评估】")
#         print(f"  - 平均 CIDEr:  {analyze_cider_score:.4f}")
#         print(f"  - 平均 METEOR: {np.mean(analyze_meteor_scores):.4f}\n")
#     if answer_preds_cider:
#         answer_cider_score, _ = cider_scorer.compute_score(answer_gts_cider, answer_preds_cider)
#         print(f"【<answer> 内容评估 (格式化后)】")
#         print(f"  - 平均 CIDEr:  {answer_cider_score:.4f}")
#         print(f"  - 平均 METEOR: {np.mean(answer_meteor_scores):.4f}\n")
#     if avg_ious_all:
#         print(f"【分割评估 (按类别IoU via SAM2)】")
#         print(f"  - 样本平均IoU: {overall_avg_iou:.4f}")
#         print(f"  - 类别平均IoU: {per_category_avg_iou:.4f}")
#         print(f"  - (基于 {len(avg_ious_all)} 个样本的类别匹配IoU计算)")
#         print(f"  - 完美无物体匹配: {perfect_no_object_matches} 个")
#         print(f"  - 误报物体: {false_positive_objects} 个")
#         print(f"  - 漏报物体: {false_negative_objects} 个\n")
#     else:
#         print("【分割评估 (按类别IoU via SAM2)】")
#         print("  - 未计算任何IoU分数。请检查数据和路径。\n")
#     print("🎉 评估完成！")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="评估VLM推理结果，包含NLTK文本指标和SAM2按类别IoU。")
#     parser.add_argument("results_path", type=str, help="指向推理结果JSON文件的路径。")
#     parser.add_argument("--sam_checkpoint", type=str, default="/root/sam2/checkpoints/sam2_hiera_large.pt", help="SAM2模型检查点路径。")
#     parser.add_argument("--sam_config", type=str, default="sam2_hiera_l.yaml", help="SAM2模型配置名称。")
#     args = parser.parse_args()
#     evaluate(args)


import os
import json
import re
import numpy as np
import ast
import cv2
from tqdm import tqdm
from typing import List, Tuple, Dict

# ========================================
#             手动配置部分
# ========================================

RESULTS_PATH = "/root/inference_results_ablation_format_answer_test_metadata.json"
SAM_CHECKPOINT = "/root/sam2/checkpoints/sam2_hiera_large.pt"
SAM_CONFIG = "sam2_hiera_l.yaml"

# ========================================
#             依赖导入
# ========================================

import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from pycocoevalcap.cider.cider import Cider

# 自动检查NLTK资源
print("正在检查NLTK资源...")
for pkg in ["punkt", "wordnet"]:
    try:
        if pkg == "punkt":
            nltk.data.find(f"tokenizers/{pkg}")
        else:
            nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        print(f"未找到NLTK资源 '{pkg}'，正在下载...")
        nltk.download(pkg)
print("NLTK资源检查完毕。")

# --- SAM2 集成 ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("⚠️ 警告: 未找到 'sam2' 库。掩码评估部分将被跳过。")
    SAM2_AVAILABLE = False

_sam2_predictor = None

def get_sam2_predictor(checkpoint_path: str, model_cfg_name: str):
    """初始化 SAM2 模型"""
    global _sam2_predictor
    if not SAM2_AVAILABLE:
        return None
    if _sam2_predictor is None:
        print("首次初始化SAM2预测器...")
        if not os.path.exists(checkpoint_path):
            print(f"❌ 错误: 找不到SAM2检查点: {checkpoint_path}")
            return None
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            print(f"加载SAM2模型配置 '{model_cfg_name}'...")
            sam2_model = build_sam2(model_cfg_name, checkpoint_path)
            _sam2_predictor = SAM2ImagePredictor(sam2_model)
            print("✅ SAM2加载成功。")
        except Exception as e:
            print(f"❌ 加载SAM2模型失败: {e}")
            _sam2_predictor = "failed"
    return _sam2_predictor if _sam2_predictor != "failed" else None

# ========================================
#             工具函数区
# ========================================

def sam2_generate_masks(coordinates: List[Tuple], rgb_image_path: str, predictor) -> List[np.ndarray]:
    if not coordinates or not os.path.exists(rgb_image_path) or predictor is None:
        return []
    try:
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
        tqdm.write(f"SAM2生成掩码时出错: {e}")
        return []

def load_gt_mask_image(mask_path: str) -> np.ndarray:
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法加载掩码图像: {mask_path}")
    return img

def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection) / float(union) if union > 0 else 0.0

def parse_xml_tags(text: str) -> dict:
    tags = {}
    tags['answer'] = (m.group(1).strip() if (m := re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)) else "")
    bbox_matches = re.findall(r'<bbox>(.*?)</bbox>', text, re.DOTALL)
    tags['bbox'] = "\n".join([b.strip() for b in bbox_matches]) if bbox_matches else ""
    return tags

def is_no_object_answer(answer_text: str) -> bool:
    if not answer_text:
        return False
    answer_lower = answer_text.lower().strip()
    no_object_patterns = [
        "there is no suitable referring result",
    ]
    return any(p in answer_lower for p in no_object_patterns) or answer_lower == "none"

def is_no_object_bbox(bbox_text: str) -> bool:
    if not bbox_text:
        return True
    bbox_lower = bbox_text.lower().strip()
    return bbox_lower in ["none", "null", ""]

def extract_bbox_with_names(bbox_text: str) -> Dict[str, Tuple[int, int, int, int]]:
    if not bbox_text or is_no_object_bbox(bbox_text):
        return {}
    objects_dict = {}
    pattern = re.compile(r'("?)(.+?)\1:\s*\[\s*([\d\s,.-]+)\s*\]')
    
    for line in bbox_text.split('\n'):
        line = line.strip()
        if not line: continue
        # 使用 re.search 更加健壮，以防行首有意外字符
        match = pattern.search(line)
        if match:
            try:
                obj_name = match.group(2).strip()
                coords_str = match.group(3)
                coords = [int(float(c.strip())) for c in coords_str.split(',')]
                if len(coords) == 4:
                    objects_dict[obj_name] = tuple(coords)
            except (ValueError, IndexError) as e:
                tqdm.write(f"解析bbox坐标时出错: '{line}', 错误: {e}")
    return objects_dict

def normalize_object_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r'\b(the|a|an)\b', '', name)
    name = re.sub(r'\b(active|interacting with right hand|interacting with left hand)\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def match_objects_by_name(pred_objects: Dict[str, Tuple], gt_entities: List[str]) -> Dict[str, str]:
    matches = {}
    normalized_gt = {normalize_object_name(gt): gt for gt in gt_entities}
    unmatched_gt = set(normalized_gt.keys())

    for pred_name in pred_objects.keys():
        normalized_pred = normalize_object_name(pred_name)
        if normalized_pred in unmatched_gt:
            matches[pred_name] = normalized_gt[normalized_pred]
            unmatched_gt.remove(normalized_pred)
            continue
    
    for pred_name in pred_objects.keys():
        if pred_name in matches: continue
        normalized_pred = normalize_object_name(pred_name)
        for norm_gt in list(unmatched_gt):
            if normalized_pred in norm_gt or norm_gt in normalized_pred:
                matches[pred_name] = normalized_gt[norm_gt]
                unmatched_gt.remove(norm_gt)
                break
    return matches

def format_answer_sentence(answer_str: str) -> str:
    if is_no_object_answer(answer_str):
        return "There is no suitable referring result."
    try:
        answer_list = ast.literal_eval(answer_str)
        if not isinstance(answer_list, list):
            return answer_str
    except (ValueError, SyntaxError):
        return answer_str

    if not answer_list:
        return "There is no suitable referring result."
    
    parts = []
    for i, item in enumerate(answer_list):
        clean_item = item.lower().replace("the ", "").replace("active ", "").strip()
        if "hand" in clean_item:
            parts.append(f"the mask of the {clean_item} is <MSK_{i}>")
        else:
            parts.append(f"the mask of the active {clean_item} interacting with right hand is <MSK_{i}>")

    if len(parts) == 1:
        return parts[0].replace(" is <MSK_0>", " is <MSK_0>.")
    else:
        return ", and ".join(parts) + "."

def calculate_meteor_score(candidate: str, reference: str) -> float:
    if not candidate.strip() or not reference.strip():
        return 0.0 if candidate.strip() != reference.strip() else 1.0
    return meteor_score([word_tokenize(reference.lower())], word_tokenize(candidate.lower()))

# ========================================
#             主评估逻辑
# ========================================

def evaluate():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    cider_scorer = Cider()
    answer_preds_cider, answer_gts_cider = {}, {}
    answer_meteor_scores = []
    
    sample_level_ious = [] 
    instance_level_ious = []

    perfect_no_object_matches = 0
    false_positive_samples = 0 
    false_negative_objects = 0

    sam_predictor = get_sam2_predictor(SAM_CHECKPOINT, SAM_CONFIG)
    
    # *** 新增诊断标志 ***
    metadata_warning_printed = False

    print("\n开始评估 (answer + grounding)...")
    for i, res in enumerate(tqdm(results, desc="评估进度")):
        pred_text = res.get("prediction", "")
        gt_text = res.get("ground_truth", "")
        image_path = res.get("image_path")
        metadata = res.get("metadata", {})
        sample_id = str(i)

        pred_tags, gt_tags = parse_xml_tags(pred_text), parse_xml_tags(gt_text)

        # ===== 文本评估 (<answer>内容) =====
        if pred_tags["answer"] and gt_tags["answer"]:
            formatted_pred_answer = format_answer_sentence(pred_tags["answer"])
            clean_gt_answer = re.sub(r'<analyzing>.*?</analyzing>\s*', '', gt_tags["answer"]).strip()
            answer_preds_cider[sample_id] = [formatted_pred_answer]
            answer_gts_cider[sample_id] = [clean_gt_answer]
            answer_meteor_scores.append(calculate_meteor_score(formatted_pred_answer, clean_gt_answer))

        # ===== Grounding IoU 评估 =====
        pred_objects = extract_bbox_with_names(pred_tags["bbox"])

        gt_has_no_objects = is_no_object_answer(gt_tags["answer"]) or is_no_object_bbox(gt_tags["bbox"])
        pred_has_no_objects = is_no_object_answer(pred_tags["answer"]) or not pred_objects
        
        if gt_has_no_objects and pred_has_no_objects:
            perfect_no_object_matches += 1
            sample_level_ious.append(1.0)
            continue

        if gt_has_no_objects and not pred_has_no_objects:
            false_positive_samples += 1
            sample_level_ious.append(0.0)
            continue

        if not gt_has_no_objects and pred_has_no_objects:
            # 这里的 metadata.get("entities") 可能也为空，但我们需要统计GT中的物体总数
            # 如果 metadata 中没有 entities，我们尝试从 answer 标签解析
            gt_entities_for_fn = metadata.get("entities", [])
            if not gt_entities_for_fn:
                try:
                    parsed_answer = ast.literal_eval(gt_tags['answer'])
                    if isinstance(parsed_answer, list):
                        gt_entities_for_fn = parsed_answer
                except:
                    pass
            false_negative_objects += len(gt_entities_for_fn) if gt_entities_for_fn else 1
            sample_level_ious.append(0.0)
            continue

        # --- Case 4: GT和Pred都认为有物体 ---
        if not image_path or not os.path.exists(image_path) or sam_predictor is None:
            sample_level_ious.append(0.0)
            continue
        
        mask_path = image_path.replace("/RGB/", "/mask/").replace(".jpg", ".png")
        if not os.path.exists(mask_path):
            sample_level_ious.append(0.0)
            continue

        try:
            gt_mask_image = load_gt_mask_image(mask_path)
        except FileNotFoundError as e:
            tqdm.write(str(e))
            sample_level_ious.append(0.0)
            continue

        # *** 核心诊断和修正区域 ***
        masks_idx = metadata.get("masks_idx", [])
        gt_entities = metadata.get("entities", [])
        
        # 如果 "entities" 键找不到，尝试从 "entity" 键获取 (一个常见的拼写错误)
        if not gt_entities and "entity" in metadata:
            gt_entities = metadata.get("entity", [])
            if not isinstance(gt_entities, list): # 确保它是一个列表
                 gt_entities = [gt_entities]

        # 如果 gt_entities 仍然为空，打印诊断信息
        if not gt_entities and not metadata_warning_printed:
            tqdm.write("\n" + "="*80)
            tqdm.write(f"⚠️  诊断警告: 在样本 {i} 中, Ground Truth 物体列表 'gt_entities' 为空。")
            tqdm.write(f"这导致了 Instance mIoU 为 0。请检查你的 JSON 文件中的 'metadata' 结构。")
            tqdm.write(f"当前样本的 metadata: {metadata}")
            tqdm.write(f"请确认 'metadata' 中包含物体名称列表的键名是否为 'entities'。")
            tqdm.write("如果键名不同 (例如 'entity_names', 'objects' 等), 请在代码中修改它。")
            tqdm.write("这个警告只会打印一次。")
            tqdm.write("="*80 + "\n")
            metadata_warning_printed = True # 确保只打印一次，避免刷屏

        if not gt_entities or not masks_idx:
            sample_level_ious.append(0.0)
            continue

        gt_masks = {
            entity: (gt_mask_image == idx).astype(np.uint8)
            for idx, entity in zip(masks_idx, gt_entities)
            if idx > 0
        }

        matches = match_objects_by_name(pred_objects, gt_entities)
        current_sample_object_ious = []
        
        for gt_entity in gt_entities:
            best_iou_for_this_gt = 0.0
            if gt_entity not in gt_masks: continue # 确保该实体有对应的GT掩码

            matched_pred_names = [p_name for p_name, g_name in matches.items() if g_name == gt_entity]

            if matched_pred_names:
                for pred_name in matched_pred_names:
                    bbox = pred_objects[pred_name]
                    pred_masks = sam2_generate_masks([bbox], image_path, sam_predictor)
                    if pred_masks:
                        iou = calculate_mask_iou(pred_masks[0], gt_masks[gt_entity])
                        best_iou_for_this_gt = max(best_iou_for_this_gt, iou)

            current_sample_object_ious.append(best_iou_for_this_gt)
            instance_level_ious.append(best_iou_for_this_gt)

        if current_sample_object_ious:
            sample_avg_iou = np.mean(current_sample_object_ious)
            sample_level_ious.append(sample_avg_iou)
        else:
            sample_level_ious.append(0.0)

    # ==========================================
    #        汇总评分输出
    # ==========================================

    sample_mIoU = np.mean(sample_level_ious) if sample_level_ious else 0.0
    instance_mIoU = np.mean(instance_level_ious) if instance_level_ious else 0.0

    print("\n--- 🎯 最终评估结果 ---\n")
    if answer_preds_cider:
        common_ids = set(answer_gts_cider.keys()).intersection(set(answer_preds_cider.keys()))
        gts_for_cider = {i: answer_gts_cider[i] for i in common_ids}
        preds_for_cider = {i: answer_preds_cider[i] for i in common_ids}
        
        if gts_for_cider and preds_for_cider:
            answer_cider_score, _ = cider_scorer.compute_score(gts_for_cider, preds_for_cider)
            print("【<answer> 内容评估】")
            print(f"  - 平均 CIDEr:  {answer_cider_score:.4f}")
            print(f"  - 平均 METEOR: {np.mean(answer_meteor_scores):.4f}\n")

    if sample_level_ious:
        print("【Grounding 分割评估 (SAM2)】")
        print(f"  - 样本平均 IoU (Sample mIoU): {sample_mIoU:.4f}  (衡量每个样本的平均表现)")
        print(f"  - 实例平均 IoU (Instance mIoU): {instance_mIoU:.4f} (仅衡量匹配物体的分割准确度)")
        print(f"\n  --- 计数统计 ---")
        print(f"  - 完美无物体匹配: {perfect_no_object_matches}")
        print(f"  - 误报样本 (GT无, Pred有): {false_positive_samples}")
        print(f"  - 漏报物体 (GT有, Pred无): {false_negative_objects}\n")
    
    print("✅ 评估完成。")

if __name__ == "__main__":
    evaluate()