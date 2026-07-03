from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
import torch.nn as nn
from copy import deepcopy
from open_r1.vlm_modules.vlm_module import VLMBaseModule
from PIL import Image
# 在文件顶部添加必要的导入
import math
import torch.nn.functional as F


def truncated_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        tensor.copy_(torch.normal(mean, std, size=tensor.size()))
        while True:
            mask = (tensor < a) | (tensor > b)
            if not mask.any():
                break
            tensor[mask] = torch.normal(mean, std, size=tensor[mask].size())


class FusionModule(nn.Module):
    def __init__(self, channels=1, hidden_dim=1024, input_dim=2048, output_dim=3584):
        super(FusionModule, self).__init__()
        
        self.channels = channels
        # self.input_dim = channels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.layer_norm_fc = nn.LayerNorm(self.hidden_dim)
      
        self.conv_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.att_layer_norm = nn.LayerNorm(hidden_dim)

        self.linear_transform = torch.nn.Linear(hidden_dim, output_dim)
        # self.output_layer_norm = nn.LayerNorm(output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        truncated_normal_(self.fc1.weight, mean=0.0, std=0.02)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)

        for conv in [self.conv_q, self.conv_k, self.conv_v]:
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)

        truncated_normal_(self.linear_transform.weight, mean=0.0, std=0.02)
        if self.linear_transform.bias is not None:
            nn.init.constant_(self.linear_transform.bias, 0)

    def forward(self, x, input_embedding):
        # x: caption features [batch_size, 2048]
        # input_embedding: Qwen的input_embedding [batch_size, 3584]

        x = x.to(input_embedding.device, dtype=input_embedding.dtype)
        x = self.layer_norm_fc(self.fc1(x))
        b, dim = x.shape 
        h = w = int(math.sqrt(dim))
        x = x.reshape(b, self.channels, h, w)
        
        b, c, h, w = x.shape
        
        q = self.conv_q(x)  # (b, c, h, w)
        k = self.conv_k(x)  # (b, c, h, w)
        v = self.conv_v(x)  # (b, c, h, w)
      
        q = q.view(b, c, -1)  # (b, c, h*w)
        k = k.view(b, c, -1)  # (b, c, h*w)
        v = v.view(b, c, -1)  # (b, c, h*w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn_weights = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)  # (b, h*w, h*w)
        out = torch.bmm(attn_weights, v.transpose(1, 2))  # (b, h*w, c)
        out = self.att_layer_norm(out.view(b, -1))
        
        out = self.linear_transform(out)
        # out = self.output_layer_norm(out) 
        out = out.to(input_embedding.dtype)
        # weighted sum with original input embedding
        out = out + input_embedding

        return out

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    # [DEPRECATED] Old post_model_init used a simple nn.Linear projector.
    # Replaced by FusionModule-based implementation below.

    def post_model_init(self, model, processing_class):
        """
        使用 FusionModule 替换原来的 MLP projector，并与模型 dtype/device 对齐
        """
        if model is None:
            return

        caption_embedding_dim = 2048
        llm_hidden_size = model.config.hidden_size

        if not hasattr(model, 'goal_projector'):
            model.goal_projector = FusionModule(
                channels=1,
                hidden_dim=1024,
                input_dim=caption_embedding_dim,
                output_dim=llm_hidden_size,
            )
            model_dtype = next(model.parameters()).dtype
            model_device = next(model.parameters()).device
            model.goal_projector.to(model_device, dtype=model_dtype)

            # if not hasattr(model, 'goal_gate_alpha'):
            #     model.goal_gate_alpha = 0.1
            print(f"Attached FusionModule projector ({caption_embedding_dim} -> {llm_hidden_size}) to the model.")
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        # return ['pixel_values', 'image_grid_thw']
        return ['pixel_values', 'image_grid_thw', 'goal_caption_embedding']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    # [DEPRECATED] Old prepare_prompt used List[Dict] format. 
    # Replaced by Dict[List] compatible version below.

    def prepare_prompt(self, processing_class, inputs: dict[str, list]):
        """
        Prepares the prompt text from the collated batch dictionary.
        This function now correctly handles the Dict[List] format from the data collator.
        """
        # The 'inputs' is a dictionary of lists, e.g., {'prompt': [p1, p2], 'image_path': [ip1, ip2]}.
        # We need to reconstruct the list of individual sample dictionaries for maybe_apply_chat_template.
        
        # Find a key that exists to determine the batch size. 'prompt' is a good candidate.
        if "prompt" not in inputs or not inputs["prompt"]:
            return []
        
        batch_size = len(inputs["prompt"])
        keys = inputs.keys()
        
        # Reconstruct the list of dictionaries (List[Dict]) from the dictionary of lists (Dict[List])
        examples_list = []
        for i in range(batch_size):
            # For each sample in the batch, create a dictionary
            example_dict = {key: inputs[key][i] for key in keys}
            examples_list.append(example_dict)

        # Now, iterate over the reconstructed list of examples, which is the format expected by TRL's function.
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in examples_list]
        
        return prompts_text

    # def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
    #     # FIXME
    #     # This could only process pure-multimodal or pure-text inputs
    #     additional_output = None
    #     if len(images) > 0:
    #         prompt_inputs = processing_class(
    #             text=prompts_text,
    #             images=images,
    #             return_tensors=return_tensors,
    #             padding=padding,
    #             padding_side=padding_side,
    #             add_special_tokens=add_special_tokens)
    #         additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]
    #     else:
    #         prompt_inputs = processing_class(
    #             text=prompts_text,
    #             return_tensors=return_tensors,
    #             padding=padding,
    #             padding_side=padding_side,
    #             add_special_tokens=add_special_tokens)
    #     return prompt_inputs, additional_output
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        additional_output = None
        
        # 新增：加载图像路径为 PIL.Image 对象
        loaded_images = []
        for img in images:  # 假设 images 是列表，可能嵌套
            if isinstance(img, str):  # 如果是单个路径字符串
                loaded_images.append(Image.open(img))  # 使用 PIL 加载图像
            elif isinstance(img, list):  # 如果是路径列表
                loaded_images.extend([Image.open(p) for p in img])
            else:
                loaded_images.append(img)  # 如果已经是图像对象，直接添加
        
        if len(loaded_images) > 0:  # 修改为使用 loaded_images
            prompt_inputs = processing_class(
                text=prompts_text,
                images=loaded_images,  # 使用加载后的图像
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
            additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs, additional_output
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case "ego_irg":
                SYSTEM_PROMPT = (
                    "You are a multimodal assistant trained for referring image grounding. "
                    "Follow the explicit output format `<answer>` and `<bbox>` as given in the dataset instructions. "
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        import json
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        def resize_bbox(bbox, input_height, input_width, image_height, image_width):
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
            bbox[2] = bbox[2] / input_width * image_width
            bbox[3] = bbox[3] / input_height * image_height
            return bbox
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

        for i, (content, sol) in enumerate(zip(contents, solution)):
            image_grid_thw = kwargs.get("image_grid_thw")[i]
            image_path = kwargs.get("image_path")[i][0]
            image = Image.open(image_path)
            image_width, image_height = image.size
            input_height = int(image_grid_thw[1]*14)
            input_width = int(image_grid_thw[2]*14)
            
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[i]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "rec":
                    return Qwen2VLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")

if __name__ == "__main__":
    # 测试FusionModule
    b, c = 16, 2048
    input_tensor = torch.randn(b, c)  
    ori_input_embedding = torch.randn(b, 3584)
    
    fusion_layer = FusionModule(channels=1, hidden_dim=1024, input_dim=2048, output_dim=3584)  
    output = fusion_layer(input_tensor, ori_input_embedding)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("FusionModule test passed!")