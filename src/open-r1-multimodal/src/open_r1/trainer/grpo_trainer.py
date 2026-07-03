"""
[DEPRECATED] The first 876 lines of this file contained an earlier
version of VLMGRPOTrainer (the GRPO trainer for vision-language models). They have been removed for clarity.
See the git history or deprecated/ folder for the original code.
"""

import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from accelerate.utils import is_peft_model, set_seed
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from open_r1.vlm_modules.vlm_module import VLMBaseModule
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class VLMGRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.vlm_module = vlm_module

        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass
        elif isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`."
            )
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        model = model_cls.from_pretrained(model_id, **model_init_kwargs)

        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None))
            for component, processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    processing_component = getattr(processing_class, component, processing_class)
                    setattr(processing_component, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer",  None) is not None:
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase"
                pad_token_id = processing_class.pad_token_id
        
        self.vlm_module.post_model_init(model, processing_class)

        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None:
            print("Applying LoRA...")
            # def find_all_linear_names(model, multimodal_keywords):
            #     cls = torch.nn.Linear
            #     lora_module_names = set()
            #     for name, module in model.named_modules():
            #         if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            #             continue
            #         if isinstance(module, cls):
            #             lora_module_names.add(name)
            #     # if hasattr(model, 'goal_projector'):
            #     #     lora_module_names.add("goal_projector")
            #     for m in list(lora_module_names):
            #         if "embed_tokens" in m:
            #             lora_module_names.remove(m)
            #     return list(lora_module_names)

            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                
                for name, module in model.named_modules():
                    # 跳过视觉相关模块
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    # 跳过FusionModule相关参数
                    if "goal_projector" in name:
                        continue
                    # 仅收集叶子 Linear 层
                    if isinstance(module, cls):
                        lora_module_names.add(name)

                # 过滤掉不需要 LoRA 的模块
                filtered_names = {
                    m for m in lora_module_names
                    if not ("embed_tokens" in m)
                }

                return list(filtered_names)
            
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False
                    
        # 确保FusionModule的参数是可训练的
        if hasattr(model, 'goal_projector'):
            for param in model.goal_projector.parameters():
                param.requires_grad = True
            print("FusionModule parameters set to trainable")

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in trainable_params)
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"Trainable parameter: {n}, shape: {p.shape}")
        print(f"Total trainable parameters: {total_params}")

        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        self.beta = args.beta
        if self.beta == 0.0:
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        self.vlm_module.post_model_init(self.ref_model, processing_class)

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        def data_collator(features: list) -> dict[str, list]:
                """
                将一个样本字典的列表 (List[Dict]) 转换为一个列表的字典 (Dict[List])。
                这是 transformers 标准的 collator 行为。
                """
                if not features:
                    return {}
                first = features[0]
                batch = {}
                for k in first.keys():
                    batch[k] = [f[k] for f in features]
                return batch
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1,
            pad_token_id=pad_token_id,
        )
        if hasattr(self.vlm_module, "get_eos_token_id"):
            self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
        self.beta = args.beta
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        self.num_iterations = args.num_iterations
        self._step = 0
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        model.warnings_issued["estimate_tokens"] = True

        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size must be divisible by the number of generations per prompt."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size must be divisible by the number of generations per prompt."
                )

        set_seed(args.seed, device_specific=True)

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if is_deepspeed_zero3_enabled():
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        model.config.use_cache = False
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()
            try:
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                args.gradient_checkpointing = False
            except:
                pass
        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )
        if use_reentrant:
            model.enable_input_require_grads()
        return model
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        # The patched forward now handles everything internally
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits
        
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _prepare_inputs(self, inputs):
        return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        return [e for e in ele] if isinstance(ele, list) else [ele]

    # --- START OF MODIFICATION: This function is now corrected to handle the Dict[List] data structure ---
    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model) -> dict[str, Union[torch.Tensor, Any]]:
        # `inputs` is a Dict[List] from the data collator, e.g., {'prompt': ['p1', 'p2'], 'image_path': ['path1', 'path2']}.
        # The old code incorrectly iterated over `inputs`, treating it as a List[Dict]. We fix that here.
        
        # For compatibility with parts of the code that expect a List[Dict], we can reconstruct it.
        # This is slightly inefficient but makes the logic robust and easier to debug.
        num_samples = len(inputs[next(iter(inputs))])
        inputs_list_of_dicts = [{key: inputs[key][i] for key in inputs} for i in range(num_samples)]

        device = self.accelerator.device

        # CORRECT: Access the list of prompts directly from the dictionary key.
        prompts = inputs["prompt"]
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs) # This function should handle Dict[List]
        
        # CORRECT: Access the list of embeddings directly.
        goal_caption_embeddings = inputs.get('goal_caption_embedding')
        if goal_caption_embeddings is None or any(emb is None for emb in goal_caption_embeddings):
             raise ValueError("Missing 'goal_caption_embedding' in one of the batch items.")
        
        # list_of_tensors = [torch.tensor(emb, dtype=torch.float) for emb in goal_caption_embeddings]
        # goal_caption_embeddings_tensor = torch.stack(list_of_tensors).to(device)

        # Get the target dtype from the model itself to ensure consistency (e.g., torch.bfloat16)
        if hasattr(model, 'module'):
            model_dtype = model.module.dtype
        else:
            model_dtype = model.dtype
        list_of_tensors = [torch.tensor(emb) for emb in goal_caption_embeddings]
        # Stack the tensors and then cast to the correct device and dtype in one go
        goal_caption_embeddings_tensor = torch.stack(list_of_tensors).to(device=device, dtype=model_dtype)

        images = []
        # CORRECT: Check for 'image_path' key and iterate over its list value.
        if "image_path" in inputs and inputs["image_path"] is not None:
            for path in inputs["image_path"]:
                if path and os.path.exists(path) and not os.path.isdir(path):
                    try:
                        img = PIL.Image.open(path).convert("RGB")
                        w, h = img.size
                        if w < 28 or h < 28:
                            if w < h:
                                new_w, new_h = 28, int(h * (28/w))
                            else:
                                new_h, new_w = 28, int(w * (28/h))
                            img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                        images.append(img)
                    except Exception as e:
                        warnings.warn(f"Could not open or resize image at {path}. Error: {e}. Skipping.")
                        images.append(None) # Append None if image is corrupted
                else:
                    images.append(None) # Append None if path is invalid or None
        elif "image" in inputs and inputs["image"] is not None:
             images = inputs["image"] # Assumes images are already PIL objects
        
        # Filter out None images and corresponding data to prevent crashes.
        valid_indices = [i for i, img in enumerate(images) if img is not None]
        if len(valid_indices) < len(images):
            num_skipped = len(images) - len(valid_indices)
            warnings.warn(f"{num_skipped} sample(s) were skipped due to missing or invalid images.")
            
            # If all images in the batch are invalid, we cannot proceed.
            if not valid_indices:
                # Returning an empty dict or raising an error are options.
                # For training, it's better to skip the batch. We can return a dict with a special key.
                # However, the trainer loop expects a loss tensor. A simpler approach for now is to raise.
                raise ValueError("No valid images found in the batch. Cannot proceed with generation.")

            images = [images[i] for i in valid_indices]
            prompts_text = [prompts_text[i] for i in valid_indices]
            goal_caption_embeddings_tensor = goal_caption_embeddings_tensor[valid_indices]
            inputs_list_of_dicts = [inputs_list_of_dicts[i] for i in valid_indices]
            prompts = [prompts[i] for i in valid_indices]

        prompt_inputs, additional_output = self.vlm_module.prepare_model_inputs(
            self.processing_class, prompts_text, images, return_tensors="pt", 
            padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs['goal_caption_embedding'] = goal_caption_embeddings_tensor
        
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if additional_output is not None:
            for i, (input_i, additional_output_i) in enumerate(zip(inputs_list_of_dicts, additional_output)):
                input_i.update(additional_output_i)

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            generate_kwargs = {k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()}
            generate_returned_result = unwrapped_model.generate(**generate_kwargs, generation_config=self.generation_config)
            
            prompt_length = prompt_ids.size(1)
            
            if not self.vlm_module.is_embeds_input():
                prompt_completion_ids = generate_returned_result
                completion_ids = prompt_completion_ids[:, prompt_length:]
            else:
                completion_ids = generate_returned_result
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] for k in multimodal_keywords if k in prompt_inputs}
        
        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, **multimodal_inputs)
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta > 0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, **multimodal_inputs)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
            else:
                ref_per_token_logps = None

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # CORRECT: Use the reconstructed list_of_dicts for this check.
        if is_conversational(inputs_list_of_dicts[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_func, PreTrainedModel):
                texts = [p + c[0]['content'] for p, c in zip(prompts, completions)] if is_conversational(inputs_list_of_dicts[0]) else [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                # CORRECT: Use the reconstructed list_of_dicts to build reward_kwargs correctly.
                reward_kwargs = {key: [ex[key] for ex in inputs_list_of_dicts] for key in inputs_list_of_dicts[0] if key not in ["prompt", "completion"]}
                rewards_per_func[:, i] = torch.tensor(reward_func(prompts=prompts, completions=completions, **reward_kwargs), dtype=torch.float32, device=device)

        rewards_per_func = self.accelerator.gather(rewards_per_func)
        rewards = rewards_per_func.sum(dim=1)
        
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1).repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1).repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        process_slice = slice(self.accelerator.process_index * len(prompts), (self.accelerator.process_index + 1) * len(prompts))
        advantages = advantages[process_slice]

        self._metrics["completion_length"].append(self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item())
        reward_per_func_mean = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            name = reward_func.config._name_or_path.split("/")[-1] if isinstance(reward_func, PreTrainedModel) else reward_func.__name__
            self._metrics[f"rewards/{name}"].append(reward_per_func_mean[i].item())
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        return {
            "prompt_ids": prompt_ids, "prompt_mask": prompt_mask,
            "completion_ids": completion_ids, "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps, "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages, "multimodal_inputs": multimodal_inputs
        }
    # --- END OF MODIFICATION ---

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]
        
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, **multimodal_inputs)
        per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]

        advantages = inputs["advantages"]
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss = -torch.min(coef_1 * advantages.unsqueeze(1), coef_2 * advantages.unsqueeze(1))

        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss += self.beta * per_token_kl
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        clip_ratio = ((coef_1 < coef_2).float() * completion_mask).sum() / completion_mask.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs.update(metrics)
        super().log(logs, start_time) if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0") else super().log(logs)
        self._metrics.clear()

    def create_model_card(self, model_name: Optional[str] = None, dataset_name: Optional[str] = None, tags: Union[str, list[str], None] = None):
        if not self.is_world_process_zero(): return
        base_model = self.model.config._name_or_path if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path) else None
        tags = [tags] if isinstance(tags, str) else (tags or [])
        if hasattr(self.model.config, "unsloth_version"): tags.append("unsloth")
        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:240},
            }"""
        )
        model_card = generate_model_card(
            base_model=base_model, model_name=model_name, hub_model_id=self.hub_model_id, dataset_name=dataset_name,
            tags=tags, wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(), trainer_name="GRPO", trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )
        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self) -> Sampler:
        effective_batch_size = self.args.per_device_train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        return RepeatRandomSampler(
            data_source=self.train_dataset, mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations, seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatRandomSampler(
            data_source=eval_dataset, mini_repeat_count=self.num_generations, seed=self.args.seed
        )
