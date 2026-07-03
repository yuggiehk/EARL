
# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple, Optional
import os
from transformers.utils import logging
logger = logging.get_logger(__name__)

def qwen2_5vl_vision_flash_attn_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output


def monkey_patch_qwen2_5vl_flash_attn():
    Qwen2_5_VLVisionFlashAttention2.forward = qwen2_5vl_vision_flash_attn_forward


# [DEPRECATED] Lines 48-265 of this file previously contained an earlier
# (commented-out) version of qwen2_5vl_forward. Removed for clarity.
# See git history or deprecated/ folder for the original.


# ----------------------- Fix the process pending bug when using data mixture of image-text data and pure-text under deepseed zero3-----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from typing import List, Union
from torch.nn import CrossEntropyLoss
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

def qwen2_5vl_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        goal_caption_embedding: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            
            # 简化的图像处理逻辑 (适配 Zero2)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                
                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)


            # 处理视频 (如果需要)
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # --- 残差注入 Goal Caption Embedding（不改变序列长度；对文本位注入） ---
            if goal_caption_embedding is not None and torch.any(goal_caption_embedding) and hasattr(self, 'goal_projector'):
                device = inputs_embeds.device
                dtype = inputs_embeds.dtype

                # 仅对文本位做池化，得到每个样本的文本全局向量 pooled: [B, H]
                text_mask = (input_ids != self.config.image_token_id)  # [B, L]
                denom = text_mask.sum(dim=1, keepdim=True).clamp_min(1)
                pooled = (inputs_embeds * text_mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, H]
                pooled = pooled.to(dtype)

                # 对齐 caption 向量到 projector 权重的 dtype，且对齐到 inputs_embeds 的 device
                proj_dtype = next(self.goal_projector.parameters()).dtype
                goal_caption_embedding = goal_caption_embedding.to(device=device, dtype=proj_dtype)

                # FusionModule: 返回 pooled + 0.001 * g(caption)
                fused = self.goal_projector(goal_caption_embedding, pooled)  # [B, H]
                if fused.dtype != dtype:
                    fused = fused.to(dtype)

                # 抽出“调制量” m = fused - pooled，并对齐 dtype
                modulation = (fused - pooled).to(dtype)

                # 可选门控系数 alpha（未设置时默认为 1.0）
                alpha = getattr(self, 'goal_gate_alpha', 1.0) if hasattr(self, 'goal_gate_alpha') else 1.0

                # 残差注入到文本位（不改序列长度，不动 attention_mask / labels）
                inputs_embeds = inputs_embeds + alpha * text_mask.unsqueeze(-1) * modulation.unsqueeze(1)

                # 统一整个 inputs_embeds dtype 为模型 embedding 权重的 dtype（通常是 bfloat16）
                target_dtype = self.model.embed_tokens.weight.dtype
                inputs_embeds = inputs_embeds.to(target_dtype)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # 位置编码处理保持不变
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    None,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 如果注入了 goal embedding，调整 labels
            # if goal_caption_embedding is not None and torch.any(goal_caption_embedding) and hasattr(self, 'goal_projector'):
            #     goal_label_padding = torch.full(
            #         (labels.size(0), 1), -100, dtype=labels.dtype, device=labels.device
            #     )
            #     labels = torch.cat([goal_label_padding, labels], dim=1)

            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        if os.getenv("DEBUG_DTYPE") == "1" and (cache_position is None or (isinstance(cache_position, torch.Tensor) and cache_position.item() == 0)):
            print(
                "[DTYPE DEBUG]",
                "inputs_embeds:", inputs_embeds.dtype,
                "embed_weight:", self.model.embed_tokens.weight.dtype,
                "proj_weight:", self.goal_projector.weight.dtype if hasattr(self, "goal_projector") else None,
            )

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


        
def monkey_patch_qwen2_5vl_forward():
    Qwen2_5_VLForConditionalGeneration.forward = qwen2_5vl_forward

# ----------------------- Set the Weights only as False in torch.load (In Pytorch 2.6, this is default as True)-----------------------
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
from deepspeed.utils import logger, log_dist
def weigths_only_load(self, path: str, map_location=None):
    logger.info(f"[Torch] Loading checkpoint from {path}...")
    partition = torch.load(path, map_location=map_location, weights_only=False)
    logger.info(f"[Torch] Loaded checkpoint from {path}.")
    return partition

def monkey_patch_torch_load():
    TorchCheckpointEngine.load = weigths_only_load



