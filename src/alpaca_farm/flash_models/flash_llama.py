# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Callable, List, Optional, Tuple, Union

import einops
import torch
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import modeling_llama

from .. import utils
from . import apex_patch, tensor_ops

logger = logging.getLogger(__name__)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_embedding(q, k, cos, sin):
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaAttention(modeling_llama.LlamaAttention):
    def __init__(self, config: modeling_llama.LlamaConfig):
        super().__init__(config=config)

    def forward(  # noqa
        self,
        hidden_states: torch.Tensor,  # (total_nnz, hidden_size).
        seqlens: torch.Tensor,  # (bsz,).
        cu_seqlens: torch.Tensor,  # (bsz+1,).
        rotary_tensors: tuple[torch.Tensor, torch.Tensor],
        # position_ids is only used for non-flash version, when past_key_value is not None. For flash version,
        # rotary_tensors already takes positions into account.
        position_ids: Optional[torch.Tensor] = None,
        # Crucial loop invariant: We assume past_key_value (input/output) is always in padded format.
        # More precisely, each tensor is of size (bsz, num_heads, seqlen, head_dim).
        # Otherwise we can't extend it with the current key/value embedding through torch.cat easily.
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask_k: Optional[torch.Tensor] = None,
        pad_back: Optional[Callable] = None,
    ):
        if past_key_value is None:
            # (total_nnz, hidden_size) -> (total_nnz, num_heads, head_dim).
            query_states, key_states, value_states = [
                einops.rearrange(func(hidden_states), "t (h d) -> t h d", h=self.num_heads)
                for func in (self.q_proj, self.k_proj, self.v_proj)
            ]
            query_states, key_states = apply_rotary_embedding(query_states, key_states, *rotary_tensors)
            qkv = torch.stack([query_states, key_states, value_states], dim=1)

            assert qkv.dtype in (
                torch.float16,
                torch.bfloat16,
            ), f"Flash attention expected mixed precision. But found qkv dtype: {qkv.dtype}"
            attn_output = flash_attn_unpadded_qkvpacked_func(
                qkv=qkv,
                cu_seqlens=cu_seqlens,
                max_seqlen=seqlens.max(),
                dropout_p=0.0,
                causal=True,
                softmax_scale=self.head_dim**-0.5,
            )
            attn_output = einops.rearrange(attn_output, "t h d -> t (h d)")
            attn_output = self.o_proj(attn_output)

            if use_cache:
                key_states, value_states = tuple(
                    einops.rearrange(pad_back(tensor), "b s h d -> b h s d") for tensor in (key_states, value_states)
                )
                past_key_value = (key_states, value_states)
            return attn_output, None, past_key_value
        else:
            return super(LlamaAttention, self).forward(  # noqa
                hidden_states=hidden_states,
                attention_mask=attention_mask_k,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )


class LlamaDecoderLayer(modeling_llama.LlamaDecoderLayer):
    def __init__(self, config: modeling_llama.LlamaConfig):
        super().__init__(config=config)
        del self.self_attn
        self.self_attn = LlamaAttention(config=config)

    def forward(  # noqa
        self,
        hidden_states: torch.Tensor,
        seqlens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_tensors: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask_k: Optional[torch.Tensor] = None,
        pad_back: Optional[Callable] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(  # noqa
            hidden_states=hidden_states,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            rotary_tensors=rotary_tensors,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask_k=attention_mask_k,
            pad_back=pad_back,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = apex_patch.apex_rmsnorm(self.post_attention_layernorm, hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (present_key_value,)
        return outputs


class LlamaModel(modeling_llama.LlamaModel):
    def __init__(self, config: modeling_llama.LlamaConfig):
        super().__init__(config=config)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self._cache_rotary_embeddings()

    def _cache_rotary_embeddings(self, max_position_embeddings=2048, base=10000):
        dim = self.config.hidden_size // self.config.num_attention_heads

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)  # (seqlen, head_dim).
        self.register_buffer("sin_cached", emb.sin(), persistent=False)  # (seqlen, head_dim).

    def _make_rotary_tensors(self, position_ids: torch.Tensor):
        # position_ids only affects the cos and sin applied to the query and key embeddings.
        # flash path: position_ids size = (total_nnz,); cos sin size = (total_nnz, 1, head_dim)
        # nonflash path: we don't create rotary tensors here, and rely on the builtin RotaryEmbedding.
        #   this assumes position_ids size = (bsz, seqlen).
        assert position_ids.dim() == 1
        # (total_nnz, 1, head_dim)
        cos, sin = [tensor[position_ids].unsqueeze(1) for tensor in (self.cos_cached, self.sin_cached)]
        return cos, sin

    def forward(  # noqa
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        assert not output_attentions
        assert inputs_embeds is None

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embed_tokens(input_ids)

        execute_flash = past_key_values is None

        if execute_flash:
            if position_ids is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                is_selected = attention_mask == 1
                position_ids = torch.cat([t[i] for t, i in utils.zip_(position_ids, is_selected)])
            rotary_tensors = self._make_rotary_tensors(position_ids)
            hidden_states, pad_back, cu_seqlens_q, max_seqlen_q = tensor_ops.unpad_input(hidden_states, attention_mask)
            attention_mask_k = None
        else:
            if position_ids is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids[:, -1].unsqueeze(-1)
            rotary_tensors = None
            hidden_states, pad_back, cu_seqlens_q, max_seqlen_q = hidden_states, lambda x: x, None, None
            # Broadcast assumes query_len == 1.
            attention_mask_k = torch.zeros(
                size=attention_mask.size(), dtype=hidden_states.dtype, device=hidden_states.device
            ).masked_fill(~attention_mask.bool(), torch.tensor(torch.finfo(hidden_states.dtype).min))[:, None, None, :]

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (pad_back(hidden_states),)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                seqlens=attention_mask.sum(dim=1),
                cu_seqlens=cu_seqlens_q,
                rotary_tensors=rotary_tensors,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask_k=attention_mask_k,
                pad_back=pad_back,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        hidden_states = apex_patch.apex_rmsnorm(self.norm, hidden_states)
        hidden_states = pad_back(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_dict:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
            )
        return tuple(v for v in (hidden_states, next_cache, all_hidden_states) if v is not None)


class LlamaForCausalLM(modeling_llama.LlamaForCausalLM):
    def __init__(self, config: modeling_llama.LlamaConfig):
        super().__init__(config)
        self.model = LlamaModel(config)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            if past_key_values is None:  # flash path
                position_ids = attention_mask.long().cumsum(-1) - 1
                is_selected = attention_mask == 1
                position_ids = torch.cat(
                    [
                        this_position_ids[this_is_selected]
                        for this_position_ids, this_is_selected in utils.zip_(position_ids, is_selected)
                    ]
                )
            else:  # non-flash path
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
