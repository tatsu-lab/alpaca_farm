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

from typing import Callable, List, Optional, Tuple, Union

import einops
import torch
import transformers
from flash_attn.flash_attn_interface import flash_attn_unpadded_func
from torch import nn
from transformers.models.opt import modeling_opt
from transformers.utils import logging

from . import apex_patch, tensor_ops

logger = logging.get_logger(__name__)


class OPTDecoderLayer(modeling_opt.OPTDecoderLayer):
    def forward(  # noqa
        self,
        # (bsz x seqlen, hidden_size) or (bsz, 1, hidden_size) if past_key_value is not None.
        hidden_states: torch.Tensor,
        pad_back: Callable,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        # Crucial loop invariant: We assume past_key_value (input/output) is always in padded format.
        # More precisely, each tensor is of size (bsz, seqlen, hidden_size).
        # Otherwise we can't extend it with the current key/value embedding through torch.cat easily.
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask_k: Optional[torch.Tensor] = None,  # (bsz, seqlen+1,).
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = apex_patch.apex_layernorm(self.self_attn_layer_norm, hidden_states)
        query = self.self_attn.q_proj(hidden_states)
        key = self.self_attn.k_proj(hidden_states)
        value = self.self_attn.v_proj(hidden_states)

        num_heads, head_dim = self.self_attn.num_heads, self.self_attn.head_dim
        if past_key_value is None:  # hidden_states should be in unpadded format to run flash-attn.
            query, key, value = tuple(
                einops.rearrange(tensor, "nnz (h d) -> nnz h d", h=num_heads, d=head_dim)
                for tensor in (query, key, value)
            )
            hidden_states = flash_attn_unpadded_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_q,
                dropout_p=(self.self_attn.dropout if self.training else 0.0),
                causal=True,
                softmax_scale=self.self_attn.scaling,
            )
            hidden_states = einops.rearrange(hidden_states, "nnz h d -> nnz (h d)")
        else:  # hidden_states should be in padded format.
            query = query * self.self_attn.scaling
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)

            query_states = einops.rearrange(query, "b s (h d) -> (b h) s d", h=num_heads, d=head_dim)
            key_states = einops.rearrange(key, "b l (h d) -> (b h) l d", h=num_heads, d=head_dim)
            value_states = einops.rearrange(value, "b l (h d) -> (b h) l d", h=num_heads, d=head_dim)

            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
            attn_weights = (
                # attention_mask_k broadcast correctness assumes query_len == 1.
                einops.rearrange(attn_weights, "(b h) s l -> b h s l", h=num_heads)
                + attention_mask_k[:, None, None, :]
            )
            attn_weights = einops.rearrange(attn_weights, "b h s l -> (b h) s l")
            if attn_weights.dtype == torch.float16:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
            else:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
            hidden_states = torch.bmm(attn_probs, value_states)
            hidden_states = einops.rearrange(hidden_states, "(b h) s d -> b s (h d)", h=num_heads, d=head_dim)

            # Below requires pytorch 2.0. Installing pytorch 2.0 however may break other packages.
            # Only migrate when things become more stable.
            # hidden_states = F.scaled_dot_product_attention(
            #     query=query,
            #     key=key,
            #     value=value,
            #     attn_mask=attention_mask_k[:, None, None, :].bool(),  # This assumes query_len == 1.
            #     dropout_p=(self.self_attn.dropout if self.training else 0.0),
            #     causal=False,
            # )
            # hidden_states = einops.rearrange(hidden_states, "b h s d -> b s (h d)")

        hidden_states = self.self_attn.out_proj(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = apex_patch.apex_layernorm(self.final_layer_norm, hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if use_cache:
            if past_key_value is None:
                key, value = tuple(
                    einops.rearrange(pad_back(tensor), "b s h d -> b s (h d)", h=num_heads, d=head_dim)
                    for tensor in (key, value)
                )
            present_key_value = (key, value)  # (bsz, seqlen+1, hidden_size).
            outputs += (present_key_value,)

        return outputs


class OPTDecoder(modeling_opt.OPTDecoder):
    def __init__(self, config: modeling_opt.OPTConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, transformers.models.opt.modeling_opt.BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # This simplified fast implementation only supports a subset of configurations.
        # We also ignore use_cache, but we don't assert that because it's True at training time
        # (even though it's not actually used) and I don't know how to set it to False at training time only.
        # We can add support for specific configurations as needed.
        assert attention_mask is not None
        assert output_attentions is False
        assert head_mask is None
        assert self.gradient_checkpointing is False
        assert inputs_embeds is None
        assert self.final_layer_norm is not None
        assert self.project_in is None
        assert self.project_out is None
        assert self.layerdrop == 0
        for layer in self.layers:
            assert layer.do_layer_norm_before is True

        # past_key_values is a list of tuples (key, value). key/value each of size (bsz, seqlen, hidden_size).
        past_key_values_length = past_key_values[0][0].shape[1] if past_key_values is not None else 0

        # Embed inputs and positions
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        inputs_embeds = self.embed_tokens(input_ids)

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)
        assert (
            inputs_embeds.size() == pos_embeds.size()
        ), "Internal error: inputs_embeds and pos_embeds not of same shape."
        hidden_states = inputs_embeds + pos_embeds

        if past_key_values_length == 0:
            # Unpad hidden states: (bsz, seqlen, hidden_size) -> (total_nnz, hidden_size)
            hidden_states, pad_back, cu_seqlens_q, max_seqlen_q = tensor_ops.unpad_input(hidden_states, attention_mask)
            attention_mask_k = None
        else:
            hidden_states, pad_back, cu_seqlens_q, max_seqlen_q = hidden_states, lambda x: x, None, None
            attention_mask_k = torch.zeros(
                size=attention_mask.size(), dtype=inputs_embeds.dtype, device=inputs_embeds.device
            ).masked_fill(~attention_mask.bool(), torch.tensor(torch.finfo(inputs_embeds.dtype).min))

        next_decoder_cache = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (pad_back(hidden_states),)
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = layer(
                hidden_states=hidden_states,
                pad_back=pad_back,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                past_key_value=past_key_value,
                attention_mask_k=attention_mask_k,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
        hidden_states = apex_patch.apex_layernorm(self.final_layer_norm, hidden_states)

        hidden_states = pad_back(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_dict:
            return transformers.models.opt.modeling_opt.BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
            )
        return tuple(v for v in (hidden_states, next_cache, all_hidden_states) if v is not None)


class OPTModel(modeling_opt.OPTModel):
    def __init__(self, config: modeling_opt.OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder(config)
        self.post_init()


class OPTForCausalLM(modeling_opt.OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)
        self.post_init()
