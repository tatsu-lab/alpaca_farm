import copy
from typing import Optional

import fire
import torch
import transformers
from flash_attn.bert_padding import unpad_input
from torch import nn
from transformers.models.llama import modeling_llama

from alpaca_farm import utils
from alpaca_farm.flash_models import apex_patch, flash_llama


class LLaMADecoderLayerNF(modeling_llama.LlamaDecoderLayer):
    def __init__(self, config: modeling_llama.LlamaConfig):
        super().__init__(config)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class LLaMAModelNF(transformers.LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([LLaMADecoderLayerNF(config) for _ in range(config.num_hidden_layers)])

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        print(outputs.past_key_values[0][0].sum())
        return outputs


class LLaMAForCausalLMNF(transformers.LlamaForCausalLM):
    def __init__(self, config: modeling_llama.LlamaConfig):
        super().__init__(config)
        self.model = LLaMAModelNF(config)


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
        ).to(inputs_embeds.device)

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


@torch.inference_mode()
def test_llama_attention(dtype=torch.float16):
    # Test flash and standard attention produce comparable results.
    # Right pad only.
    device = torch.device("cuda")

    batch_size, original_seqlen, num_heads, head_dim = 4, 13, 8, 32
    hidden_size = num_heads * head_dim

    seqlens = torch.randint(low=1, high=original_seqlen, size=(batch_size,), device=device)
    attention_mask = torch.arange(original_seqlen, device=device)[None, :] < seqlens[:, None]

    # TODO(lxuechen): Test with past_key_values_length.
    position_ids = attention_mask.long().cumsum(-1) - 1
    is_selected = attention_mask == 1
    flash_position_ids = torch.cat(
        [
            this_position_ids[this_is_selected]
            for this_position_ids, this_is_selected in utils.zip_(position_ids, is_selected)
        ]
    )
    nonflash_position_ids = position_ids.masked_fill_(attention_mask == 0, 1)

    hidden_states = torch.randn(batch_size, original_seqlen, hidden_size, device=device, dtype=dtype)

    hidden_states_unpad, indices, cu_seqlens, max_s = unpad_input(hidden_states, attention_mask)
    expanded_attention_mask = _prepare_decoder_attention_mask(
        attention_mask, (batch_size, original_seqlen), hidden_states, 0
    )

    config = modeling_llama.LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=1,
        num_attention_heads=num_heads,
    )
    block = flash_llama.LlamaAttention(config=config).to(device)

    # Create a small dummy model just for creating rotary tensors.
    dummy_model = flash_llama.LlamaModel(config).to(device)
    rotary_tensors = dummy_model._make_rotary_tensors(flash_position_ids)

    with torch.cuda.amp.autocast(dtype=dtype):
        out1, _, _ = block.forward(
            hidden_states=hidden_states_unpad,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            rotary_tensors=rotary_tensors,
        )

        out2, _, _ = super(flash_llama.LlamaAttention, block).forward(
            hidden_states=hidden_states,
            attention_mask=expanded_attention_mask,
            position_ids=nonflash_position_ids,
        )
        out2, _, _, _ = unpad_input(out2, attention_mask)

    torch.testing.assert_close(out1, out2, atol=1e-3, rtol=0.0)
    print(".")


@torch.inference_mode()
def test_decoding():
    # model_name = "/juice5/scr5/nlp/crfm/human-feedback/models/selfinstruct/llama-teeny"
    model_name = "/self/nlp/scr-sync/nlp/crfm/human-feedback/models/selfinstruct/sft_v5_llama_7b_regen_v7_3ep/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # Batch decoding requires left pad, because if right pad, you next token logits could be based on the embedding of
    # a pad token, which is wrong (even though the OPT model increments the position id correctly).
    # In general, any decoder-only HF transformer requires left pad for batch decoding.
    tokenizer.padding_side = "left"
    clone_tokenizer = copy.deepcopy(tokenizer)

    model1 = flash_llama.LlamaForCausalLM.from_pretrained(
        model_name, device_map={"": device}, low_cpu_mem_usage=True
    ).eval()
    model2 = transformers.LlamaForCausalLM.from_pretrained(
        model_name, device_map={"": device}, low_cpu_mem_usage=True
    ).eval()

    if tokenizer.pad_token is None:
        utils.stable_resize_token_embeddings_and_tokenizer(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model1,
        )
        utils.stable_resize_token_embeddings_and_tokenizer(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=clone_tokenizer,
            model=model2,
        )

    tensors = tokenizer(
        ["i have a good ", "this is a very long sentence that is very long and "],
        return_tensors="pt",
        padding=True,
    )
    tensors = {k: v.to(device) for k, v in tensors.items()}
    print(f'input size: {tensors["input_ids"].shape}')

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        # greedy
        out1 = model1.generate(
            inputs=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            max_new_tokens=100,
            do_sample=False,
            num_beams=1,
        )
        text = tokenizer.batch_decode(out1, skip_special_tokens=True)
        print(text)

        out2 = model2.generate(
            inputs=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            max_new_tokens=100,
            do_sample=False,
            num_beams=1,
        )
        text = tokenizer.batch_decode(out2, skip_special_tokens=True)
        print(text)
        print(torch.ne(out1, out2))
        print(out1 - out2)
        assert torch.eq(out1, out2).all().item()

        # temperature
        out = model1.generate(
            inputs=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=3,
        )
        text = tokenizer.batch_decode(out, skip_special_tokens=True)
        print(text)

        out = model2.generate(
            inputs=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=3,
        )
        text = tokenizer.batch_decode(out, skip_special_tokens=True)
        print(text)


@torch.inference_mode()
def test_forward(dtype=torch.bfloat16, padding_side="left"):
    model_name = "/self/nlp/scr-sync/nlp/huggingface_hub_llms/llama-7b/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = padding_side
    clone_tokenizer = copy.deepcopy(tokenizer)

    model1 = flash_llama.LlamaForCausalLM.from_pretrained(
        model_name, device_map={"": device}, low_cpu_mem_usage=True
    ).eval()
    model2 = transformers.LlamaForCausalLM.from_pretrained(
        model_name, device_map={"": device}, low_cpu_mem_usage=True
    ).eval()

    if tokenizer.pad_token is None:
        utils.smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model1,
        )
        utils.smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=clone_tokenizer,
            model=model2,
        )

    tensors = tokenizer(
        ["i have a good ", "this is a very long sentence that is very long and ", "what type of food do you like?"],
        return_tensors="pt",
        padding=True,
    )
    tensors = {k: v.to(device) for k, v in tensors.items()}

    with torch.cuda.amp.autocast(dtype=dtype):
        out1 = model1(**tensors, output_hidden_states=True, return_dict=True)
        out2 = model2(**tensors, output_hidden_states=True, return_dict=True)

    def clear_padded(tensor):
        tensor.masked_fill_(~tensors["attention_mask"][..., None].bool(), 0.0)
        # tensor[:2, ...] = 0.
        return tensor

    # Error accumulates! The diff for later hidden states is much larger.
    atol = 1e-2 if dtype == torch.float16 else 1e-1
    rtol = 0
    for layer_idx, (h1, h2) in enumerate(utils.zip_(out1.hidden_states, out2.hidden_states)):
        h1, h2 = tuple(clear_padded(tensor) for tensor in (h1, h2))
        if not torch.allclose(h1, h2, atol=atol, rtol=rtol):
            print(
                f"found large error for hidden states at layer: {layer_idx}. "
                f"maximum diff: {(h1 - h2).abs().max().item()}. "
                f"num entries with large diff: {((h1 - h2).abs() > 3).sum()}. "
                f"norm of diff: {(h1 - h2).norm().item()}. "
            )


def all_test_forward():  # This function is not called by pytest.
    for dtype in (torch.float16, torch.bfloat16):
        for padding_side in ("left", "right"):
            test_forward(dtype, padding_side)


def test_fused_rms_norm():
    device = torch.device("cuda")
    norm = transformers.models.llama.modeling_llama.LlamaRMSNorm(256).to(device=device)
    x = torch.randn(16, 128, 256, device=device)

    y1 = norm(x)
    y2 = apex_patch.apex_rmsnorm(norm, x)
    torch.testing.assert_close(y2, y1)


def main(task, **kwargs):
    # python -m models.flash_llama test_llama_attention
    # CUDA_VISIBLE_DEVICES=0 python -m tests.test_flash_llama test_llama_attention
    # CUDA_VISIBLE_DEVICES=0 python -m tests.test_flash_llama test_decoding
    # CUDA_VISIBLE_DEVICES=0 python -m tests.test_flash_llama test_forward
    # CUDA_VISIBLE_DEVICES=0 python -m tests.test_flash_llama test_fused_rms_norm
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
