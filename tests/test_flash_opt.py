import fire
import pytest
import torch
import tqdm
import transformers
from ml_swissknife import utils
from torch import nn
from transformers.models.opt import modeling_opt
from transformers.utils import logging

from alpaca_farm import constants
from alpaca_farm.flash_models import flash_opt

logger = logging.get_logger(__name__)


# --- Include standard models to compare activation and help debug ---
class OPTDecoderLayerNF(modeling_opt.OPTDecoderLayer):
    pass


class OPTDecoderNF(modeling_opt.OPTDecoder):
    def __init__(self, config: modeling_opt.OPTConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([OPTDecoderLayerNF(config) for _ in range(config.num_hidden_layers)])
        self.post_init()

    def forward(
        self,
        *args,
        **kwargs,
    ):
        out = super(OPTDecoderNF, self).forward(*args, **kwargs)
        # print(out.past_key_values[0][0][:, :, -1].sum())
        return out


class OPTModelNF(modeling_opt.OPTModel):
    def __init__(self, config: modeling_opt.OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoderNF(config)
        self.post_init()


class OPTForCausalLMNF(modeling_opt.OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModelNF(config)
        self.post_init()


# --- End of reckless repetition ---


@pytest.mark.parametrize("padding_side", ("left", "right"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@torch.inference_mode()
def test_forward(dtype, padding_side):
    # For some reason, the intermediate tests pass (within each Transformer-block assert attention outputs similar).
    # But the final logit test doesn't pass.

    model_name = "facebook/opt-125m"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = padding_side
    tensors = tokenizer(
        ["i have a good ", "this is a very long sentence that is very long and "],
        return_tensors="pt",
        padding=True,
    )
    tensors = {k: v.to(device) for k, v in tensors.items()}
    print(f'input size: {tensors["input_ids"].shape}')

    model1 = flash_opt.OPTForCausalLM.from_pretrained(model_name).to(device).eval()
    model2 = modeling_opt.OPTForCausalLM.from_pretrained(model_name).to(device).eval()

    with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
        out1 = model1(**tensors, output_hidden_states=True)
        out2 = model2(**tensors, output_hidden_states=True)

    # Outputs are only guaranteed to match at non-padding locations. Clear irrelevant values.
    def clear_padded(tensor):
        tensor = tensor.masked_fill(~tensors["attention_mask"][..., None].bool(), 0.0)
        return tensor

    # Error accumulates! The diff for later hidden states is much larger.
    atol = 1e-2 if dtype == torch.float16 else 1e-1
    rtol = 0
    for h1, h2 in utils.zip_(out1.hidden_states, out2.hidden_states):
        h1, h2 = tuple(clear_padded(tensor) for tensor in (h1, h2))
        torch.testing.assert_close(h1, h2, atol=atol, rtol=rtol)


def all_test_forward():  # This function is not called by pytest.
    for dtype in (torch.float16, torch.bfloat16):
        for padding_side in ("left", "right"):
            test_forward(dtype, padding_side)


@torch.inference_mode()
def test_decoding():
    model_name = "facebook/opt-125m"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # Batch decoding requires left pad, because if right pad, you next token logits could be based on the embedding of
    # a pad token, which is wrong (even though the OPT model increments the position id correctly).
    # In general, any decoder-only HF transformer requires left pad for batch decoding.
    tokenizer.padding_side = "left"
    tensors = tokenizer(
        ["i have a good ", "this is a very long sentence that is very long and "],
        return_tensors="pt",
        padding=True,
    )
    tensors = {k: v.to(device) for k, v in tensors.items()}
    print(f'input size: {tensors["input_ids"].shape}')

    model1: transformers.OPTForCausalLM = flash_opt.OPTForCausalLM.from_pretrained(model_name).to(device).eval()
    model2: transformers.OPTForCausalLM = OPTForCausalLMNF.from_pretrained(model_name).to(device).eval()

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
def profile_decoding():
    # For short sequences, the mixed flash/non-flash approach is still slower.
    model_name = "facebook/opt-1.3b"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=constants.DEFAULT_CACHE_DIR)
    tokenizer.padding_side = "left"
    text = [
        "i have a good ",
        "this is a very long sentence that is very long and ",
        "this is a very long sentence ",
        "this is a very",
    ] * 16
    tensors = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
    )
    tensors = {k: v.to(device) for k, v in tensors.items()}
    print(f'input size: {tensors["input_ids"].shape}')

    model1: transformers.OPTForCausalLM = flash_opt.OPTForCausalLM.from_pretrained(
        model_name, cache_dir=constants.DEFAULT_CACHE_DIR
    )
    model2: transformers.OPTForCausalLM = OPTForCausalLMNF.from_pretrained(
        model_name, cache_dir=constants.DEFAULT_CACHE_DIR
    )

    nbatches = 4
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        for model, msg in (
            (model2, "native"),
            (model1, "flash"),
        ):
            torch.cuda.empty_cache()
            model.to(device).eval()
            model.generate(
                inputs=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                max_new_tokens=500,
                do_sample=False,
                num_beams=1,
            )
            torch.cuda.synchronize()

            with utils.Timer(msg):
                for _ in tqdm.tqdm(range(nbatches)):
                    model.generate(
                        inputs=tensors["input_ids"],
                        attention_mask=tensors["attention_mask"],
                        max_new_tokens=500,
                        do_sample=False,
                        num_beams=1,
                    )
                torch.cuda.synchronize()


def main(task, *args, **kwargs):
    globals()[task](*args, **kwargs)


if __name__ == "__main__":
    # Plain python run for hacking.
    # python -m tests.test_flash_opt --task all_test_forward

    # pytest for systematic testing.
    # pytest -xs tests/test_flash_opt.py
    fire.Fire(main)
