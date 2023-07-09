import argparse
import json
import logging
import os

import numpy as np
import torch
import transformers
from huggingface_hub import HfApi, hf_hub_download

from alpaca_farm.models.reward_model import RewardConfig, RewardModel
from alpaca_farm.utils import stable_resize_token_embeddings_and_tokenizer

min_transformers_version = "4.29.2"


def get_alpaca_farm_model_names():
    api = HfApi()
    models = api.list_models(author="tatsu-lab", search="alpaca-farm")
    model_names = [model.modelId for model in models]
    model_names = [name.replace("tatsu-lab/alpaca-farm-", "").replace("-wdiff", "") for name in model_names]
    return model_names


def build_argparse(model_names):
    parser = argparse.ArgumentParser("Download AlpacaFarm models")
    parser.add_argument("--llama-7b-hf-dir", type=str, required=True)
    parser.add_argument("--alpaca-farm-model-name", choices=model_names + ["all"], default="all", required=True)
    parser.add_argument("--models-save-dir", default="./pretrained_models", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--path-to-sft10k", type=str, help="Necessary for reconstructing reward models.")
    args = parser.parse_args()
    if args.path_to_sft10k is None:
        args.path_to_sft10k = os.path.join(args.models_save_dir, "sft10k")
    return args


def load_weight_diff(hf_hub_name, is_reward_model=False, device="cpu", path_to_sft10k=None):
    if is_reward_model:
        model_tuned = RewardModel.from_pretrained(
            hf_hub_name,
            device_map={"": torch.device(device)},
            torch_dtype=torch.float32,
            flash_attn=False,
            config=RewardConfig(backbone_model_name_or_path=path_to_sft10k),
        )
    else:
        model_tuned = transformers.AutoModelForCausalLM.from_pretrained(
            hf_hub_name, device_map={"": torch.device(device)}, torch_dtype=torch.float32
        )
    tokenizer_tuned = transformers.AutoTokenizer.from_pretrained(hf_hub_name)
    return model_tuned.eval(), tokenizer_tuned


def load_raw_model(model_dir, device="cpu"):
    config_path = os.path.join(model_dir, "config.json")
    config = json.load(open(config_path, "r"))
    transformers_version = config["transformers_version"]
    if transformers_version < min_transformers_version:
        logging.warning(
            f"Your base LLaMA checkpoint is converted with transformers=={transformers_version}, "
            f"but transformers>={min_transformers_version} is expected. "
            f"This may produce a corrupted checkpoint and lead to unexpected behavior. "
            f"Please regenerate your base LLaMA checkpoint with transformers>={min_transformers_version}."
        )

    model_raw = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, device_map={"": torch.device(device)}, torch_dtype=torch.float32
    )
    tokenizer_raw = transformers.AutoTokenizer.from_pretrained(model_dir)
    if tokenizer_raw.pad_token is None:
        stable_resize_token_embeddings_and_tokenizer(
            model=model_raw, tokenizer=tokenizer_raw, special_tokens_dict=dict(pad_token="[PAD]")
        )
    return model_raw.eval(), tokenizer_raw


def reconstruct_tuned_model(model_tuned, model_raw, is_reward_model=False):
    # modifies model_tuned in-place
    state_dict_diff = model_tuned.state_dict()
    state_dict_raw = model_raw.state_dict()
    if is_reward_model:
        # reward model adds nesting to main transformer
        state_dict_raw = {f"backbone_model.{k}": v for k, v in state_dict_raw.items()}
    for key in state_dict_raw:
        if state_dict_raw[key].size() != state_dict_diff[key].size():
            # weights with a size mismatch are not diff'd in the upload
            continue
        state_dict_diff[key].add_(state_dict_raw[key])


def integrity_check(model_tuned, hf_hub_name):
    model_sum = sum(param.sum() for param in model_tuned.state_dict().values()).item()
    model_sum_file = hf_hub_download(repo_id=hf_hub_name, filename="model_sum.txt")
    with open(model_sum_file, "r") as f:
        model_sum_hf_hub = float(f.read())
    return np.isclose(model_sum_hf_hub, model_sum)


if __name__ == "__main__":
    model_names = get_alpaca_farm_model_names()
    args = build_argparse(model_names)

    model_names = model_names if args.alpaca_farm_model_name == "all" else [args.alpaca_farm_model_name]
    for model_name in model_names:
        print("Downloading", model_name)

        hf_hub_name = f"tatsu-lab/alpaca-farm-{model_name}-wdiff"
        is_reward_model = "reward-model" in model_name
        save_dir = os.path.join(args.models_save_dir, model_name)

        model_tuned, tokenizer_tuned = load_weight_diff(hf_hub_name, is_reward_model, args.device, args.path_to_sft10k)
        model_raw, tokenizer_raw = load_raw_model(args.llama_7b_hf_dir, args.device)
        reconstruct_tuned_model(model_tuned, model_raw, is_reward_model)

        if not integrity_check(model_tuned, hf_hub_name):
            print("Model weights integrity check failed. Did you use the latest llama-7b HF weights?")
        model_tuned.save_pretrained(save_dir)
        tokenizer_tuned.save_pretrained(save_dir)

        print("Downloaded to", save_dir)
