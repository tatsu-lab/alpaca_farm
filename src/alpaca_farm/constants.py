# maps to constants.py
import os
from pathlib import Path

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

FAST_MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "/self/scr-sync/nlp/huggingface_hub_llms"))
DEFAULT_CACHE_DIR = ""
WANDB_PROJECT = "human-feedback"

MODEL_NAME_TO_CONFIG = {
    "llama-7b": {"model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096},
    "llama-13b": {"model_type": "llama", "num_hidden_layers": 40, "hidden_size": 5120},
    "llama-30b": {"model_type": "llama", "num_hidden_layers": 60, "hidden_size": 6656},
    "llama-65b": {"model_type": "llama", "num_hidden_layers": 80, "hidden_size": 8192},
}

MODEL_NAME_TO_FAMILY = {
    "distilgpt2": "gpt2",
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2",
    "gpt2-large": "gpt2",
    "gpt2-xl": "gpt2",
    "facebook/opt-iml-max-1.3b": "opt",
    "facebook/opt-125m": "opt",
    "facebook/opt-350m": "opt",
    "facebook/opt-1.3b": "opt",
    "facebook/opt-2.7b": "opt",
    "facebook/opt-6.7b": "opt",
    "facebook/opt-13b": "opt",
    "facebook/opt-30b": "opt",
    "llama-teeny": "llama",
    "llama-7b": "llama",
    "llama-13b": "llama",
    "llama-30b": "llama",
    "llama-65b": "llama",
    "EleutherAI/pythia-2.8b-deduped": "pythia",
    "EleutherAI/pythia-6.9b-deduped": "pythia",
    "EleutherAI/pythia-12b-deduped": "pythia",
}
