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
