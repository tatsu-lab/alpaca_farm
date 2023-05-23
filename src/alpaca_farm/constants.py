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

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_CACHE_DIR = None
WANDB_PROJECT = "alpaca_farm"

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

# Huggingface model naming convention.
WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF2_WEIGHTS_INDEX_NAME = "tf_model.h5.index.json"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
FLAX_WEIGHTS_INDEX_NAME = "flax_model.msgpack.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
GENERATION_CONFIG_NAME = "generation_config.json"
MODEL_CARD_NAME = "modelcard.json"

TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
