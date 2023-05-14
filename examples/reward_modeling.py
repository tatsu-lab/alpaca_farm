import contextlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import einops
import torch
import torch.nn.functional as F
import transformers
from transformers.trainer_utils import EvalPrediction

from alpaca_farm import common, constants, torch_ops
from alpaca_farm.models import reward_model


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=constants.SHARED_MODEL_DIR / "selfinstruct" / "sft_v6_llama_7b_regen_v7_3ep",
        metadata={"help": "Name of or path to the base generative LM."},
    )


@dataclass
class DataArguments:
    train_file_path: Optional[str] = field(default=None)
    eval_file_path: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of training samples to use. By default uses the whole dataset."}
    )
    train_sql: Optional[str] = field(
        default=None,
        metadata={"help": "SQL query to select training data."},
    )
    eval_sql: Optional[str] = field(
        default=None,
        metadata={"help": "SQL query to select validation data."},
    )
    prompt_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the prompt to use. If using SQL for data loading, prompt_name must be provided."},
    )
    eval_size: int = field(
        default=500,
        metadata={
            "help": "When no eval data is specified (`eval_file_path` and `eval_sql` are both None), "
                    "controls the number of samples from the original training set split out to use for evaluation."
        },
    )
    dedup_instruction: bool = field(
        default=True,
        metadata={"help": "If true, deduplicates training data by input and instruction. "},
    )
    database_key: str = field(
        default="instruction_following",
        metadata={"help": "Name of the database for SQL queries."},
    )
    randomize_label_frac: float = field(
        default=0.0,
        metadata={"help": "Fraction of train+eval data for which to randomize label."},
    )
    randomize_label_seed: int = field(
        default=42,
        metadata={"help": "Seed for randomizing labels."},
    )

    def __post_init__(self):
        if sum(opt is not None for opt in [self.train_file_path, self.train_sql]) != 1:
            raise ValueError("Exactly one of `train_file_path` or `train_sql` must be specified.")
        if sum(opt is not None for opt in [self.eval_file_path, self.eval_sql]) > 1:
            raise ValueError("At most one of `eval_file_path` or `eval_sql` must be specified.")
        if any([self.train_sql, self.eval_sql]) and self.prompt_name is None:
            raise ValueError("If using SQL for data loading, prompt_name must be provided.")

        train_df_postprocessor = []
        eval_df_postprocessor = []
        if self.dedup_instruction:
            train_df_postprocessor.append(DedupInstructionDFPostProcessor())
        if self.randomize_label_frac > 0:
            train_df_postprocessor.append(
                RandomizeRewardModelingLabelsDFPostProcessor(self.randomize_label_frac, self.randomize_label_seed)
            )

        self.train_df_postprocessor = SequentialPostProcessor(train_df_postprocessor)
        self.eval_df_postprocessor = SequentialPostProcessor(eval_df_postprocessor)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pad_token: str = field(default=constants.DEFAULT_PAD_TOKEN)
    cache_dir: str = field(default=constants.DEFAULT_CACHE_DIR)
    wandb_project: str = field(default=constants.WANDB_PROJECT)
    flash_attn: bool = field(default=True, metadata={"help": "Whether to use flash attention."})
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    label_names: List[str] = field(
        default_factory=lambda: ["index_0", "index_1", "choice"],
        metadata={
            "help": "Names of the labels in the dataset. "
                    "This is needed to get transformers.Trainer to not throw those tensors away before `compute_loss`."
                    "By default, the trainer throws away columns it doesn't recognize when creating the "
                    "`train_dataloader` (see `_remove_unused_columns`). "
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
                    "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    initialize_model_on_cpu: bool = field(
        default=False,
        metadata={
            "help": "Whether to initialize the model on CPU. "
                    "If True, models on all processes will be first initialized on CPU; this is RAM-costly but faster."
        },
    )
    end_sequence_with_eos: bool = field(
        default=False,
        metadata={
            "help": "Whether to end sequences with EOS. "
                    "Ending with EOS might help the reward model realize it's time to predict."
        },
    )


class Trainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # input_ids, attention_mask each of size (bsz, num_candidates, seq_len).
        # index_0, index_1 each of size (bsz, num_pairs); indexes into input_ids.
        # choice of size (bsz, num_pairs); 1 if index_1's seq is chosen, 0 otherwise.
        input_ids, attention_mask, index_0, index_1, choice = common.unpack_dict(
            inputs, keys=("input_ids", "attention_mask", "index_0", "index_1", "choice")
        )
        num_candidates, num_pairs = input_ids.size(1), choice.size(1)
        input_ids_flat, attention_mask_flat = tuple(
            einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids, attention_mask)
        )
        outputs = model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        rewards_flat = outputs.rewards
        rewards = einops.rearrange(rewards_flat, "(b c) -> b c", c=num_candidates)  # Size: (bsz, num_candidates).

        rewards_0, rewards_1 = tuple(
            torch_ops.batch_select(rewards, index) for index in (index_0, index_1)
        )  # Size: (bsz, num_pairs).
        logits = rewards_1 - rewards_0  # Size: (bsz, num_pairs).
        # Type casting of `choice` is due to amp.autocast context manager.
        loss = F.binary_cross_entropy_with_logits(logits, choice.to(logits.dtype), reduction="mean")
        return (loss, dict(logits=logits)) if return_outputs else loss


def compute_reward_modeling_metrics(eval_prediction: EvalPrediction) -> Dict:
    # eval_prediction.label_ids is a tuple that matches up with `training_args.label_names`.
    logits = torch.tensor(eval_prediction.predictions).squeeze(-1)
    labels = torch.tensor(eval_prediction.label_ids[-1]).squeeze(-1)
    predictions = (logits >= 0.0).long()
    accuracy = predictions.eq(labels).float().mean().item()
    label_positive_rate = (labels == 1).float().mean().item()
    positive_rate = (predictions == 1).float().mean().item()
    true_positive_rate = (predictions * labels).float().sum().item() / labels.sum().item()
    false_positive_rate = (predictions * (1 - labels)).float().sum().item() / (1 - labels).sum().item()
    return dict(
        accuracy=accuracy,
        label_positive_rate=label_positive_rate,
        positive_rate=positive_rate,
        true_positive_rate=true_positive_rate,
        false_positive_rate=false_positive_rate,
    )


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ["WANDB_PROJECT"] = training_args.wandb_project

    if training_args.deepspeed is not None:
        ctx_mgr = contextlib.nullcontext()
        device_map = None
        low_cpu_mem_usage = None
    elif training_args.initialize_model_on_cpu:
        ctx_mgr = contextlib.nullcontext()
        device_map = None
        low_cpu_mem_usage = True
    else:
        ctx_mgr = common.staggered_object_creation(local_rank=training_args.local_rank)
        device_map = {"": training_args.device.index}
        low_cpu_mem_usage = True

    with ctx_mgr:
        config = reward_model.RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)
        model = reward_model.RewardModel(
            flash_attn=training_args.flash_attn,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            config=config,
        )
        common.let_model_save_mem_when_zero_grad(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",  # Ensure reward is always extracted at the last token embedding.
        use_fast=False,  # Fast GPT2 tokenizer can break when we start counting the truncations.
    )
    tokenizer.padding = training_args.padding
    data_module = data_utils.make_binary_reward_modeling_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )
    print(f"number of training examples: {len(data_module['train_dataset'])}")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_reward_modeling_metrics,
        **data_module,
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_state()
    common.safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
