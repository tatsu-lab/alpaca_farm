# maps to common.py
import os
import time
import types
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import torch
import torch.distributed as dist
import transformers
from torch import Tensor, nn
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from transformers.trainer import WEIGHTS_NAME, is_deepspeed_zero3_enabled

from . import constants, logging, utils
from .types import AnyPath, AnyPathOrNone

logger = logging.get_logger(__name__)

# TODO(lxuechen): Put everything flasn-attn bit in own folder.
try:
    from flash_attn import bert_padding
except ImportError as e:
    logger.warning(f"Failed to import flash attention with error {e}")

# Separate this out, since llama model is not stable.
try:
    from .flash_models import flash_llama
except ImportError as e:
    logger.warning(f"Failed to import flash attention llama with error {e}")


def apex_is_installed():
    try:
        import apex

        return True
    except ImportError as _:
        return False


class staggered_object_creation(object):
    """
    Objection creation in a distributed setting could be very RAM-intensive.

    This function staggers the creation of objects on odd and even ranks, so that not all objects
    are created at once.

    Assumes local_rank == -1 means no distributed training.
    """

    def __init__(self, local_rank: int):
        super().__init__()
        self.local_rank = local_rank

    def __enter__(self, *args, **kwargs):
        del args, kwargs
        if self.local_rank != -1 and self.local_rank % 2 == 0:
            dist.barrier()
        return self

    def __exit__(self, *args, **kwargs):
        del args, kwargs
        if self.local_rank != -1:
            if self.local_rank % 2 == 1:
                dist.barrier()
            dist.barrier()  # Final safety barrier.

    def __call__(self, func):
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator


def make_generative_lm(
    model_name_or_path: str,
    flash_attn: bool,
    fp16: Optional[bool] = None,
    bf16: Optional[bool] = None,
    mixed_precision: Optional[str] = None,
    **kwargs,
):
    if fp16 is None:
        fp16 = mixed_precision == "fp16"
    if bf16 is None:
        bf16 = mixed_precision == "bf16"

    if flash_attn and not fp16 and not bf16:
        logger.warning(
            "Flash attention does not support fp32. Reverting to standard attention.", main_process_only=True
        )
        flash_attn = False

    if flash_attn:
        model_cls = flash_llama.LlamaForCausalLM
    else:
        model_cls = transformers.LlamaForCausalLM

    return model_cls.from_pretrained(model_name_or_path, **kwargs)


def let_model_save_mem_when_zero_grad(model: nn.Module):
    def new_zero_grad(self, set_to_none: bool = True) -> None:
        r"""Sets gradients of all model parameters to zero. See similar function
        under :class:`torch.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """
        if getattr(self, "_is_replica", False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead."
            )

        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    # Make zero_grad `set_to_none=True` by default.
    # Need this runtime method patching, since self is used within zero_grad.
    model.zero_grad = types.MethodType(new_zero_grad, model)
    return model


def pad_to_multiples_of_x(tensor: Tensor, x: int = 8):
    """Pad a tensor along the batch dimension to a multiple of x."""
    total_nnz, hidden_size = tensor.size()
    pad_len = (x - total_nnz % x) % x
    if pad_len != 0:
        tensor = torch.cat(
            [
                tensor,
                torch.zeros([pad_len, hidden_size], device=tensor.device, dtype=tensor.dtype),
            ],
            dim=0,
        )

    def unpad_x(padded_tensor):
        return padded_tensor[:-pad_len] if pad_len > 0 else padded_tensor

    return tensor, unpad_x


def unpad_input(padded: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, Callable, torch.Tensor, int]:
    """Wrapper for unpad_input in official flash-attn."""
    batch_size, padded_seqlen = padded.shape[:2]
    unpadded, indices, cu_seqlens, max_seqlen = bert_padding.unpad_input(padded, attention_mask)

    def pad_back(unpadded: torch.Tensor):
        return bert_padding.pad_input(unpadded, indices, batch_size, padded_seqlen)

    return unpadded, pad_back, cu_seqlens, max_seqlen


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, give_rw_access=True, rank0_only=True
):
    """Collects the state dict and dump to disk."""
    now = time.perf_counter()

    if trainer.fsdp is not None:
        # NOTE(rtaori): technically should be rank0_only=True (otherwise duplicates model in RAM),
        # but currently there seems to be a bug in FSDP that causes it to hang.
        # Migration to Pytorch 2 should fix this.
        # Once we migrate, we can also implement more efficient loading:
        # https://github.com/pytorch/pytorch/blob/master/torch/distributed/fsdp/api.py#L286-L295
        # NOTE(tianyi): tested on sphinx6, seems to work fine with rank0_only=False
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
        with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = trainer.model.state_dict()
            if trainer.args.should_save:
                trainer._save(output_dir, state_dict=state_dict)  # noqa

    elif trainer.deepspeed is not None:
        # --- The stuff below is almost a copy from transformers.trainer.Trainer.save_model (transformers==4.27.3) ---
        # this takes care of everything as long as we aren't under zero3
        if trainer.args.should_save:
            trainer._save(output_dir)

        if is_deepspeed_zero3_enabled():
            # It's too complicated to try to override different places where the weights dump gets
            # saved, so since under zero3 the file is bogus, simply delete it. The user should
            # either use deepspeed checkpoint to resume or to recover full weights use
            # zero_to_fp32.py stored in the checkpoint.
            if trainer.args.should_save:
                file = os.path.join(output_dir, WEIGHTS_NAME)
                if os.path.isfile(file):
                    logger.warning(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                    os.remove(file)

            # now save the real model if stage3_gather_16bit_weights_on_model_save=True
            # if false it will not be saved.
            # This must be called on all ranks
            if not trainer.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                logger.warning(
                    "deepspeed.save_16bit_model didn't save the model, since"
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                trainer.deepspeed.save_checkpoint(output_dir)
                # --- End of shameless copy ---

                # Auto-convert the checkpoint to fp32 for easier downstream use.
                # Only rank0 shall do the checkpoint conversion to prevent race conditions.
                if trainer.args.should_save:
                    try:
                        os.system(
                            f"python {output_dir}/zero_to_fp32.py  '{output_dir}' '{output_dir}/pytorch_model.bin'"
                        )
                    except Exception as e:
                        logger.fatal(f"Failed to convert zero3 checkpoint to fp32: {e}")

    else:  # Also support saving for non-FSDP models.
        # NOTE(lxuechen): Saving and loading T5 has weird pickle issues due to device map.
        #  Wasn't able to exactly pinpoint. But saving to and loading from CPU seems to work.
        #  In principle, trainer.save_model() should do the same thing, but breaks in practice.
        #  We drop T5 support.
        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

    if trainer.args.should_save:
        if give_rw_access:
            try:
                os.system(f"chmod -R a+xwr {output_dir}")
            except Exception as e:
                logger.fatal(f"Failed to give read-write access to {output_dir}: {e}")
        logger.warning(f"Saving model took {time.perf_counter() - now:.2f} seconds.")


def unpack_dict(d: Dict, keys: Sequence[str], return_type: type = tuple) -> Union[Sequence, Dict]:
    if return_type in (tuple, list):
        return return_type(d[key] for key in keys)
    elif return_type == dict:
        return {key: d[key] for key in keys}
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


def model_name_or_path_exists(model_name_or_path: AnyPath) -> bool:
    try:
        transformers.PretrainedConfig.get_config_dict(model_name_or_path)
    except OSError:
        return os.path.exists(Path(model_name_or_path) / "trainer_state.json")
    return True


# TODO(lxuechen): Simplify this logic.
def get_pretrained_model_name_with_model_name_or_path(model_name_or_path: AnyPathOrNone) -> Optional[str]:
    """Get the name of the pretrained model with a model name or path.

    Examples:
    >>> get_pretrained_model_name_with_model_name_or_path(
        "/juice5/scr5/nlp/crfm/human-feedback/models/selfinstruct/sft_opt_6b_clean_v0_3ep")
    "facebook/opt-6.7b"  # Fine-tuned model started from this model.

    >>> get_pretrained_model_name_with_model_name_or_path("facebook/opt-125m")
    "facebook/opt-125m"  # This should not be "opt-125m".
    """
    if model_name_or_path is None:
        return None

    if not model_name_or_path_exists(model_name_or_path):
        raise ValueError(f"Model name or path does not exist: {model_name_or_path}")

    pretrained_model_name = model_name_or_path

    # While `model_name_or_path` points to a dir, find the original pretrained model name by recursively going down the
    # chain. This works by recursively extracting the name/path to the parent model from the `config.json` file, until
    # we get a non-path name.
    while os.path.isdir(pretrained_model_name):
        pretrained_model_name = utils.jload(os.path.join(pretrained_model_name, "config.json"))["_name_or_path"]

    # When recursive finding doesn't work, we revert to the dumb hardcoded rule. This only supports llama models so far.
    if pretrained_model_name not in constants.MODEL_NAME_TO_FAMILY:
        logger.warning(
            "One of the `_name_or_path` to the root node is not a pretrained model name or local path on disk. "
            "This may be due to copying checkpoints across machines. "
            "Falling back to hardcoded rule to figure out the pretrained model based on config.json."
        )
        config = utils.jload(os.path.join(model_name_or_path, "config.json"))
        model_config = unpack_dict(config, keys=("model_type", "num_hidden_layers", "hidden_size"), return_type=dict)
        pretrained_model_name = [k for k, v in constants.MODEL_NAME_TO_CONFIG.items() if v == model_config][0]

    return str(pretrained_model_name)


def get_transformer_hidden_size(model: transformers.PreTrainedModel):
    if isinstance(model, transformers.GPT2LMHeadModel):
        hidden_size_attr_name = "n_embd"
    elif isinstance(model, transformers.OPTForCausalLM):
        hidden_size_attr_name = "word_embed_proj_dim"
    elif isinstance(model, transformers.T5ForConditionalGeneration):
        hidden_size_attr_name = "d_model"
    else:
        # Hack to deal with the fact that transformers library changed the LLaMA model name.
        llama_cls = getattr(
            transformers, "LLaMAForCausalLM" if hasattr(transformers, "LLaMAForCausalLM") else "LlamaForCausalLM"
        )
        if isinstance(model, llama_cls):
            hidden_size_attr_name = "hidden_size"
        else:
            raise ValueError(f"Unknown base_model type: {type(model)}")
    return getattr(model.config, hidden_size_attr_name)
