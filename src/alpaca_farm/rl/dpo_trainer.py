import copy
import functools
import inspect

import torch
import torch.nn.functional as F
import transformers
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.trainer_utils import FSDPOption

from .. import common

LABEL_NAMES = ["input_ids_w", "labels_w", "attention_mask_w", "input_ids_l", "labels_l", "attention_mask_l"]


class Trainer(transformers.Trainer):
    def __init__(self, model, args, *argv, **kwargs):
        args.label_names = LABEL_NAMES
        super().__init__(model, args, *argv, **kwargs)
        self.ref_model = self._prepare_model(copy.deepcopy(model))

    def _prepare_model(self, model) -> FSDP:
        """Generic code that prepares a model by sharding according to FSDP and distributing the shards to devices.

        Run this after `super().__init__` finished.
        This is a copy of a code portion from `Trainer.__init__` from transformers.
        """
        if FSDPOption.OFFLOAD in self.args.fsdp:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = CPUOffload(offload_params=False)

        auto_wrap_policy = None

        if FSDPOption.AUTO_WRAP in self.args.fsdp:
            if self.args.fsdp_config["fsdp_min_num_params"] > 0:
                auto_wrap_policy = functools.partial(
                    size_based_auto_wrap_policy, min_num_params=self.args.fsdp_config["fsdp_min_num_params"]
                )
            elif self.args.fsdp_config.get("fsdp_transformer_layer_cls_to_wrap", None) is not None:
                transformer_cls_to_wrap = set()
                for layer_class in self.args.fsdp_config["fsdp_transformer_layer_cls_to_wrap"]:
                    transformer_cls = get_module_class_from_name(model, layer_class)
                    if transformer_cls is None:
                        raise Exception("Could not find the transformer layer class to wrap in the model.")
                    else:
                        transformer_cls_to_wrap.add(transformer_cls)
                auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    # Transformer layer class to wrap
                    transformer_layer_cls=transformer_cls_to_wrap,
                )
        mixed_precision_policy = None
        dtype = None
        if self.args.fp16:
            dtype = torch.float16
        elif self.args.bf16:
            dtype = torch.bfloat16
        if dtype is not None:
            mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        if not isinstance(model, FSDP):
            # XXX: Breaking the self.model convention but I see no way around it for now.
            signature = inspect.signature(FSDP.__init__).parameters.keys()
            kwargs = {}
            for arg in ["limit_all_gathers", "forward_prefetch", "backward_prefetch"]:
                if arg in signature:
                    kwargs[arg] = getattr(self, arg)
            model = FSDP(
                model,
                sharding_strategy=self.fsdp,
                cpu_offload=cpu_offload,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision_policy,
                device_id=self.args.device,
                **kwargs,
            )
        return model

    def compute_loss(self, model, inputs, return_outputs=False):
        # TODO: This implementation is simple and readable, but it's not efficient.
        #  Since the instruction+input is shared for the winning and losing sequences, one can in principle
        #  just do a single forward pass on this part for model and ref_model, instead of doing the full forward
        #  twice (one for winning and one for losing sequence) for model and ref_model.
        #  So here's the efficient implementation:
        #   1. Do a single forward pass on the instruction+input for model. Retain the kv cache.
        #   2. Do a forward pass on the winning response for model, using the kv cache.
        #   3. Do a forward pass on the losing response for model, using the kv cache.
        #   4. Follow a similar procedure for ref_model, except don't retain activations for backprop
        #       (but do temporarily retain the kv cache).
        #       There's an explicit speed/memory tradeoff here -- retaining kv cache saves FLOPs but uses more memory.
        #   5. Compute the loss.
        input_ids_w, labels_w, attention_mask_w, input_ids_l, labels_l, attention_mask_l = common.unpack_dict(
            inputs, LABEL_NAMES
        )
        labels_w, labels_l = labels_w[..., 1:], labels_l[..., 1:]

        with torch.no_grad():
            ref_logits_w = self.ref_model(input_ids=input_ids_w, attention_mask=attention_mask_w).logits[..., :-1, :]
            ref_logits_l = self.ref_model(input_ids=input_ids_l, attention_mask=attention_mask_l).logits[..., :-1, :]
            ref_logprobs_w = F.cross_entropy(ref_logits_w.transpose(-1, -2), labels_w, reduction="none").sum(-1)
            ref_logprobs_l = F.cross_entropy(ref_logits_l.transpose(-1, -2), labels_l, reduction="none").sum(-1)

        logits_w = model(input_ids=input_ids_w, attention_mask=attention_mask_w).logits[..., :-1, :]
        logits_l = model(input_ids=input_ids_l, attention_mask=attention_mask_l).logits[..., :-1, :]
        logprobs_w = F.cross_entropy(logits_w.transpose(-1, -2), labels_w, reduction="none").sum(-1)
        logprobs_l = F.cross_entropy(logits_l.transpose(-1, -2), labels_l, reduction="none").sum(-1)

        preference_logits = self.args.beta * ((logprobs_w - ref_logprobs_w) - (logprobs_l - ref_logprobs_l))
        loss = -F.logsigmoid(preference_logits).mean(0)
        return loss
