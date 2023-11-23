import copy

import torch
import torch.nn.functional as F
import transformers

from .. import common

LABEL_NAMES = ["input_ids_w", "labels_w", "attention_mask_w", "input_ids_l", "labels_l", "attention_mask_l"]


class Trainer(transformers.Trainer):
    def __init__(self, model, args, *argv, **kwargs):
        args.label_names = LABEL_NAMES
        super().__init__(model, args, *argv, **kwargs)
        # TODO: Load model on GPU and shard it.
        self.ref_model = copy.deepcopy(model).to("cuda")

    def compute_loss(self, model, inputs, return_outputs=False):
        # TODO: This implementation is simple and readable, but it's not efficient.
        #  Since the instruction+input is shared for the winning and losing sequences, one can in principle
        #  just do a single forward pass on this part for each model, instead of doing the full forward
        #  twice (one for winning and one for losing sequence).
        #  So here's the efficient implementation:
        #   1. Do a single forward pass on the instruction+input for policy. Retain the kv cache.
        #   2. Do a forward pass on the winning response for model, using the kv cache.
        #   3. Do a forward pass on the losing response for model, using the kv cache.
        #   4. Follow a similar procedure for the reference model, except don't need to retain activations for backprop.
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
