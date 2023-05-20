from typing import Sequence

import torch
import torch.nn.functional as F

from .types import Tensor


def batch_select(input: Tensor, index: Tensor):
    """Select elements from a batched tensor with a batched index tensor.

    Example:
        input = torch.tensor([
            [0, 1, 2],
            [3, 0, 9],
            [6, 7, 8],
        ])
        index = torch.tensor([[0, 1], [1, 0], [0, 0]])
        batch_select(input, index) = tensor([
            [0, 1],
            [0, 3],
            [6, 6]
        ])
    """
    dummy_index = torch.arange(input.size(0), device=input.device).unsqueeze(-1)
    return input[dummy_index, index]


def pad_sequence_from_left(
    sequences: Sequence[Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
):
    """Mirror of `torch.nn.utils.rnn.pad_sequence`, but pad from left."""
    sequences = tuple(sequence.flip(0) for sequence in sequences)
    padded_sequence = torch._C._nn.pad_sequence(sequences, batch_first, padding_value)
    padded_sequence = padded_sequence.flip(int(batch_first))
    return padded_sequence


def compute_logprobs(logits: Tensor, labels: Tensor, ignore_index: int) -> Tensor:
    """Compute per-token logprobs, zeroing out places with ignore_index (padding)."""
    return -F.cross_entropy(logits.permute(0, 2, 1), labels, reduction="none", ignore_index=ignore_index)


def whiten(values: Tensor, shift_mean=True, epsilon=1e-8) -> Tensor:
    assert values.size(0) >= 8, f"Internal error: Minibatch size {values.size(0)} is insufficient for whitening."
    mean, std = values.mean(), values.std(unbiased=False)  # noqa
    whitened = (values - mean) / (std + epsilon)
    if not shift_mean:
        whitened = whitened + mean
    return whitened
