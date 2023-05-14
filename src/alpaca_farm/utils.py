# maps to ml_swissknife/utils.py
from typing import Sequence, Callable, Optional, Union
from .types import Numeric
import numpy as np
import json

def alleq(l: Sequence, f: Optional[Callable] = lambda x, y: x == y):
    """Check all arguments in a sequence are equal according to a given criterion.
    Args:
        f: A bi-variate boolean function.
        l: A list/tuple.
    Returns:
        True if everything is equal; otherwise False.
    """
    return all(f(l[0], li) for li in l[1:])

def zip_(*args: Sequence):
    """Assert sequences of same length before zipping."""
    if len(args) == 0:
        return []
    assert alleq(args, lambda x, y: len(x) == len(y))
    return zip(*args)

def jdumps(obj, indent=4, default=str):
    return json.dumps(obj, indent=indent, default=default)

def flatten_nested_pystruct(sequence: Sequence):
    """Flatten nested python list/tuple/set and return a list of elements."""
    if not isinstance(sequence, (tuple, list, set)):
        return [sequence]
    return [i for entry in sequence for i in flatten_nested_pystruct(entry)]

def mean(*seqs: Sequence[Numeric]) -> Union[Numeric, Sequence[Numeric]]:
    singleton = len(seqs) == 1
    means = [float(np.mean(seq)) for seq in seqs]
    return means[0] if singleton else means

def smart_tokenizer_and_embedding_resize(
        special_tokens_dict,
        tokenizer,
        model,
):
    """Resize tokenizer and embedding in a smart way.

    Notes:
        For new tokens, the embedding value is the average of all old embedding vectors.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
