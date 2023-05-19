from typing import Optional, Sequence

import tiktoken
import logging

from .. import openai_utils


def openai_completions(
    prompts: list[str],
    model_name: str,
    tokens_to_avoid: Optional[Sequence[str]] = None,
    tokens_to_favor: Optional[Sequence[str]] = None,
    is_skip_multi_tokens_to_avoid: bool = True,
    is_strip: bool = True,
    num_procs: Optional[int] = None,
    batch_size: Optional[int] = None,
    **decoding_kwargs,
) -> list[str]:
    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        logging.info(f"Auto annotating {n_examples} samples using {model_name}.")

    if tokens_to_avoid or tokens_to_favor:
        tokenizer = tiktoken.encoding_for_model(model_name)

        logit_bias = decoding_kwargs.get("logit_bias", {})
        if tokens_to_avoid is not None:
            for t in tokens_to_avoid:
                curr_tokens = tokenizer.encode(t)
                if len(curr_tokens) != 1 and is_skip_multi_tokens_to_avoid:
                    logging.warning(
                        f"Token {t} has more than one token, skipping because `is_skip_multi_tokens_to_avoid`."
                    )
                    continue
                for tok_id in curr_tokens:
                    logit_bias[tok_id] = -100  # avoids certain tokens

        if tokens_to_favor is not None:
            for t in tokens_to_favor:
                curr_tokens = tokenizer.encode(t)
                for tok_id in curr_tokens:
                    logit_bias[tok_id] = 7  # increase log prob of tokens to match

        decoding_kwargs["logit_bias"] = logit_bias

    if is_strip:
        prompts = [p.strip() for p in prompts]

    if openai_utils.requires_chatml(model_name):
        decoding_args = openai_utils.OpenAIDecodingArgumentsChat()
        num_procs = num_procs or 5
        batch_size = batch_size or 1
    else:
        decoding_args = openai_utils.OpenAIDecodingArguments()
        num_procs = num_procs or 1
        batch_size = batch_size or 10

    logging.info(f"Kwargs to completion: {decoding_args} {decoding_kwargs}")

    completions = openai_utils.openai_completion(
        prompts=prompts,
        decoding_args=decoding_args,  # not useful, openai_completion should initialize this if None
        return_text=True,
        batch_size=batch_size,
        model_name=model_name,
        num_procs=num_procs,
        **decoding_kwargs,
    )

    return completions
