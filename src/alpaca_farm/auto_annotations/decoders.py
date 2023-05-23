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

import logging
from typing import Optional, Sequence

import tiktoken

from .. import openai_utils


def openai_completions(
    prompts: list[str],
    model_name: str,
    tokens_to_favor: Optional[Sequence[str]] = None,
    tokens_to_avoid: Optional[Sequence[str]] = None,
    is_skip_multi_tokens_to_avoid: bool = True,
    is_strip: bool = True,
    num_procs: Optional[int] = None,
    batch_size: Optional[int] = None,
    **decoding_kwargs,
) -> list[str]:
    """Get openai completions for the given prompts. Allows additional parameters such as tokens to avoid and
    tokens to favor.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str
        Name of the model to use for decoding.

    tokens_to_favor : list of str, optional
        Substrings to favor in the completions. We will add a positive bias to the logits of the tokens constituting
        the substrings.

    tokens_to_avoid : list of str, optional
        Substrings to avoid in the completions. We will add a large negative bias to the logits of the tokens
        constituting the substrings.

    is_skip_multi_tokens_to_avoid : bool, optional
        Whether to skip substrings from tokens_to_avoid that are constituted by more than one token => avoid undesired
        side effects on other tokens.

    is_strip : bool, optional
        Whether to strip trailing and leading spaces from the prompts.

    decoding_kwargs :
        Additional kwargs to pass to `openai_utils.openai_completion`.

    Example
    -------
    >>> prompts = ["Respond with one digit: 1+1=", "Respond with one digit: 2+2="]
    >>> openai_completions(prompts, "text-davinci-003", tokens_to_avoid=["2"," 2"])
    ['\n\nAnswer: \n\nTwo (or, alternatively, the number "two" or the numeral "two").', '\n\n4']
    >>> openai_completions(prompts, "text-davinci-003", tokens_to_favor=["2"])
    ['2\n\n2', '\n\n4']
    >>> openai_completions(prompts, "text-davinci-003", tokens_to_avoid=["2 a long sentence that is not a token"])
    ['\n\n2', '\n\n4']
    >>> chat_prompt = ["<|im_start|>user\n1+1=<|im_end|>", "<|im_start|>user\nRespond with one digit: 2+2=<|im_end|>"]
    >>> openai_completions(chat_prompt, "gpt-3.5-turbo", tokens_to_avoid=["2"," 2"])
    ['As an AI language model, I can tell you that 1+1=  𝟮.', '4']
    """
    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        logging.info(f"Auto annotating {n_examples} prompts using {model_name}.")

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

    logging.info(f"Kwargs to completion: {decoding_kwargs}")

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
