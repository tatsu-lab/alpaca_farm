# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Light wrapper for OpenAI API.

Reference API:
    https://beta.openai.com/docs/api-reference/completions/create

Internal map:
    https://github.com/lxuechen/ml-swissknife/blob/main/ml_swissknife/openai_utils.py
"""
import copy
import dataclasses
import functools
import logging
import math
import multiprocessing
import os
import random
import sys
import time
from typing import Optional, Sequence, Union

import openai
import tqdm
from openai import openai_object

StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclasses.dataclass
class OpenAIDecodingArgumentsBase(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    # Heuristic stop when about to generate next function.
    # stop: Optional[Tuple[str, ...]] = ("}\n\nstatic", "}\n\n/*")
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # If you need these, pass them in as decoding_kwargs.
    # best_of: int = 1
    # logit_bias: dict = None


@dataclasses.dataclass
class OpenAIDecodingArguments(OpenAIDecodingArgumentsBase):
    suffix: Optional[str] = None
    logprobs: Optional[int] = None
    echo: bool = False


@dataclasses.dataclass
class OpenAIDecodingArgumentsChat(OpenAIDecodingArgumentsBase):
    # currently there are no arguments that are different than not chat version
    pass


def requires_chatml(model: str) -> bool:
    """Whether a model requires the ChatML format."""
    # TODO: this should ideally be an OpenAI function... Maybe it already exists?
    return "turbo" in model or "gpt-4" in model


def convert_dict_to_openai_object(data: dict) -> openai_object.OpenAIObject:
    return_data = openai_object.OpenAIObject()
    return_data.update(data)
    return return_data


def _openai_completion_helper(
    prompt_batch: Sequence[StrOrOpenAIObject],
    is_chat: bool,
    sleep_time: int,
    openai_organization_ids : Optional[Sequence[str]]= None,
    openai_api_key: Optional[str] = None,
    **shared_kwargs,
):
    if openai_api_key is not None:
        openai.api_key = openai_api_key

    # randomly select orgs
    if openai_organization_ids is not None:
        openai.organization = random.choice(openai_organization_ids)

    # copy shared_kwargs to avoid modifying it
    shared_kwargs = copy.deepcopy(shared_kwargs)

    while True:
        try:
            if is_chat:
                completion_batch = openai.ChatCompletion.create(messages=prompt_batch[0], **shared_kwargs)

                choices = completion_batch.choices
                for choice in choices:
                    assert choice.message.role == "assistant"
                    if choice.message.content == "":
                        choice["text"] = " "  # annoying doesn't allow empty string
                    else:
                        choice["text"] = choice.message.content

            else:
                completion_batch = openai.Completion.create(prompt=prompt_batch, **shared_kwargs)
                choices = completion_batch.choices

            for choice in choices:
                choice["total_tokens"] = completion_batch.usage.total_tokens / len(prompt_batch)
            break
        except openai.error.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                shared_kwargs["max_tokens"] = int(shared_kwargs["max_tokens"] * 0.8)
                logging.warning(f"Reducing target length to {shared_kwargs['max_tokens']}, Retrying...")
            else:
                logging.warning("Hit request rate limit; retrying...")
                if openai_organization_ids is not None and len(openai_organization_ids) > 1:
                    openai.organization = random.choice([o for o in openai_organization_ids if o != openai.organization])
                    logging.warning(f"Switching to organization: {openai.organization} for OAI API key.")
                time.sleep(sleep_time)  # Annoying rate limit on requests.
    return choices


def _openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="text-davinci-003",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    num_procs=1,
    **decoding_kwargs,
) -> Union[Union[StrOrOpenAIObject], Sequence[StrOrOpenAIObject], Sequence[Sequence[StrOrOpenAIObject]],]:
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    logging.info(f"Decoding with OpenAI API model {model_name} and numproc == {num_procs}.")
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    # convert prompts to chat format
    is_chat = requires_chatml(model_name)
    is_chat_format = isinstance(prompts[0], dict)
    if is_chat:
        if batch_size > 1:
            logging.warning("batch_size > 1 is not supported yet for chat models. Setting to 1")
            batch_size = 1
        if not is_chat_format:
            prompts = [prompt_to_chatml(prompt) for prompt in prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    shared_kwargs = dict(
        model=model_name,
        **decoding_args.__dict__,
    )
    shared_kwargs.update(decoding_kwargs)  # override default arguments if specified
    with multiprocessing.Pool(num_procs) as p:
        partial_completion_helper = functools.partial(
            _openai_completion_helper,
            sleep_time=sleep_time,
            is_chat=is_chat,
            **shared_kwargs
        )
        completions = list(
            tqdm.tqdm(
                p.imap(partial_completion_helper, prompt_batches),
                desc="prompt_batches",
                total=len(prompt_batches),
            )
        )
    # flatten the list
    completions = [completion for completion_batch in completions for completion in completion_batch]

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def string_to_dict(to_convert):
    """Converts a string with equal signs to dictionary. E.g.
    >>> string_to_dict(" name=user university=stanford")
    {'name': 'user', 'university': 'stanford'}
    """
    return {s.split("=", 1)[0]: s.split("=", 1)[1] for s in to_convert.split(" ") if len(s) > 0}


def prompt_to_chatml(prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"):
    """Convert a text prompt to ChatML formal

    Examples
    --------
    >>> prompt = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>system
    name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\nWho's
    there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
    >>> print(prompt)
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>system name=example_user
    Knock knock.
    <|im_end|>
    <|im_start|>system name=example_assistant
    Who's there?
    <|im_end|>
    <|im_start|>user
    Orange.
    <|im_end|>
    >>> prompt_to_chatml(prompt)
    [{'role': 'system', 'content': 'You are a helpful assistant.'},
     {'role': 'user', 'content': 'Knock knock.'},
     {'role': 'assistant', 'content': "Who's there?"},
     {'role': 'user', 'content': 'Orange.'}]
    """
    prompt = prompt.strip()
    assert prompt.startswith(start_token)
    assert prompt.endswith(end_token)

    message = []
    for p in prompt.split("<|im_start|>")[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()

        if role.startswith("system") and role != "system":
            # based on https://github.com/openai/openai-cookbook/blob/main/examples
            # /How_to_format_inputs_to_ChatGPT_models.ipynb
            # and https://github.com/openai/openai-python/blob/main/chatml.md it seems that system can specify a
            # dictionary of other args
            other_params = string_to_dict(role.split("system", 1)[-1])
            role = "system"
        else:
            other_params = dict()

        message.append(dict(content=content, role=role, **other_params))

    return message


# Keep the private function for backwards compat.
openai_completion = _openai_completion
