import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type, Union, List

import jsonschema
from dataclasses_json import DataClassJsonMixin

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType

import backoff

logger = logging.getLogger("aide")


@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        logger.info(f"Backoff exception: {e}")
        return False


import backoff
import logging
from typing import Callable

logger = logging.getLogger("aide")


@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        logger.info(f"Backoff exception: {e}")
        return False


def opt_messages_to_list(
    system_message: str | None,
    user_messages: List | None,
    convert_system_to_user: bool = False,
) -> list[dict[str, str]]:
    messages = []
    if system_message:
        if convert_system_to_user:
            messages.append({"role": "user", "content": system_message})
        else:
            messages.append({"role": "system", "content": system_message})
    if user_messages is None:
        user_messages = []
    else:
        user_messages = [
            (
                {"role": "user", "content": message}
                if isinstance(message, str)
                else message
            )
            for message in user_messages
        ]
    messages.extend(user_messages)

    return messages


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif isinstance(prompt, list):
        return "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        out.append(f"{header_prefix} {k}\n")
        out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    tools: Union[
        List, Dict[str, Any], Type
    ]  # Pydantic Object to be converted to OpenAI tool dict or Anthropic tool class

    def __init__(self, tools: Union[List, Dict[str, Any], Type]):
        self.tools = tools

    @property
    def tool_dict(self):
        pass

    @property
    def tool_choice_dict(self):
        pass
