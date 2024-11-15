"""Backend for Anthropic API."""

import json
import time
import logging
from typing import Any, Dict, Type, Union, List

import anthropic
from pydantic import BaseModel
from .utils import FunctionSpec, OutputType, backoff_create, opt_messages_to_list
from funcy import notnone, once, select_values
from langchain_anthropic.chat_models import convert_to_anthropic_tool

logger = logging.getLogger("aide")

_client: anthropic.Anthropic = None  # type: ignore

ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


@once
def _setup_anthropic_client():
    global _client
    _client = anthropic.Anthropic(max_retries=0)


def query(
    system_message: str | None,
    user_messages: List | None,
    functions: BaseModel | List | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_anthropic_client()

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 4096  # default for Claude models

    if functions is not None:
        func_spec = AnthropicFunctionSpec(functions)
        filtered_kwargs["tools"] = func_spec.tool_dict
        filtered_kwargs["tool_choice"] = func_spec.tool_choice_dict

    # Anthropic doesn't allow not having a user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and (user_messages is None or len(user_messages) == 0):
        user_messages.append(system_message)
        system_message = None
        # system_message, user_message = user_message, system_message

    # Anthropic passes the system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_messages)

    t0 = time.time()
    message = backoff_create(
        _client.messages.create,
        ANTHROPIC_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0
    output = message
    # if function is None: 
    #     assert len(message.content) == 1 and message.content[0].type == "text"
    #     output = message.content[0].text
    # else:
    #     assert (
    #         len(message.content) >= 1 and message.content[0].type == "tool_use"
    #     ), f"function_call is empty, it is not a function call: {message.content}"
        
    #     assert (
    #         message.content[0].name == function.__name__
    #     ), "Function name mismatch"
    #     try:
    #         output = message.content[0].input
    #     except json.JSONDecodeError as e:
    #         logger.error(
    #             f"Error decoding the function arguments: {message.content[0].input}"
    #         )
    #         raise e

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
    }

    return output, req_time, in_tokens, out_tokens, info


class AnthropicFunctionSpec(FunctionSpec):
    def __init__(self, tools: Union[List, Dict[str, Any], Type], ):
        super().__init__(tools)

    @property
    def tool_dict(self):
        if isinstance(self.tools, List):
            return [convert_to_anthropic_tool(tool) for tool in self.tools]
        
        return convert_to_anthropic_tool(self.tools)

    @property
    def tool_choice_dict(self) -> dict:
        if isinstance(self.tools, List):
            return {
                "type": "auto"
            }
        
        return {
            "type": "tool",
            "name": self.tool.__name__
        }