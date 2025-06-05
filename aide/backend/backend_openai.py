"""Backend for OpenAI API."""

import json
import logging
import time
import openai

import httpx
import os

from pydantic import BaseModel
from typing import Any, Dict, Type, Union, List

from aide.actions import get_action
from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from funcy import notnone, once, select_values
from langchain_core.utils.function_calling import convert_to_openai_tool

from aide.function import get_function

logger = logging.getLogger("aide")


# Add at top of backend_openai.py
logger.info("OpenAI configuration:", dir(openai))
logger.info("OpenAI version:", openai.__version__)

logger.info("HTTP_PROXY:", os.environ.get("HTTP_PROXY"))
logger.info("HTTPS_PROXY:", os.environ.get("HTTPS_PROXY"))


_client: openai.OpenAI = None  # type: ignore


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openai_client():
    global _client
    
    try:
        custom_http_client = httpx.Client(trust_env=False) 
        _client = openai.OpenAI(http_client=custom_http_client, max_retries=0)
        print("OpenAI client initialized with custom httpx.Client (proxies explicitly set to None).")
        logger.info("OpenAI client initialized with custom httpx.Client (proxies explicitly set to None).")
    except TypeError as e:
        logger.error(f"Error initializing OpenAI client with custom httpx_client: {e}")
        try:
            # Try without parameters
            _client = openai.OpenAI(max_retries=0)
            logger.warning("OpenAI client falling back to default initialization after custom httpx_client failed.")
        except Exception as e2:
            logger.error(f"Failed to initialize OpenAI client: {e2}")
            _client = None


def get_tool_message(tool_call_id: str, name: str, argments: dict) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"arguments": json.dumps(argments), "name": name},
            }
        ],
    }


def get_tool_response_message(tool_call_id: str, content: str | dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content if isinstance(content, str) else json.dumps(content),
    }


def query(
    system_message: str | None,
    user_messages: List | None,
    functions: BaseModel | List | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openai_client()
    if _client is None:
        logger.error("OpenAI client initialization failed")
        return "Error: Could not initialize OpenAI client", 0, 0, 0, {}
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(
        system_message, user_messages, convert_system_to_user=convert_system_to_user
    )

    if functions is not None:
        func_spec = OpenAIFunctionSpec(functions)
        filtered_kwargs["tools"] = func_spec.tool_dict
        filtered_kwargs["tool_choice"] = func_spec.tool_choice_dict

    t0 = time.time()
    logger.info(f"Sending request to OpenAI API with {len(messages)} messages.")
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens
    logger.info(f"in_tokens: {in_tokens}, out_tokens: {out_tokens}")

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
        "stop_reason": choice.finish_reason,
    }
    if functions is None or choice.message.tool_calls is None:
        output = choice.message.content
        return output, req_time, in_tokens, out_tokens, info
    else:
        if len(choice.message.tool_calls) == 0:
            output = choice.message.content
            return output, req_time, in_tokens, out_tokens, info
        else:
            tool_results = []
            for tool_call in choice.message.tool_calls:
                logger.info(f"LLM requested tool call with name: '{tool_call.function.name}'") # DEBUG PRINT
                func = get_function(tool_call.function.name)
                if func is None:
                    action_cls = get_action(tool_call.function.name)
                    if action_cls:
                        action = action_cls(**json.loads(tool_call.function.arguments))
                        action.tool_call_metadata = {
                            "provider": "openai",
                            "tool_call_id": tool_call.id,
                        }
                        return action, req_time, in_tokens, out_tokens, info
                    logger.warning(
                        f"Function {tool_call.function.name} not found, returning arguments."
                    )
                    return (
                        json.loads(tool_call.function.arguments),
                        req_time,
                        in_tokens,
                        out_tokens,
                        info,
                    )
                else:
                    tool_result = func(**json.loads(tool_call.function.arguments))
                    tool_results.append(
                        get_tool_response_message(tool_call.id, tool_result)
                    )

            user_messages = (
                user_messages.append(choice.message)
                if isinstance(user_messages, list)
                else [choice.message]
            )
            user_messages.extend(tool_results)

            next_output, next_req_time, next_in_tokens, next_out_tokens, next_info = (
                query(
                    system_message=system_message,
                    user_messages=user_messages,
                    functions=functions,
                    convert_system_to_user=convert_system_to_user,
                    **model_kwargs,
                )
            )
            # Aggregate metrics
            req_time += next_req_time
            in_tokens += next_in_tokens
            out_tokens += next_out_tokens
            info = {"calls": [info, next_info]}  # Combine metadata

            return next_output, req_time, in_tokens, out_tokens, info


class OpenAIFunctionSpec(FunctionSpec):

    def __init__(self, tools: Union[List, Dict[str, Any], Type]):
        super().__init__(tools)

    @property
    def tool_dict(self):
        if isinstance(self.tools, List):
            return [convert_to_openai_tool(tool) for tool in self.tools]

        return [convert_to_openai_tool(self.tools)]

    @property
    def tool_choice_dict(self):
        if isinstance(self.tools, List):
            return "auto"

        return {
            "type": "function",
            "function": {"name": self.tools.__name__},
        }
