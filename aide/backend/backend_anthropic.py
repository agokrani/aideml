"""Backend for Anthropic API."""

import json
import time
import logging
from typing import Any, Dict, Type, Union, List

import anthropic
from pydantic import BaseModel

from aide.actions import get_action
from aide.function import get_function
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

def get_tool_message(tool_call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use", 
                "id": tool_call_id, 
                "name": name, 
                "input": arguments
            }
        ]
    }

def get_tool_response_message(tool_call_id: str, content: str|dict) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": content if isinstance(content, str) else json.dumps(content)
            }
        ]
    }

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
        user_messages = user_messages.append(system_message) if isinstance(user_messages, list) else [system_message]
        system_message = None

    # Anthropic passes the system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_messages)

    t0 = time.time()
    logger.info(f"Sending request to Anthropic API with {len(messages)} messages.")
    
    message = backoff_create(
        _client.messages.create,
        ANTHROPIC_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0
    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
    } 
    if functions is None or all(content.type != "tool_use" for content in message.content):
        output = "\n\n".join(content.text for content in message.content)
        return output, req_time, in_tokens, out_tokens, info
    else: 
        tool_results, assistant_response = [], []
        for content in message.content:
            if content.type == "text": 
                assistant_response.insert(0, {"type": "text", "text": content.text})
            if content.type == "tool_use":
                func = get_function(content.name)
                if func is None:
                    action_cls = get_action(content.name)
                    if action_cls: 
                        action = action_cls(**content.input)
                        action.tool_call_metadata = {
                            "provider": "anthropic",
                            "tool_call_id": content.id
                        }
                        return action, req_time, in_tokens, out_tokens, info
                    logger.warning(f"Function {content.name} not found, returning arguments.")
                    return content.input, req_time, in_tokens, out_tokens, info
                else:
                    tool_result = func(**content.input)
                    assistant_response.append({"type": "tool_use", "id": content.id, "name": content.name, "input": content.input})
                    tool_results.append({"type": "tool_result", "tool_use_id": content.id, "content": tool_result})
        
        if isinstance(user_messages, list):
            user_messages.append({"role": "assistant", "content": assistant_response})
        else:
            user_messages = [{"role": "assistant", "content": assistant_response}]
        
        user_messages.append({"role": "user", "content": tool_results})        

        next_output, next_req_time, next_in_tokens, next_out_tokens, next_info = query(
            system_message=system_message,
            user_messages=user_messages,
            functions=functions,
            convert_system_to_user=convert_system_to_user,
            **model_kwargs,
        )
        
        # Aggregate metrics
        req_time += next_req_time
        in_tokens += next_in_tokens
        out_tokens += next_out_tokens
        info = {
            "calls": [info, next_info]  # Combine metadata
        }
        
        return next_output, req_time, in_tokens, out_tokens, info

class AnthropicFunctionSpec(FunctionSpec):
    def __init__(self, tools: Union[List, Dict[str, Any], Type], ):
        super().__init__(tools)

    @property
    def tool_dict(self):
        if isinstance(self.tools, List):
            return [convert_to_anthropic_tool(tool) for tool in self.tools]
        
        return [convert_to_anthropic_tool(self.tools)]

    @property
    def tool_choice_dict(self) -> dict:
        if isinstance(self.tools, List):
            return {
                "type": "auto",
            }
        
        return {
            "type": "tool",
            "name": self.tools.__name__
        }