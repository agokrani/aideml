"""Backend for OpenAI API."""

import json
import logging
import time
import openai

from pydantic import BaseModel
from typing import (
    Any, 
    Dict, 
    Type, 
    Union, 
    List
)

from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from funcy import notnone, once, select_values
from langchain_core.utils.function_calling import convert_to_openai_tool




logger = logging.getLogger("aide")

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
    _client = openai.OpenAI(max_retries=0)


def query(
    system_message: str | None,
    user_messages: List | None,
    functions: BaseModel | List | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openai_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_messages, convert_system_to_user=convert_system_to_user)

    if functions is not None:
        func_spec = OpenAIFunctionSpec(functions)
        filtered_kwargs["tools"] = func_spec.tool_dict
        filtered_kwargs["tool_choice"] = func_spec.tool_choice_dict
        
        # Disable parallel tool calls 
        # TODO: To allow it in funture
        filtered_kwargs["parallel_tool_calls"] = False

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]
    
    output = choice.message
    # if functions is None:
    #     output = choice.message.content
    # else:
    #     if len(choice.message.tool_calls) == 0:
    #         output = choice.message.content
    #     else:
    #         if not isinstance(functions, List):
    #             assert (
    #                 choice.message.tool_calls[0].function.name == functions.__name__
    #             ), "Function name mismatch"
    #         try:
    #             import pdb;pdb.set_trace()
    #             output = json.loads(choice.message.tool_calls[0].function.arguments)
    #         except json.JSONDecodeError as e:
    #             logger.error(
    #                 f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
    #             )
    #             raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info


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
            "function": {"name": self.tool.__name__},
        }