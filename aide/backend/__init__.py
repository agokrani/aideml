import logging
from typing import List
from pydantic import BaseModel

from aide.function import get_function
from . import backend_anthropic, backend_openai, backend_openrouter, backend_gdm
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md

logger = logging.getLogger("aide")


def determine_provider(model: str) -> str:
    if model.startswith("gpt-") or model.startswith("o1-"):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    elif model.startswith("gemini-"):
        return "gdm"
    # all other models are handle by openrouter
    else:
        return "openrouter"


provider_to_query_func = {
    "openai": backend_openai.query,
    "anthropic": backend_anthropic.query,
    "gdm": backend_gdm.query,
    "openrouter": backend_openrouter.query,
}


provider_to_tool_message_func = {
    "openai": backend_openai.get_tool_message,
    "anthropic": backend_anthropic.get_tool_message,
}

provider_to_tool_response_message_func = {
    "openai": backend_openai.get_tool_response_message,
    "anthropic": backend_anthropic.get_tool_response_message,
}

def query(
    system_message: PromptType | None,
    user_messages: List | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    functions: BaseModel | List | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        function (BaseModel | None, optional): Optional BaseModel object defining a function call. If given, the return value will be a dict.
        return_function (bool): If True then return the function call when function is not found. Else return arguments of the function
    Returns:
        OutputType: A string completion if function is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info("---Querying model---", extra={"verbose": True})
    system_message = compile_prompt_to_md(system_message) if system_message else None
    if system_message:
        logger.info(f"system: {system_message}", extra={"verbose": True})
    # Directly use user_messages as is, without compilation
    if user_messages:
        for idx, msg in enumerate(user_messages):
            logger.info(f"user message {idx}: {msg}", extra={"verbose": True})
    if functions is not None:
        if isinstance(functions, list):
            for idx, function in enumerate(functions):
                logger.info(f"function {idx} spec: {function.model_json_schema()}", extra={"verbose": True})
        else:
            logger.info(f"function spec: {functions.model_json_schema()}", extra={"verbose": True})

    provider = determine_provider(model)
    query_func = provider_to_query_func[provider]
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=system_message,
        user_messages=user_messages,
        functions=functions,
        convert_system_to_user=convert_system_to_user,
        **model_kwargs,
    )

    logger.info(f"response: {output}", extra={"verbose": True})
    logger.info(f"---Query complete---", extra={"verbose": True})
    logger.info(f"in_tokens: {in_tok_count}, out_tokens: {out_tok_count}")

    return output
