"""Backend for OpenRouter API"""

import logging
import os
import time
from typing import List
from funcy import notnone, once, select_values
import openai
from pydantic import BaseModel

from aide.backend.utils import (
    OutputType,
    backoff_create,
)

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

import httpx

@once
def _setup_openrouter_client():
    global _client
    try:
        custom_http_client = httpx.Client(trust_env=False) 
        _client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            max_retries=0,
            http_client=custom_http_client
        )
        logger.info("OpenRouter client initialized with custom httpx.Client (trust_env=False).")
    except Exception as e:
        logger.error(f"Error initializing OpenRouter client: {e}")
        _client = None


def query(
    system_message: str | None,
    user_messages: List | None,
    functions: BaseModel | List | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openrouter_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    if functions is not None:
        raise NotImplementedError(
            "We are not supporting function calling in OpenRouter for now."
        )

    # in case some backends dont support system roles, just convert everything to user
    messages = []
    if system_message:
        messages.append({"role": "user", "content": system_message})
    for message in user_messages:
        if message:
            messages.append({"role": "user", "content": message})

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        extra_body={
            "provider": {
                "order": ["Fireworks"],
                "ignore": ["Together", "DeepInfra", "Hyperbolic"],
            },
        },
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    # output = completion.choices[0].message.content
    output = completion.choices[0].message

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
