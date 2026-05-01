"""
llm.py

Handles LLM API interactions, supporting OpenAI and local models.
Provides robust JSON and text generation wrappers with retry logic and rate limit handling.
Uses environment variables per AGENTS.md requirements.
"""
import json
import os
import re
import time
from typing import Dict, List

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
# GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()

LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "").strip()
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3").strip()

_HTTP_CLIENT = None
_OPENAI_CLIENT = None
_CACHED_PROVIDER = ""


def _resolve_provider() -> tuple[str, str, str, str]:
    # if GEMINI_API_KEY:
    #     return "gemini", GEMINI_API_KEY, GEMINI_MODEL, GEMINI_BASE_URL
    if OPENAI_API_KEY:
        return "openai", OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
    if LOCAL_BASE_URL:
        return "local", "local-dummy-key", LOCAL_MODEL, LOCAL_BASE_URL
    return "", "", "", ""


def _get_client() -> OpenAI:
    global _HTTP_CLIENT, _OPENAI_CLIENT, _CACHED_PROVIDER

    provider, api_key, _model, base_url = _resolve_provider()
    if not provider:
        raise RuntimeError("No LLM API key or local URL configured. Set OPENAI_API_KEY or LOCAL_BASE_URL.")

    if _OPENAI_CLIENT is not None and _CACHED_PROVIDER == provider:
        return _OPENAI_CLIENT

    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.Client(timeout=60.0, trust_env=True)

    kwargs = {"api_key": api_key, "http_client": _HTTP_CLIENT}
    if base_url:
        kwargs["base_url"] = base_url

    _OPENAI_CLIENT = OpenAI(**kwargs)
    _CACHED_PROVIDER = provider
    return _OPENAI_CLIENT


def _chat_completion(messages: List[Dict[str, str]], max_tokens: int = 300, json_mode: bool = False) -> str:
    client = _get_client()
    provider, _api_key, model_name, _base_url = _resolve_provider()

    kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    kwargs["max_completion_tokens"] = max_tokens

    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            try:
                resp = client.chat.completions.create(**kwargs)
            except TypeError:
                kwargs.pop("max_completion_tokens", None)
                kwargs["max_tokens"] = max_tokens
                resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            msg = str(exc)
            
            if "temperature" in msg.lower() and "unsupported" in msg.lower():
                kwargs.pop("temperature", None)
                continue
                
            wait_s = 0.0
            m = re.search(r"try again in\s+([0-9.]+)s", msg.lower())
            if m:
                wait_s = float(m.group(1))
            if "rate_limit" in msg.lower() or "429" in msg:
                wait_s = max(wait_s, 2.5 * attempt)
            if attempt == max_attempts:
                raise
            time.sleep(wait_s if wait_s > 0 else float(attempt))

    return ""


def generate_json(prompt: str, max_tokens: int = 240) -> dict:
    try:
        text = _chat_completion(
            messages=[
                {"role": "system", "content": "Return only valid JSON. No markdown."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            json_mode=True,
        )
        if not text:
            return {}
            
        # Clean up markdown code blocks if the LLM hallucinated them
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            
        return json.loads(text)
    except Exception as exc:
        print(f"Error calling JSON generation: {exc}")
        return {}


def generate_text(prompt: str, max_tokens: int = 320) -> str:
    try:
        return _chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, json_mode=False)
    except Exception as exc:
        print(f"Error calling text generation: {exc}")
        return ""
