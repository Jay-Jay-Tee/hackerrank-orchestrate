import json
import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env file (if present)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")


def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or your environment before running the agent."
        )

    client_kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL
    client_kwargs["http_client"] = httpx.Client(timeout=60.0, trust_env=False)
    return OpenAI(**client_kwargs)

def generate_json(prompt: str) -> dict:
    """Calls OpenAI to generate a JSON response."""
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return only a valid JSON object."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        result_text = response.choices[0].message.content or "{}"
        return json.loads(result_text)
    except Exception as e:
        print(f"Error calling OpenAI JSON API: {e}")
        return {
            "internal_request_type": "other",
            "product_area": "unknown",
            "decision": "escalate",
            "confidence": 0.0
        }

def generate_text(prompt: str) -> str:
    """Calls OpenAI to generate standard text."""
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=280,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"Error calling OpenAI Text API: {e}")
        return "I'm currently unable to access my knowledge base. Please contact a human agent for further assistance."
