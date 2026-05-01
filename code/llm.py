import os
import json
import requests
from dotenv import load_dotenv

# Load variables from .env file (if present)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

def generate_json(prompt: str) -> dict:
    """Calls local Ollama to generate a JSON response."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 42
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120) # 2 min timeout for local
        response.raise_for_status()
        data = response.json()
        result_text = data.get("response", "")
        return json.loads(result_text)
    except Exception as e:
        print(f"Error calling Ollama JSON API: {e}")
        return {
            "internal_request_type": "other",
            "product_area": "unknown",
            "decision": "escalate",
            "confidence": 0.0
        }

def generate_text(prompt: str) -> str:
    """Calls local Ollama to generate standard text."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 42
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"Error calling Ollama Text API: {e}")
        return "I'm currently unable to access my knowledge base. Please contact a human agent for further assistance."
