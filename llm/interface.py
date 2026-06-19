"""
llm/interface.py
Thin wrapper around the Ollama HTTP API for qwen3:8b inference.
"""

from __future__ import annotations

import re

import httpx

_OLLAMA_URL = "http://localhost:11434/api/generate"
_MODEL = "qwen3:8b"
_TIMEOUT = 600.0

# Ollama 0.30.6 does not reliably suppress qwen3 thinking blocks via think:false.
# Strip any <think>...</think> content from the response before returning it.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def query_llm(prompt: str, system: str) -> str:
    """Send a prompt to Ollama and return the response text.

    Raises httpx.HTTPError on transport failure or non-2xx status.
    """
    payload = {
        "model": _MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 900,
            "num_ctx": 8192,  # analytics payload (3500) + history (~300) + system (~200) + headroom
        },
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        response = client.post(_OLLAMA_URL, json=payload)
        response.raise_for_status()
        raw = response.json()["response"]
        return _strip_thinking(raw)
