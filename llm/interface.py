"""
llm/interface.py
Thin wrapper around Groq (primary) with Ollama fallback for LLM inference.

If GROQ_API_KEY is set in the environment, requests go to Groq's API using
llama-3.3-70b-versatile. Otherwise falls back to local Ollama qwen3:8b.
"""

from __future__ import annotations

import os
import re

import httpx

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# Groq config
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.3-70b-versatile"

# Ollama fallback config
_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_MODEL = "qwen3:8b"

_TIMEOUT = 60.0


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _query_groq(prompt: str, system: str) -> str:
    headers = {
        "Authorization": f"Bearer {_GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 900,
        "stream": False,
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        response = client.post(_GROQ_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


def _query_ollama(prompt: str, system: str) -> str:
    payload = {
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 900,
            "num_ctx": 8192,
        },
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        response = client.post(_OLLAMA_URL, json=payload)
        response.raise_for_status()
        raw = response.json()["response"]
        return _strip_thinking(raw)


def query_llm(prompt: str, system: str) -> str:
    """Send a prompt to Groq (if GROQ_API_KEY set) or Ollama and return the response text.

    Raises httpx.HTTPError on transport failure or non-2xx status.
    """
    if _GROQ_API_KEY:
        return _query_groq(prompt, system)
    return _query_ollama(prompt, system)
