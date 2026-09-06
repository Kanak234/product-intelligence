"""
Ollama provider — the platform's own local language model.

No external API, no key, nothing leaves the machine. The model runs in a
container next to the application (see DOCKER.md stage 5).

Everything here is written to fail softly: if the model server is down, slow,
or returns nonsense, `complete` raises and the engine's `_call_llm` catches it
and continues deterministically. A missing model degrades the output; it never
breaks the pipeline.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from .base import ChatMessage

DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# Extraction is not a creative task. Sampling variance produces schema
# violations, which produce rejections in _merge_llm_specs, which lowers recall
# for no benefit at all.
DEFAULT_OPTIONS: Dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "num_predict": 1024,
}

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class OllamaProvider:
    """Chat completions against a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        default_model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout

    # -- the interface the engine uses ------------------------------------

    async def complete(
        self, messages: List[ChatMessage], model: Optional[str] = None
    ) -> str:
        import httpx  # local import: optional dependency

        payload = {
            "model": model or self.default_model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": DEFAULT_OPTIONS,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            body = resp.json()

        content = (body.get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("model returned an empty completion")
        return content

    async def health(self) -> bool:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def available_models(self) -> List[str]:
        """Model names the server currently has pulled. Empty list on failure."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []


def strip_json_fences(text: str) -> str:
    """
    Remove markdown fences a model adds despite being told not to.

    The system prompt forbids them; smaller models produce them anyway. This is
    cheaper than rejecting an otherwise valid response.
    """
    return _FENCE_RE.sub("", text).strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    """
    Best-effort parse of a model response into a dict.

    Returns `{}` rather than raising: an unparseable response means no proposed
    specifications, which is a normal outcome, not an error.
    """
    cleaned = strip_json_fences(text)
    if not cleaned:
        return {}

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Some models wrap the object in a sentence. Take the outermost braces.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return {}

    return parsed if isinstance(parsed, dict) else {}
