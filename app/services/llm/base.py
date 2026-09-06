"""
Message type the enrichment engine expects.

`product_intelligence/enricher.py:_call_llm` does this:

    from app.services.llm.base import ChatMessage

inside a try/except, so that the engine runs standalone when no model layer is
installed. Providing this module is what activates the LLM stage — no change to
the engine is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable


@dataclass
class ChatMessage:
    """One turn in a chat completion request."""

    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@runtime_checkable
class LLMProvider(Protocol):
    """
    The contract a provider must satisfy.

    `complete` is the only method the engine calls. `health` is used by the
    router to pick a live provider and to decide whether to bother trying.
    """

    async def complete(
        self, messages: List[ChatMessage], model: Optional[str] = None
    ) -> str: ...

    async def health(self) -> bool: ...


class LLMUnavailable(RuntimeError):
    """No provider could serve the request. Callers degrade, they do not crash."""
