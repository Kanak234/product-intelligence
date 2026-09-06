"""Local language model layer. Optional: the engine runs without it."""

from .base import ChatMessage, LLMProvider, LLMUnavailable
from .ollama_provider import OllamaProvider, parse_json_object, strip_json_fences
from .router import LLMRouter, build_default_router

__all__ = [
    "ChatMessage", "LLMProvider", "LLMUnavailable",
    "OllamaProvider", "parse_json_object", "strip_json_fences",
    "LLMRouter", "build_default_router",
]
