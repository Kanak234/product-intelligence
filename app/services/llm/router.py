"""
Router in front of the local model.

`ProductEnricher._call_llm` calls two things on this object:

    provider_name = await router.auto_select_provider()   # if not pinned
    provider      = router.get_provider(provider_name)

`evaluation.py` constructs it with no arguments, so the default constructor
must produce something usable from the environment alone.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from .base import LLMUnavailable
from .ollama_provider import OllamaProvider


class LLMRouter:
    """
    Holds the available providers and picks a healthy one.

    Health is checked, not assumed. A router that hands back a dead provider
    turns every enrichment into a timeout; checking first means the LLM stage
    is skipped quickly and the pipeline stays fast when the model is down.
    """

    def __init__(self, providers: Optional[Dict[str, object]] = None) -> None:
        if providers is None:
            providers = {"ollama": OllamaProvider()}
        if not providers:
            raise ValueError("LLMRouter needs at least one provider")
        self.providers: Dict[str, object] = dict(providers)
        self._default = next(iter(self.providers))

    # -- interface used by the engine -------------------------------------

    def get_provider(self, name: Optional[str] = None):
        if name is None:
            name = self._default
        try:
            return self.providers[name]
        except KeyError as exc:
            raise LLMUnavailable(f"no provider named {name!r}") from exc

    async def auto_select_provider(self) -> str:
        for name, provider in self.providers.items():
            health = getattr(provider, "health", None)
            if health is None:
                return name
            try:
                if await health():
                    return name
            except Exception:
                continue
        raise LLMUnavailable("no healthy LLM provider")

    # -- convenience -------------------------------------------------------

    def provider_names(self) -> List[str]:
        return list(self.providers)

    async def any_healthy(self) -> bool:
        try:
            await self.auto_select_provider()
            return True
        except LLMUnavailable:
            return False


def build_default_router() -> Optional[LLMRouter]:
    """
    Router from environment, or None when the model is switched off.

    Set `ENABLE_LLM=0` to force deterministic-only mode even with a model
    server running — useful for reproducing a result exactly.
    """
    if os.getenv("ENABLE_LLM", "1").lower() in ("0", "false", "no"):
        return None
    try:
        return LLMRouter()
    except Exception:
        return None
