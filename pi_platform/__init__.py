"""
Platform services around the enrichment engine.

Each is optional and each degrades to a no-op when unconfigured, which is why
the same codebase runs as a single stateless Streamlit process on the cloud and
as a full stack under Docker Compose.

Named `pi_platform` rather than `platform` deliberately: the latter would
shadow the standard library module of that name.
"""

from .config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
