"""Lightweight package initializer for llm_manager.

This module avoids importing heavy submodules at import time to keep
test startup fast and to prevent import errors when optional dependencies
are not installed.
"""

__version__ = "0.1.0"

# Public symbols are intentionally minimal here; import submodules explicitly
# (for example: `from llm_manager.factory import LLMFactory`) to avoid
# triggering provider library imports during package import.

__all__ = ["__version__"]
