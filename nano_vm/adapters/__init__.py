# nano_vm/adapters/__init__.py
"""
nano_vm.adapters
================
LiteLLMAdapter depends on the optional `litellm` package
(``pip install llm-nano-vm[litellm]``). It is imported lazily via module
``__getattr__`` (PEP 562) so that a bare ``pip install llm-nano-vm`` and
``import nano_vm`` never fail just because litellm is absent -- only
``LiteLLMAdapter(...)`` itself does, and only at the point of use. Same
soft-dependency principle as telemetry.py::span_step, applied to a class
that is part of the public API instead of an internal no-op fallback.
"""

from .base import LLMAdapter
from .mock_adapter import MockLLMAdapter

__all__ = ["LLMAdapter", "LiteLLMAdapter", "MockLLMAdapter"]


def __getattr__(name: str) -> object:
    if name == "LiteLLMAdapter":
        from .litellm_adapter import LiteLLMAdapter

        return LiteLLMAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
