from .base import LLMAdapter

__all__ = ["LLMAdapter"]

try:
    from .litellm_adapter import LiteLLMAdapter

    __all__.append("LiteLLMAdapter")
except ImportError:
    pass
