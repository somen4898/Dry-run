"""LLM factory — creates the right LLMPort adapter based on config."""

from __future__ import annotations
from dryrun.config import ModelConfig
from dryrun.domain.ports.llm import LLMPort


def create_llm(config: ModelConfig, purpose: str = "synthetic_user") -> LLMPort:
    """Create an LLMPort implementation based on the configured provider.

    Args:
        config: The models config section from dryrun.yaml
        purpose: Which model to use — "synthetic_user" or "agent"
    """
    model_name = getattr(config, purpose, config.synthetic_user)

    if config.provider == "anthropic":
        from dryrun.adapters.outbound.anthropic.llm import AnthropicClient

        return AnthropicClient(model=model_name)
    elif config.provider == "openai":
        from dryrun.adapters.outbound.openai.llm import OpenAIClient

        return OpenAIClient(model=model_name)
    else:
        raise ValueError(f"Unknown LLM provider: '{config.provider}'. Use 'openai' or 'anthropic'.")
