"""
Provider Abstraction Layer for SAP AI Core LLMs
Module 1.2: Multi-provider abstraction

This module provides a unified interface for working with multiple LLM providers
through SAP AI Core. It supports OpenAI, Anthropic, and other models available
in your SAP AI Core deployments.
"""
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List

warnings.filterwarnings("ignore", category=FutureWarning)

from langchain_core.language_models import BaseLanguageModel
from gen_ai_hub.proxy.langchain import init_llm
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client


class Provider(Enum):
    """Supported LLM providers via SAP AI Core."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    META = "meta"
    MISTRAL = "mistral"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_name: str
    resource_group: str
    provider: Provider
    default_temperature: float = 0.7
    default_max_tokens: int = 1024


# Available models via SAP AI Core
# These should match your deployed models
AVAILABLE_MODELS: Dict[str, ModelConfig] = {
    # OpenAI models (default resource group)
    "gpt-4o": ModelConfig(
        model_name="gpt-4o",
        resource_group="default",
        provider=Provider.OPENAI,
    ),
    "gpt-5": ModelConfig(
        model_name="gpt-5",
        resource_group="default",
        provider=Provider.OPENAI,
    ),

    # Anthropic models (anthropic resource group)
    "claude-4-sonnet": ModelConfig(
        model_name="anthropic--claude-4-sonnet",
        resource_group="anthropic",
        provider=Provider.ANTHROPIC,
    ),
    "claude-4.5-sonnet": ModelConfig(
        model_name="anthropic--claude-4.5-sonnet",
        resource_group="anthropic",
        provider=Provider.ANTHROPIC,
    ),
    "claude-4.5-opus": ModelConfig(
        model_name="anthropic--claude-4.5-opus",
        resource_group="anthropic",
        provider=Provider.ANTHROPIC,
    ),

    # Google models (default resource group)
    "gemini-2.0-flash": ModelConfig(
        model_name="gemini-2.0-flash",
        resource_group="default",
        provider=Provider.GOOGLE,
    ),
}


class LLMProvider:
    """
    Unified LLM provider interface for SAP AI Core.

    This class abstracts away the complexity of working with different
    LLM providers through SAP AI Core, providing a simple interface
    for model initialization and configuration.

    Example usage:
        provider = LLMProvider()

        # Get a specific model
        gpt4 = provider.get_model("gpt-4o")
        response = gpt4.invoke("Hello!")

        # Get Anthropic model
        claude = provider.get_model("claude-4-sonnet")
        response = claude.invoke("Hello!")
    """

    def __init__(self):
        """Initialize the LLM provider."""
        self._model_cache: Dict[str, BaseLanguageModel] = {}

    def get_model(
        self,
        model_key: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> BaseLanguageModel:
        """
        Get an LLM model by key.

        Args:
            model_key: Key from AVAILABLE_MODELS (e.g., "gpt-4o", "claude-4-sonnet")
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional model parameters

        Returns:
            BaseLanguageModel: Initialized LangChain model

        Raises:
            ValueError: If model_key is not found
        """
        if model_key not in AVAILABLE_MODELS:
            available = ", ".join(AVAILABLE_MODELS.keys())
            raise ValueError(
                f"Model '{model_key}' not found. Available models: {available}"
            )

        config = AVAILABLE_MODELS[model_key]
        cache_key = f"{model_key}_{temperature}_{max_tokens}"

        if cache_key not in self._model_cache:
            proxy_client = get_proxy_client(
                "gen-ai-hub",
                resource_group=config.resource_group
            )

            model = init_llm(
                config.model_name,
                proxy_client=proxy_client,
                temperature=temperature or config.default_temperature,
                max_tokens=max_tokens or config.default_max_tokens,
                **kwargs
            )
            self._model_cache[cache_key] = model

        return self._model_cache[cache_key]

    def list_available_models(self) -> List[str]:
        """List all available model keys."""
        return list(AVAILABLE_MODELS.keys())

    def get_model_config(self, model_key: str) -> ModelConfig:
        """Get configuration for a model."""
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_key}' not found")
        return AVAILABLE_MODELS[model_key]

    def get_models_by_provider(self, provider: Provider) -> List[str]:
        """Get all models for a specific provider."""
        return [
            key for key, config in AVAILABLE_MODELS.items()
            if config.provider == provider
        ]

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._model_cache.clear()


def get_llm(
    model_key: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    **kwargs: Any
) -> BaseLanguageModel:
    """
    Quick function to get an LLM model.

    Args:
        model_key: Model to use (default: "gpt-4o")
        temperature: Model temperature (default: 0.7)
        max_tokens: Max tokens (default: 1024)
        **kwargs: Additional parameters

    Returns:
        BaseLanguageModel: Initialized model
    """
    provider = LLMProvider()
    return provider.get_model(
        model_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
