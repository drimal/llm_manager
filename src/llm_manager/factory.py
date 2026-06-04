from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .exceptions import UnknownProviderError

if TYPE_CHECKING:  # pragma: no cover
    from .base import BaseLLMClient

# Maps provider name -> (module, class). Imported lazily so creating a client for
# one provider never imports another provider's SDK.
_PROVIDERS = {
    "openai": ("openai_client", "OpenAIClient"),
    "anthropic": ("anthropic_client", "AnthropicClient"),
    "bedrock": ("bedrock_client", "BedrockClient"),
    "ollama": ("ollama_client", "OllamaClient"),
    "gemini": ("gemini_client", "GeminiClient"),
}


class LLMFactory:
    """Factory for constructing provider-specific LLM clients."""

    @staticmethod
    def available_providers() -> list[str]:
        """Return the list of supported provider names."""
        return sorted(_PROVIDERS)

    @staticmethod
    def get_client(provider_name: str, **kwargs: Any) -> BaseLLMClient:
        """Create an LLM client for ``provider_name``.

        Provider implementations are imported lazily to avoid importing heavy
        third-party SDKs that aren't needed.

        Args:
            provider_name: One of :meth:`available_providers`.
            **kwargs: Constructor arguments for the chosen provider client
                (e.g. ``api_key``, ``base_url``, ``system_prompt``).

        Raises:
            UnknownProviderError: If ``provider_name`` is not supported.
        """
        key = provider_name.lower()
        entry = _PROVIDERS.get(key)
        if entry is None:
            raise UnknownProviderError(
                f"Provider '{provider_name}' is not supported. "
                f"Available providers: {', '.join(LLMFactory.available_providers())}"
            )
        import importlib

        module_name, class_name = entry
        module = importlib.import_module(f".providers.{module_name}", __package__)
        client_cls = getattr(module, class_name)
        return client_cls(**kwargs)

    @staticmethod
    def from_model_id(
        model_id: str,
        config_path: str | Path,
        system_prompt: str | None = None,
        **overrides: Any,
    ) -> BaseLLMClient:
        """Create a client for a registry-defined model id.

        The registry (a YAML file) maps a friendly ``model_id`` to a provider,
        a concrete model name, and the environment variables that supply that
        provider's credentials. The returned client uses the resolved model name
        as its default, so you can call ``client.generate(prompt)`` without
        repeating the model.

        Args:
            model_id: A model id defined under ``models:`` in the config file.
            config_path: Path to the registry YAML file.
            system_prompt: Optional system prompt for the client.
            **overrides: Constructor overrides (e.g. an explicit ``api_key``)
                that take precedence over values resolved from the environment.

        Raises:
            UnknownProviderError: If ``model_id`` is not defined in the registry.
        """
        from .providers.provider_registry import ProviderRegistry

        registry = ProviderRegistry(config_path)
        model_config = registry.get_model_config(model_id)
        if model_config is None:
            raise UnknownProviderError(
                f"Unknown model id '{model_id}'. "
                f"Known model ids: {', '.join(registry.list_models())}"
            )

        ctor_params: dict[str, Any] = {}
        model_name = registry.configure_for_model(model_id, ctor_params)

        # model_config.params are generation defaults, not constructor args;
        # drop them so they aren't passed to the client constructor.
        for gen_key in (model_config.params or {}):
            ctor_params.pop(gen_key, None)

        ctor_params.update(overrides)
        if system_prompt is not None:
            ctor_params["system_prompt"] = system_prompt

        client = LLMFactory.get_client(provider_name=model_config.provider, **ctor_params)
        client.default_model = model_name
        return client
