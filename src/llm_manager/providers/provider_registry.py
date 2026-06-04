import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Configuration for a specific model"""

    provider: str
    model_name: str
    params: dict[str, Any] | None = None
    tags: list[str] | None = None  # For filtering in evals


class ProviderRegistry:
    """Registry for provider configurations and model variants"""

    def __init__(self, config_path: str | Path = "models.yaml"):
        self.config_path = Path(config_path)
        self._provider_env_mappings: dict[str, dict[str, Any]] = {}
        self._models: dict[str, ModelConfig] = {}
        self.load()

    def load(self) -> None:
        """Load provider and model configurations"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        self._provider_env_mappings = config.get("providers", {})

        # Parse model definitions
        for model_id, model_def in config.get("models", {}).items():
            self._models[model_id] = ModelConfig(
                provider=model_def["provider"],
                model_name=model_def["model_name"],
                params=model_def.get("params"),
                tags=model_def.get("tags", []),
            )

    def get_model_config(self, model_id: str) -> ModelConfig | None:
        """Get configuration for a specific model"""
        return self._models.get(model_id)

    def get_models_by_tag(self, tag: str) -> list[str]:
        """Get all model IDs with a specific tag"""
        return [
            model_id
            for model_id, config in self._models.items()
            if config.tags and tag in config.tags
        ]

    def list_models(self) -> list[str]:
        """List all available model IDs"""
        return list(self._models.keys())

    def configure_for_model(self, model_id: str, params: dict[str, Any]) -> str:
        """
        Configure params dict for the specified model.
        Returns the model name to use.
        """
        model_config = self.get_model_config(model_id)
        if not model_config:
            raise ValueError(f"Unknown model: {model_id}")

        # Get provider env mappings
        provider_mapping = self._provider_env_mappings.get(model_config.provider, {})
        env_vars: dict[str, str] = provider_mapping.get("env_vars", {})

        # Load env vars into params
        for param_key, env_key in env_vars.items():
            value = os.getenv(env_key)
            if value:
                params[param_key] = value

        # Apply model-specific param overrides if any
        if model_config.params:
            params.update(model_config.params)

        return model_config.model_name
