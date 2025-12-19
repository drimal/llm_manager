from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
import yaml
import os

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    provider: str
    model_name: str
    params: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None  # For filtering in evals

class ProviderRegistry:
    """Registry for provider configurations and model variants"""
    
    def __init__(self, config_path: str = "models.yaml"):
        self.config_path = Path(config_path)
        self._provider_env_mappings: Dict[str, Dict[str, str]] = {}
        self._models: Dict[str, ModelConfig] = {}
        self.load()
    
    def load(self) -> None:
        """Load provider and model configurations"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self._provider_env_mappings = config.get("providers", {})
        
        # Parse model definitions
        for model_id, model_def in config.get("models", {}).items():
            self._models[model_id] = ModelConfig(
                provider=model_def["provider"],
                model_name=model_def["model_name"],
                params=model_def.get("params"),
                tags=model_def.get("tags", [])
            )
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return self._models.get(model_id)
    
    def get_models_by_tag(self, tag: str) -> List[str]:
        """Get all model IDs with a specific tag"""
        return [
            model_id for model_id, config in self._models.items()
            if config.tags and tag in config.tags
        ]
    
    def list_models(self) -> List[str]:
        """List all available model IDs"""
        return list(self._models.keys())
    
    def configure_for_model(
        self, 
        model_id: str, 
        params: Dict[str, Any]
    ) -> str:
        """
        Configure params dict for the specified model.
        Returns the model name to use.
        """
        model_config = self.get_model_config(model_id)
        if not model_config:
            raise ValueError(f"Unknown model: {model_id}")
        
        # Get provider env mappings
        provider_mapping = self._provider_env_mappings.get(model_config.provider, {})
        
        # Load env vars into params
        for param_key, env_key in provider_mapping.get("env_vars", {}).items():
            value = os.getenv(env_key)
            if value:
                params[param_key] = value
        
        # Apply model-specific param overrides if any
        if model_config.params:
            params.update(model_config.params)
        
        return model_config.model_name