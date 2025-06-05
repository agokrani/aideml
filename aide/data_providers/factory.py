"""Factory function for creating data providers."""

from typing import TYPE_CHECKING

from .base import DataProvider
from .local import LocalDataProvider
from .huggingface import HuggingFaceProvider
from .kaggle import KaggleProvider

if TYPE_CHECKING:
    from ..utils.config import Config


def create_data_provider(cfg: "Config") -> DataProvider:
    """
    Create appropriate data provider based on config.
    
    Args:
        cfg: Configuration object
        
    Returns:
        DataProvider instance
        
    Raises:
        ValueError: If no valid data source is specified
    """
    # New format with provider specification
    if hasattr(cfg, 'data') and cfg.data:
        if cfg.data.provider == "local":
            if not cfg.data.path:
                raise ValueError("Local provider requires 'path' field")
            return LocalDataProvider(cfg.data.path)
        elif cfg.data.provider == "huggingface":
            return HuggingFaceProvider(cfg.data.dataset)
        elif cfg.data.provider == "kaggle":
            return KaggleProvider(cfg.data.dataset)
        else:
            raise ValueError(f"Unknown provider: {cfg.data.provider}")
    
    # Legacy format - local provider
    if cfg.data_dir:
        return LocalDataProvider(cfg.data_dir)
    
    raise ValueError("No valid data source specified. Use either 'data_dir' or 'data' config.")