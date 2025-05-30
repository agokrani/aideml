"""Kaggle data provider for datasets from Kaggle."""

from pathlib import Path
from typing import Dict, Any
import logging

from .base import DataProvider

logger = logging.getLogger("aide")


class KaggleProvider(DataProvider):
    """Provider for Kaggle datasets."""
    
    def __init__(self, dataset_name: str):
        """
        Initialize Kaggle data provider.
        
        Args:
            dataset_name: Name of the dataset on Kaggle (format: username/dataset-name)
        """
        self.dataset_name = dataset_name
    
    def prepare_local_data(self, target_dir: Path, use_symlinks: bool) -> None:
        """Download Kaggle dataset directly to target directory."""
        try:
            import kaggle
        except ImportError:
            raise ImportError(
                "kaggle library is required for Kaggle provider. "
                "Install with: pip install kaggle"
            )
        
        logger.info(f"Downloading Kaggle dataset '{self.dataset_name}' to {target_dir}")
        
        # Check if dataset already exists in target directory
        if target_dir.exists() and any(target_dir.iterdir()):
            logger.info(f"Dataset already exists at {target_dir}, skipping download")
            # TODO: Add force re-download flag support in future
            return
        
        # Download dataset directly to target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(self.dataset_name, path=target_dir, unzip=True)
        logger.info(f"Downloaded Kaggle dataset '{self.dataset_name}' to {target_dir}")
    
    def prepare_modal_data(self, task_id: str) -> str:
        """Download directly to Modal volume."""
        from ..runtime.modal_data_functions import download_kaggle_data
        
        logger.info(f"Downloading Kaggle dataset '{self.dataset_name}' to Modal volume for task {task_id}")
        
        result = download_kaggle_data.remote(self.dataset_name, task_id)
        logger.info(f"Modal download result: {result}")
        
        return result
    
    def get_modal_requirements(self) -> Dict[str, Any]:
        """Return Modal requirements for Kaggle provider."""
        return {
            "pip": ["kaggle"],
            "secrets": ["kaggle-creds"]
        }
    
    @property
    def name(self) -> str:
        """Return the provider name."""
        return "kaggle"