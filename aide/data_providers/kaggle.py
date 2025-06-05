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
    
    def prepare_local_data(self, target_dir: Path, use_symlinks: bool, **kwargs) -> None:
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
    
    def prepare_modal_data(self, task_id: str, **kwargs) -> str:
        """Download directly to Modal volume."""
        from ..runtime.modal_data_functions import download_kaggle_data, app, volume
        try:
            task_contents = volume.listdir(f"/tasks/{task_id}")
            if task_contents:  # Directory exists and has contents
                logger.info(f"Dataset already exists for task {task_id}, skipping download")
                return f"Dataset {self.dataset_name} already exists at tasks/{task_id}"
        except Exception:
            # Directory doesn't exist, proceed with download
            logger.info(f"Downloading Kaggle dataset '{self.dataset_name}' to Modal volume for task {task_id}")
            with app.run():
                result = download_kaggle_data.remote(self.dataset_name, task_id)
            logger.info(f"Modal download result: {result}")
        
            return result
    
    @property
    def name(self) -> str:
        """Return the provider name."""
        return "kaggle"