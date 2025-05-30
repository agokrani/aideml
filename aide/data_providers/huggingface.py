"""HuggingFace data provider for datasets from Hugging Face Hub."""

from pathlib import Path
from typing import Dict, Any
import logging

from .base import DataProvider

logger = logging.getLogger("aide")


class HuggingFaceProvider(DataProvider):
    """Provider for HuggingFace datasets."""
    
    def __init__(self, dataset_name: str):
        """
        Initialize HuggingFace data provider.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
        """
        self.dataset_name = dataset_name
    
    def prepare_local_data(self, target_dir: Path, use_symlinks: bool) -> None:
        """Download HuggingFace dataset directly to target directory."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required for HuggingFace provider. "
                "Install with: pip install datasets"
            )
        
        logger.info(f"Downloading HuggingFace dataset '{self.dataset_name}' to {target_dir}")
        
        # Check if dataset already exists in target directory
        if target_dir.exists() and any(target_dir.iterdir()):
            logger.info(f"Dataset already exists at {target_dir}, skipping download")
            # TODO: Add force re-download flag support in future
            return
        
        # Download dataset directly to target directory
        dataset = load_dataset(self.dataset_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(target_dir)
        logger.info(f"Downloaded HuggingFace dataset '{self.dataset_name}' to {target_dir}")
    
    def prepare_modal_data(self, task_id: str) -> str:
        """Download directly to Modal volume."""
        from ..runtime.modal_data_functions import download_huggingface_data
        
        logger.info(f"Downloading HuggingFace dataset '{self.dataset_name}' to Modal volume for task {task_id}")
        
        result = download_huggingface_data.remote(self.dataset_name, task_id)
        logger.info(f"Modal download result: {result}")
        
        return result
    
    def get_modal_requirements(self) -> Dict[str, Any]:
        """Return Modal requirements for HuggingFace provider."""
        return {
            "pip": ["datasets", "transformers"],
            "secrets": ["huggingface-token"]
        }
    
    @property
    def name(self) -> str:
        """Return the provider name."""
        return "huggingface"