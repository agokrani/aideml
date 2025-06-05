"""HuggingFace data provider for datasets from Hugging Face Hub."""

from pathlib import Path
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

    def prepare_local_data(self, target_dir: Path, use_symlinks: bool, dataset_kwargs: dict | None = None, **kwargs) -> None:
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
        target_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            dataset = load_dataset(self.dataset_name, **dataset_kwargs)
            dataset.save_to_disk(target_dir)
            logger.info(f"Downloaded HuggingFace dataset '{self.dataset_name}' to {target_dir}")
        except ValueError as e:
            if "Invalid pattern" in str(e) and "**" in str(e):
                # Fallback to snapshot_download for Git repositories
                logger.info("Dataset appears to be a Git repository, using snapshot_download fallback")
                try:
                    from huggingface_hub import snapshot_download
                except ImportError:
                    raise ImportError(
                        "huggingface_hub library is required for Git repository downloads. "
                        "Install with: pip install huggingface_hub"
                    )
                
                snapshot_download(
                    repo_id=self.dataset_name,
                    repo_type="dataset",
                    local_dir=target_dir,
                    **dataset_kwargs
                )
                logger.info(f"Downloaded HuggingFace repository '{self.dataset_name}' to {target_dir}")
            else:
                raise  # Re-raise if it's a different ValueError

    def prepare_modal_data(self, task_id: str, dataset_kwargs: dict | None = None, **kwargs) -> str:
        """Download directly to Modal volume."""
        from ..runtime.modal_data_functions import download_huggingface_data, app, volume
        
        # Check if task directory exists and is not empty
        try:
            task_contents = volume.listdir(f"/tasks/{task_id}")
            if task_contents:  # Directory exists and has contents
                logger.info(f"Dataset already exists for task {task_id}, skipping download")
                return f"Dataset {self.dataset_name} already exists at tasks/{task_id}"
        except Exception:
            # Directory doesn't exist, proceed with download
            logger.info(f"Downloading HuggingFace dataset '{self.dataset_name}' to Modal volume for task {task_id}")
            with app.run():
                result = download_huggingface_data.remote(self.dataset_name, task_id, dataset_kwargs)
            logger.info(f"Modal download result: {result}")
        
            return result
    
    @property
    def name(self) -> str:
        """Return the provider name."""
        return "huggingface"