"""Local data provider for filesystem-based data."""

from pathlib import Path
from typing import Dict, Any
import logging

from .base import DataProvider
from aide.utils import copytree

logger = logging.getLogger("aide")


class LocalDataProvider(DataProvider):
    """Provider for local filesystem data."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize local data provider.
        
        Args:
            data_dir: Path to local data directory
        """
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
    
    def prepare_local_data(self, target_dir: Path, use_symlinks: bool, **kwargs) -> None:
        """Copy/symlink local data using existing copytree."""
        logger.info(f"Preparing local data from {self.data_dir} to {target_dir}")
        copytree(self.data_dir, target_dir, use_symlinks=use_symlinks)

    def prepare_modal_data(self, task_id: str, **kwargs) -> str:
        """Upload local data to Modal volume (existing behavior)."""
        import modal
        from grpclib import GRPCError
        
        logger.info(f"Uploading local data to Modal volume for task {task_id}")
        
        volume = modal.Volume.from_name("agent-volume", create_if_missing=True)
        task_path = f"tasks/{task_id}"
        
        try:
            # Check if task already exists
            dirs = volume.listdir("/tasks")
            existing_tasks = [d.path.split("/")[1] for d in dirs]
            
            if task_id not in existing_tasks:
                with volume.batch_upload() as batch:
                    batch.put_directory(self.data_dir, task_path)
                logger.info(f"Uploaded local data to Modal volume at {task_path}")
            else:
                logger.info(f"Data already exists in Modal volume at {task_path}")
                
        except GRPCError:
            # First upload, tasks directory doesn't exist yet
            with volume.batch_upload() as batch:
                batch.put_directory(self.data_dir, task_path)
            logger.info(f"Created and uploaded local data to Modal volume at {task_path}")
        
        return f"Uploaded local data to tasks/{task_id}"
    
    @property
    def name(self) -> str:
        """Return the provider name."""
        return "local"