"""Base class for data providers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    def prepare_local_data(self, target_dir: Path, use_symlinks: bool) -> None:
        """
        Prepare data for local execution.
        
        Args:
            target_dir: Directory to prepare data in
            use_symlinks: Whether to use symlinks instead of copying
        """
        pass
    
    @abstractmethod  
    def prepare_modal_data(self, task_id: str) -> None:
        """
        Download data directly to Modal volume at tasks/{task_id}.
        
        Args:
            task_id: Task ID for Modal volume path
        """
        pass
    
    @abstractmethod
    def get_modal_requirements(self) -> Dict[str, Any]:
        """
        Return Modal requirements for this provider.
        
        Returns:
            Dict with pip packages, secrets, etc. needed in Modal
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass