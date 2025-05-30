from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import modal

from . import copytree


class DataProvider(ABC):
    """Abstract base class for different data sources."""

    @abstractmethod
    def prepare_local_data(self, target_dir: Path, use_symlinks: bool) -> None:
        """Prepare data for local execution."""

    @abstractmethod
    def prepare_modal_data(self, task_id: str) -> Any:
        """Download data directly to Modal volume."""

    @abstractmethod
    def get_modal_requirements(self) -> Dict[str, Any]:
        """Return pip packages, secrets, etc. needed in Modal."""


class HuggingFaceProvider(DataProvider):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def _get_local_cache_dir(self) -> Path:
        cache_root = Path.home() / ".aide" / "data_cache" / "huggingface"
        return cache_root / self.dataset_name.replace("/", "_")

    def prepare_local_data(self, target_dir: Path, use_symlinks: bool) -> None:
        from datasets import load_dataset

        cache_dir = self._get_local_cache_dir()
        if not cache_dir.exists():
            dataset = load_dataset(self.dataset_name)
            cache_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(cache_dir)

        target_dir.mkdir(parents=True, exist_ok=True)
        copytree(cache_dir, target_dir, use_symlinks=use_symlinks)

    def prepare_modal_data(self, task_id: str):
        from aide.runtime.modal_data_functions import download_huggingface_data

        return download_huggingface_data.remote(self.dataset_name, task_id)

    def get_modal_requirements(self) -> Dict[str, Any]:
        return {
            "pip": ["datasets", "transformers"],
            "secrets": ["huggingface-token"],
        }


class KaggleProvider(DataProvider):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def _get_local_cache_dir(self) -> Path:
        cache_root = Path.home() / ".aide" / "data_cache" / "kaggle"
        return cache_root / self.dataset_name.replace("/", "_")

    def prepare_local_data(self, target_dir: Path, use_symlinks: bool) -> None:
        import kaggle

        cache_dir = self._get_local_cache_dir()
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            kaggle.api.dataset_download_files(
                self.dataset_name, path=str(cache_dir), unzip=True
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        copytree(cache_dir, target_dir, use_symlinks=use_symlinks)

    def prepare_modal_data(self, task_id: str):
        from aide.runtime.modal_data_functions import download_kaggle_data

        return download_kaggle_data.remote(self.dataset_name, task_id)

    def get_modal_requirements(self) -> Dict[str, Any]:
        return {
            "pip": ["kaggle"],
            "secrets": ["kaggle-creds"],
        }


class LocalDataProvider(DataProvider):
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def prepare_local_data(self, target_dir: Path, use_symlinks: bool) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        copytree(self.data_dir, target_dir, use_symlinks=use_symlinks)

    def prepare_modal_data(self, task_id: str):
        volume = modal.Volume.from_name("agent-volume")
        task_path = f"tasks/{task_id}"
        with volume.batch_upload() as batch:
            batch.put_directory(self.data_dir, task_path)
        return f"Uploaded local data to tasks/{task_id}"

    def get_modal_requirements(self) -> Dict[str, Any]:
        return {}


@dataclass
class DataConfig:
    provider: str
    dataset: str
    path: Optional[Path] = None


def create_data_provider(cfg) -> DataProvider:
    """Create appropriate data provider based on config."""

    if hasattr(cfg, "data") and cfg.data:
        data_cfg = cfg.data
        if data_cfg.provider == "huggingface":
            return HuggingFaceProvider(data_cfg.dataset)
        if data_cfg.provider == "kaggle":
            return KaggleProvider(data_cfg.dataset)
        if data_cfg.provider == "local":
            if data_cfg.path is None:
                raise ValueError("Local provider requires `path` field")
            return LocalDataProvider(Path(data_cfg.path))

    if getattr(cfg, "data_dir", None):
        return LocalDataProvider(Path(cfg.data_dir))

    raise ValueError("No valid data source specified")
