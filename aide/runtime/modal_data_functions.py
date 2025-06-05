"""Modal functions for data downloading in Modal environment."""

import os
import modal
from pathlib import Path

image = (
    # modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    modal.Image.debian_slim()
    .pip_install_from_requirements(
        Path(__file__).parent.parent.parent / "requirements.txt"
    )
    .pip_install(
        "transformers",
        "datasets",
        "huggingface_hub[hf_transfer]",
        "kaggle",
        "humanize",
        "pydantic",
    )
)

app = modal.App(name="aide-data-prep", image=image)
volume = modal.Volume.from_name("agent-volume", create_if_missing=True)


@app.function(
    volumes={"/data": volume},
    secrets=[modal.Secret.from_dotenv(Path(__file__).parent.parent.parent)],
)
def download_huggingface_data(
    dataset_name: str, task_id: str, dataset_kwargs: dict | None = None
):
    """Download HuggingFace dataset directly to tasks/{task_id}."""
    from datasets import load_dataset

    target_path = f"/data/tasks/{task_id}"
    try:
        dataset = load_dataset(
            dataset_name, token=os.environ.get("HF_TOKEN", ""), **dataset_kwargs
        )
        dataset.save_to_disk(target_path)
        return f"Downloaded {dataset_name} to tasks/{task_id}"
    except ValueError as e:
        if "Invalid pattern" in str(e) and "**" in str(e):
            # Fallback to snapshot_download for Git repositories
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                local_dir=target_path,
                token=os.environ.get("HF_TOKEN", ""),
                **dataset_kwargs,
            )
            return f"Downloaded repository {dataset_name} to tasks/{task_id}"
        else:
            raise  # Re-raise if it's a different ValueError


@app.function(
    volumes={"/data": volume},
    secrets=[modal.Secret.from_dotenv(Path(__file__).parent.parent.parent)],
    # secrets=[modal.Secret.from_name("kaggle-creds")],
)
def download_kaggle_data(dataset_name: str, task_id: str):
    """Download Kaggle dataset directly to tasks/{task_id}."""
    import kaggle

    # Set up Kaggle credentials from secrets
    os.environ["KAGGLE_USERNAME"] = os.environ.get("KAGGLE_USERNAME", "")
    os.environ["KAGGLE_KEY"] = os.environ.get("KAGGLE_KEY", "")

    target_path = f"/data/tasks/{task_id}"

    # Download and extract dataset
    kaggle.api.dataset_download_files(dataset_name, path=target_path, unzip=True)

    return f"Downloaded {dataset_name} to tasks/{task_id}"
