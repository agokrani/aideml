"""Modal functions for data downloading in Modal environment."""

import modal

app = modal.App("aide-data-prep")
volume = modal.Volume.from_name("agent-volume", create_if_missing=True)


@app.function(
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    pip=["datasets", "transformers"]
)
def download_huggingface_data(dataset_name: str, task_id: str):
    """Download HuggingFace dataset directly to tasks/{task_id}."""
    from datasets import load_dataset
    
    dataset = load_dataset(dataset_name)
    target_path = f"/data/tasks/{task_id}"
    dataset.save_to_disk(target_path)
    
    return f"Downloaded {dataset_name} to tasks/{task_id}"


@app.function(
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("kaggle-creds")],
    pip=["kaggle"]
)
def download_kaggle_data(dataset_name: str, task_id: str):
    """Download Kaggle dataset directly to tasks/{task_id}."""
    import kaggle
    import os
    
    # Set up Kaggle credentials from secrets
    os.environ["KAGGLE_USERNAME"] = os.environ.get("KAGGLE_USERNAME", "")
    os.environ["KAGGLE_KEY"] = os.environ.get("KAGGLE_KEY", "")
    
    target_path = f"/data/tasks/{task_id}"
    
    # Download and extract dataset
    kaggle.api.dataset_download_files(dataset_name, path=target_path, unzip=True)
    
    return f"Downloaded {dataset_name} to tasks/{task_id}"