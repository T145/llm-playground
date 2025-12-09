from datasets import load_dataset
from huggingface_hub import HfApi

dataset = load_dataset("prithivMLmods/Math-Forge-Hard", split="train")
dataset = dataset.remove_columns(["single_shot"])
dataset = dataset.rename_column("problem", "text")

username = "T145"
dataset_name = "Math-Layer-Check"
api = HfApi()

api.create_repo(
    repo_id=f"{username}/{dataset_name}",
    repo_type="dataset",
    exist_ok=True,
)

dataset.push_to_hub(f"{username}/{dataset_name}")
