from datasets import load_dataset
from huggingface_hub import HfApi

dataset = load_dataset("nothingiisreal/Human_Stories", split="train")


username = "T145"
dataset_name = "stories&prompts"
api = HfApi()

api.create_repo(
    repo_id=f"{username}/{dataset_name}",
    repo_type="dataset",
    exist_ok=True,
)

dataset.push_to_hub(f"{username}/{dataset_name}")
