from huggingface_hub import HfApi, create_repo

api = HfApi()
username = "T145"
MODEL_NAME = "ZEUS-8B-V22-W8A8-INT8"
out_path = f"{MODEL_NAME}"

create_repo(
    repo_id = f"{username}/{MODEL_NAME}",
    repo_type="model",
    exist_ok=True,
)

api.upload_folder(
    folder_path=out_path,
    repo_id=f"{username}/{MODEL_NAME}",
)
