import os

import torch
from huggingface_hub import HfApi
from mergekit.common import parse_kmb
from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions


def main():
    models = [
        {
            "layer_range": [0, 24],
            "model": "T145/ZEUS-8B-V2",
        },
        {
            "layer_range": [8, 24],
            "model": "T145/ZEUS-8B-V2",
            "parameters": {
                "scale": [
                    {"filter": "o_proj", "value": 0.0},
                    {"filter": "down_proj", "value": 0.0},
                    {"value": 1.0},
                ]
            }
        },
        {
            "layer_range": [24, 32],
            "model": "T145/ZEUS-8B-V2",
        },
    ]
    merge_config = {
        "models": models,
        "merge_method": "passthrough",
        "dtype": "float16"
    }
    MODEL_NAME = "ZEUS-11.5B-V1"
    out_path = f"output/{MODEL_NAME}/"

    os.makedirs(out_path, exist_ok=True)

    run_merge(
        MergeConfiguration(**merge_config),
        out_path,
        MergeOptions(
            #allow_crimes=True,
            out_shard_size=parse_kmb("5B"),
            cuda=torch.cuda.is_available()
        )
    )

    api = HfApi()
    # username = "T145"

    # create_repo(
    #     repo_id = f"{username}/{MODEL_NAME}",
    #     repo_type="model",
    #     exist_ok=True,
    # )

    # api.upload_folder(
    #     folder_path=out_path,
    #     repo_id=f"{username}/{MODEL_NAME}",
    # )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
