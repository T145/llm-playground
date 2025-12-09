import os

import torch
from huggingface_hub import HfApi, create_repo
from mergekit.common import parse_kmb
from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions


def main():
    models = [
        {
            "model": "unsloth/Llama-3.1-Storm-8B",
            "parameters": {
                "density": 0.95,
                "weight": 0.33
            }
        },
        {
            "model": "arcee-ai/Llama-3.1-SuperNova-Lite",
            "parameters": {
                "density": 0.9,
                "weight": 0.29
            }
        },
        {
            "model": "mistralai/Ministral-8B-Instruct-2410",
            "parameters": {
                "density": 0.92,
                "weight": 0.38
            }
        },
        # {
        #     "model": "SicariusSicariiStuff/LLAMA-3_8B_Unaligned_BETA",
        #     "parameters": {
        #         "density": 0.8,
        #         "weight": 0.42
        #     }
        # },
    ]
    seed = 145
    merge_config = {
        "models": models,
        "merge_method": "dare_ties",
        "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "tokenizer_source": "union",
        "dtype": "bfloat16",
        "parameters": {
            "int8_mask": True,
            "normalize": True,
            "random_seed": seed,
        }
    }
    MODEL_NAME = "HADES-8B-V1"
    out_path = f"output/{MODEL_NAME}/"

    os.makedirs(out_path, exist_ok=True)

    run_merge(
        MergeConfiguration(**merge_config),
        out_path,
        MergeOptions(
            #allow_crimes=True,
            random_seed=seed,
            out_shard_size=parse_kmb("17B"),
            cuda=torch.cuda.is_available()
        )
    )

    api = HfApi()
    username = "T145"

    create_repo(
        repo_id = f"{username}/{MODEL_NAME}",
        repo_type="model",
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=out_path,
        repo_id=f"{username}/{MODEL_NAME}",
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
