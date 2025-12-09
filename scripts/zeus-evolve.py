import os

import torch
from huggingface_hub import HfApi, create_repo
from mergekit.common import parse_kmb
from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions


def main():
    models = [
        "akjindal53244/Llama-3.1-Storm-8B",
        "arcee-ai/Llama-3.1-SuperNova-Lite",
        "SicariusSicariiStuff/LLAMA-3_8B_Unaligned_BETA",
        "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
        "NCSOFT/Llama-VARCO-8B-Instruct"
    ]
    merge_config = {
        "models": models,
        "merge_method": "dare_ties",
        "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "tokenizer_source": "base",
        "dtype": "bfloat16"
    }
    out_path = "output/zeus/"

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
    username = "T145"
    MODEL_NAME = "ZEUS-8B-V7"

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
