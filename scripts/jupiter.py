import os

import torch
from huggingface_hub import HfApi, create_repo
from mergekit.common import parse_kmb
from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions

# allenai/Llama-3.1-Tulu-3-8B (change pad token)
# FreedomIntelligence/HuatuoGPT-o1-8B (change pad token)


def main():
    models = [
        {
            "model": "arcee-ai/Llama-3.1-SuperNova-Lite",
            "parameters": {
                "weight": [{"filter": "lm_head", "value": 0.0}, {"filter": "self_attn.o_proj", "value": 0.0}, {"filter": "mlp.down_proj", "value": 0.0}, {"value": 0.42}],
                "density": 0.9,
                "gamma": 0.01,
            },
        },
        {
            "model": "unsloth/Llama-3.1-Storm-8B",
            "parameters": {
                "weight": [{"filter": "lm_head", "value": 0.0}, {"filter": "self_attn.o_proj", "value": 0.0}, {"filter": "mlp.down_proj", "value": 0.0}, {"value": 0.25}],
                "density": 0.9,
                "gamma": 0.01,
            },
        },
        {
            "model": "VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct",
            "parameters": {
                "weight": [{"filter": "lm_head", "value": 0.0}, {"filter": "self_attn.o_proj", "value": 0.0}, {"filter": "mlp.down_proj", "value": 0.0}, {"value": 0.33}],
                "density": 0.9,
                "gamma": 0.01,
            },
        },
    ]
    seed = 145
    merge_config = {
        "models": models,
        "merge_method": "breadcrumbs_ties",
        "base_model": "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
        "tokenizer": {
            "source": "union"
        },
        "dtype": "bfloat16",
        "parameters": {
            "int8_mask": True,
            "normalize": True,
            "random_seed": seed,
        },
    }
    MODEL_NAME = "JUPITER-8B-V1"
    out_path = f"output/{MODEL_NAME}/"

    os.makedirs(out_path, exist_ok=True)

    run_merge(
        MergeConfiguration(**merge_config),
        out_path,
        MergeOptions(
            # allow_crimes=True,
            trust_remote_code=True,
            random_seed=seed,
            out_shard_size=parse_kmb("7B"),
            cuda=torch.cuda.is_available(),
        ),
    )

    api = HfApi()
    username = "T145"

    create_repo(
        repo_id=f"{username}/{MODEL_NAME}",
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
