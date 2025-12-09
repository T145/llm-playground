import os

import torch
from huggingface_hub import HfApi, create_repo
from mergekit.common import parse_kmb
from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions


def main():
    models = [
        {"model": "unsloth/Meta-Llama-3.1-8B-Instruct", "parameters": {"density": 1, "weight": 1}},
    ]
    merge_config = {
        "models": models,
        "merge_method": "ties",
        "parameters": {"density": 1, "weight": 1},
        "base_model": "unsloth/Meta-Llama-3.1-8B",
        "tokenizer_source": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "dtype": "bfloat16",
    }
    options = {
        #"allow_crimes": True,
        "out_shard_size": parse_kmb("5B"),
        "cuda": False,
    }
    MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-TIES"
    out_path = os.path.join(os.getcwd(), f"output/{MODEL_NAME}/")

    os.makedirs(out_path, exist_ok=True)

    run_merge(MergeConfiguration(**merge_config), out_path, MergeOptions(**options))

    # subprocess.Popen(f"docker run --gpus all --rm -v {os.getcwd()}/output:/models/ -e CUDA_VERSION=12.4.0 ghcr.io/ggerganov/llama.cpp:full-cuda --convert --outtype f16 --outfile /models/{MODEL_NAME}.f16.gguf /models/{MODEL_NAME}/")

    # for quant in ["Q8_0", "Q4_K_M"]:
    #     subprocess.Popen(f"docker run --gpus all --rm -v {os.getcwd()}/output:/models/ -e CUDA_VERSION=12.4.0 ghcr.io/ggerganov/llama.cpp:full-cuda --quantize /models/{MODEL_NAME}.f16.gguf /models/{MODEL_NAME}/{MODEL_NAME}.{quant}.gguf {quant}")

    # os.remove(f"{MODEL_NAME}.f16.gguf")

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
