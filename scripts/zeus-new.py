import json
import os
from itertools import chain

import torch
from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions
from ruamel.yaml import YAML
from transformers import __version__ as transformers_version
from urllib3 import PoolManager

MODEL_NAME = "ZEUS-8B-R1"


def load_gist(gist_id) -> str:
    """Translate Gist ID to URL"""

    with PoolManager() as pool:
        gist_api = pool.request("GET", f"https://api.github.com/gists/{gist_id}").json()
        files = gist_api["files"]
        file_key = next(iter(files))
        files_head_member = files[file_key]
        gist_src = files_head_member["content"]

    return gist_src


def sorted_dict(d: dict) -> dict:
    return dict(sorted(d.items()))


def create_readme(merge_config: dict, out_path: str) -> None:
    out_path = os.path.join(out_path, merge_config[-1]["name"])

    with open(os.path.join(out_path, "README.md"), "w") as readme:
        yaml = YAML(typ="safe")

        yaml.default_flow_style = False

        models = [
            [slice["model"] for slice in source if "/" in slice["model"]]
            for source in [config["slices"][0]["sources"] for config in merge_config]
        ]
        models = list(sorted(chain.from_iterable(models), key=str.lower))

        readme.writelines(["---\n", "base_model:\n"])
        yaml.dump(models, readme)
        readme.writelines(["library_name: transformers\n", "license: llama3.1\n", "tags:\n"])
        yaml.dump(["mergekit", "merge", "llama-3.1", "llama", "instruct"], readme)
        readme.write("---\n")
        readme.writelines(
            [
                f"# {MODEL_NAME.replace('-', ' ')}\n\n",
                "This model is a merge of the following pre-trained and/or finetuned LLMs, created using [mergekit](https://github.com/cg123/mergekit).\n\n",
            ]
        )

        base = merge_config[-1]["base_model"]

        models.remove(base)
        readme.write(f"* **(base)** [{base}](https://huggingface.co/{base})\n")
        readme.writelines([f"* [{model}](https://huggingface.co/{model})\n" for model in models])
        del models
        readme.writelines(
            ["\n## Merge Configuration\n\n", "The following YAML configuration was used to produce this model:\n\n", "```yaml\n"]
        )

        def tr(c: str):
            ret = list()
            it = iter(c.splitlines())

            for line in it:
                if "layer_range" in line:
                    bounds = [int(next(it)[-2:]) for _ in range(2)]
                    ret.append(f"{line} {str(bounds)}")
                else:
                    ret.append(line)

            return "\n".join(ret)

        for i, c in enumerate(merge_config):
            if i > 0:
                readme.write("\n---\n")

            yaml.dump(c, readme, transform=tr)

        readme.write("\n```")


def update_model_configs(out_path: str) -> None:
    with open(os.path.join(out_path, "config.json"), "r+") as config:
        data = json.load(config)
        data["_name_or_path"] = MODEL_NAME
        data["attn_implementation"] = "eager"
        config.seek(0)
        json.dump(sorted_dict(data), config, indent=2)
        config.truncate()

    with open(os.path.join(out_path, "tokenizer_config.json"), "r+") as config:
        data = json.load(config)
        data["padding_side"] = "right"
        data["pad_token"] = "<|eot_id|>"
        config.seek(0)
        json.dump(data, config, indent=2)
        config.truncate()

    generation_config = {
        "bos_token_id": 128000,
        "do_sample": True,
        "eos_token_id": [128001, 128008, 128009],
        "pad_token_id": 128004,
        "temperature": 0.72,
        "top_k": 40,
        "top_p": 0.4,
        "repetition_penalty": 1.18,
        "prompt": load_gist("faf215450a87456da8335b840a0f88c4").strip(),
        "transformers_version": transformers_version,
    }

    with open(os.path.join(out_path, "generation_config.json"), "w") as config:
        json.dump(sorted_dict(generation_config), config, indent=2)


def get_merge_config(random_seed: int) -> list:
    config_one = {
        "base_model": "Skywork/Skywork-o1-Open-Llama-3.1-8B",
        "dtype": "bfloat16",
        "merge_method": "slerp",
        "name": "strawberry-patch",
        "parameters": {"t": [{"value": 0.5}]},
        "slices": [
            {
                "sources": [
                    {
                        "layer_range": [0, 32],
                        "model": "FreedomIntelligence/HuatuoGPT-o1-8B",
                    },
                    {
                        "layer_range": [0, 32],
                        "model": "Skywork/Skywork-o1-Open-Llama-3.1-8B",
                    },
                ],
            },
        ],
    }

    config_two = {
        "base_model": "T145/ZEUS-8B-V22",
        "dtype": "bfloat16",
        "merge_method": "arcee_fusion",
        "slices": [
            {
                "sources": [
                    # {
                    #     "layer_range": [0, 32],
                    #     "model": "unsloth/Llama-3.1-Storm-8B",
                    #     "parameters": {"density": 0.94, "weight": 0.35},
                    # },
                    # {
                    #     "layer_range": [0, 32],
                    #     "model": "arcee-ai/Llama-3.1-SuperNova-Lite",
                    #     "parameters": {"density": 0.92, "weight": 0.26},
                    # },
                    # {
                    #     "layer_range": [0, 32],
                    #     "model": "VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct",
                    #     "parameters": {"density": 0.91, "weight": 0.2},
                    # },
                    # {
                    #     "layer_range": [0, 32],
                    #     "model": "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
                    #     "parameters": {"density": 0.93, "weight": 0.19},
                    # },
                    {
                        "layer_range": [0, 32],
                        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    },
                    {
                        "layer_range": [0, 32],
                        "model": "T145/ZEUS-8B-V22",
                    },
                ]
            },
        ],
        "tokenizer": {
            "source": "union",
            "tokens": {
                "<|begin_of_text|>": {
                    "source": "T145/ZEUS-8B-V22",
                    "force": True,
                },
                "<|eot_id|>": {
                    "source": "T145/ZEUS-8B-V22",
                    "force": True,
                },
            },
        },
        "parameters": {
            "int8_mask": 1.0,
            "normalize": 1.0,
            "random_seed": random_seed,
        },
        "name": MODEL_NAME,
    }

    return config_two


def main():
    out_path = "output/zeus/"

    os.makedirs(out_path, exist_ok=True)

    random_seed = 145
    options = MergeOptions(
        # allow_crimes=True,
        trust_remote_code=True,
        random_seed=random_seed,
        out_shard_size=4.23 * 1000 * 1000 * 1000,
        cuda=torch.cuda.is_available(),
        low_cpu_mem_usage=False,
        write_model_card=True
    )
    merge_config = get_merge_config(random_seed)

    run_merge(
        MergeConfiguration(**merge_config),
        out_path,
        options
    )
    update_model_configs(out_path)
    #create_readme(merge_config, out_path)

    # api = HfApi()
    # username = "T145"
    # MODEL_NAME = "ZEUS-8B-V7"

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/
    torch.set_num_threads(os.cpu_count())
    main()
