import itertools
import json
import logging
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
from mergekit.scripts.megamerge import has_circular_dependency
from ruamel.yaml import YAML
from transformers import __version__ as transformers_version
from urllib3 import PoolManager

sys.dont_write_bytecode = True

MODEL_NAME = "ZEUS-8B-R1"
merges = dict()


def load_gist(gist_id) -> str:
    """Translate Gist ID to URL"""

    with PoolManager() as pool:
        gist_api = pool.request("GET", f"https://api.github.com/gists/{gist_id}").json()
        files = gist_api["files"]
        file_key = next(iter(files))
        files_head_member = files[file_key]
        gist_src = files_head_member["content"]

    return gist_src


def get_merge_config(random_seed: int) -> list:
    # config_one = {
    #     "base_model": "Skywork/Skywork-o1-Open-Llama-3.1-8B",
    #     "dtype": "bfloat16",
    #     "merge_method": "slerp",
    #     "name": "strawberry-patch",
    #     "parameters": {"t": [{"value": 0.5}]},
    #     "slices": [
    #         {
    #             "sources": [
    #                 {
    #                     "layer_range": [0, 32],
    #                     "model": "FreedomIntelligence/HuatuoGPT-o1-8B",
    #                 },
    #                 {
    #                     "layer_range": [0, 32],
    #                     "model": "Skywork/Skywork-o1-Open-Llama-3.1-8B",
    #                 },
    #             ],
    #         },
    #     ],
    # }

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

    return [config_two]


def merge(m: str, merge_options: MergeOptions, force: bool, out_path: Path) -> None:
    """
    Merges a model and its dependencies

    Params:
        m: name of the model to merge
        merge_options: MergeOptions
        force: overwrite existing merge results
        out_path: output path
    """
    # check if output_path exists
    if os.path.exists(out_path / m):
        if not force:
            logging.info("Skipping %s as it already exists", m)
            del merges[m]
            return
        logging.info("Overwriting %s as --force was specified", m)

    if len(merges[m]["deps"]) != 0:
        for dep in merges[m]["deps"]:
            if dep in merges:
                merge(dep, merge_options, force, out_path)

    logging.info("Merging model %s", m)
    merge_config: MergeConfiguration = MergeConfiguration.model_validate(merges[m])
    run_merge(
        merge_config,
        str(out_path / merges[m]["name"]),
        options=merge_options,
    )
    del merges[m]


def add_model_deps(model: str, name: str, out_path: Path) -> str:
    """
    Adds a model to `name`s dependencies if it is not already there and is a merge
    """
    model_lora = model.split("+")
    dep = model

    # name must not have a slash to avoid path traversal
    # therefore, we can use it to check if its a merge from the config
    if "/" not in model_lora[0]:
        # avoid duplicate deps
        if model_lora[0] not in merges[name]["deps"]:
            merges[name]["deps"].append(model_lora[0])

        dep = str(out_path / model_lora[0])

        if len(model_lora) == 2:
            dep += "+" + model_lora[1]

    return dep


def sorted_dict(d: dict) -> dict:
    return dict(sorted(d.items()))


def merge_models(
    merge_options: MergeOptions,
    out_path: str,
    force: bool,
    verbose: bool,
) -> str:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    out_path = Path(out_path)
    merge_config = get_merge_config(merge_options.random_seed)

    for config in merge_config:
        if "name" not in config:
            logging.error("All configs must have a name.")
            sys.exit(1)

        if "/" in config["name"]:
            logging.error("The name field must not contain a slash.")
            sys.exit(1)

        merges[config["name"]] = config
        merges[config["name"]]["deps"] = list()

        if "base_model" in config:
            config["base_model"] = add_model_deps(config["base_model"], config["name"], out_path)

        if "slices" in config:
            for slc in config["slices"]:
                for src in slc["sources"]:
                    src["model"] = add_model_deps(src["model"], config["name"], out_path)

        if "models" in config:
            for model in config["models"]:
                model["model"] = add_model_deps(model["model"], config["name"], out_path)

    logging.info("Merging: %s", ", ".join(merges))

    if (dep := has_circular_dependency(merges)) is not None:
        logging.error("Circular dependency detected: %s", dep)
        sys.exit(1)

    while len(merges) != 0:
        m = list(merges.keys())[0]
        merge(m, merge_options, force, out_path)

    for c in merge_config:
        if "deps" in c:
            c.pop("deps", None)

    out_path = os.path.join(out_path, merge_config[-1]["name"])

    with open(os.path.join(out_path, "README.md"), "w") as readme:
        yaml = YAML(typ="safe")
        yaml.default_flow_style = False
        models = [
            [slice["model"] for slice in source if "/" in slice["model"]]
            for source in [config["slices"][0]["sources"] for config in merge_config]
        ]
        models = list(sorted(itertools.chain.from_iterable(models), key=str.lower))

        readme.writelines(["---\n", "base_model:\n"])
        yaml.dump(models, readme)
        readme.writelines(["library_name: transformers\n", "license: llama3.1\n", "tags:\n"])
        yaml.dump(["mergekit", "merge", "llama-3.1", "llama", "instruct"], readme)
        readme.write("---\n")
        readme.writelines(
            [
                f"# {MODEL_NAME.replace('-', ' ')}\n\n",
                "This model is a merge of the following pre-trained and finetuned LLMs, created using [mergekit](https://github.com/cg123/mergekit).\n\n",
            ]
        )
        base = merge_config[-1]["base_model"]
        models.remove(base)
        readme.write(f"* **(base)** [{base}](https://huggingface.co/{base})\n")
        readme.writelines([f"* [{model}](https://huggingface.co/{model})\n" for model in models])
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

    return out_path


def main():
    # with open("mega.yml", "r") as c:
    #     data = yaml.load_all(c, Loader=yaml.FullLoader)
    #     pprint(list(data))

    opts = MergeOptions(
        # allow_crimes=True,
        trust_remote_code=True,
        random_seed=145,
        out_shard_size=4.23 * 1000 * 1000 * 1000,
        # cuda=torch.cuda.is_available(),
        low_cpu_mem_usage=False,
        write_model_card=False,  # Handle this ourselves! :D
    )
    out_path = merge_models(opts, "./output/", False, False)

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/
    torch.set_num_threads(os.cpu_count())
    main()
