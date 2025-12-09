import itertools
import sys

from ruamel.yaml import YAML

sys.dont_write_bytecode = True

MODEL_NAME = "ZEUS-8B-V28"

config_one = {
    "base_model": "Skywork/Skywork-o1-Open-Llama-3.1-8B",
    "dtype": "bfloat16",
    "merge_method": "slerp",
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
    "parameters": {"t": [{"value": 0.5}]},
    "name": "strawberry-patch",
}

config_two = {
    "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "dtype": "bfloat16",
    "merge_method": "dare_ties",
    "slices": [
        {
            "sources": [
                {
                    "layer_range": [0, 32],
                    "model": "unsloth/Llama-3.1-Storm-8B",
                    "parameters": {"density": 0.94, "weight": 0.35},
                },
                {
                    "layer_range": [0, 32],
                    "model": "arcee-ai/Llama-3.1-SuperNova-Lite",
                    "parameters": {"density": 0.92, "weight": 0.26},
                },
                {
                    "layer_range": [0, 32],
                    "model": "VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct",
                    "parameters": {
                        "density": 0.91,
                        "weight": [
                            {"filter": "layers.21.", "value": 0.0},
                            {"filter": "layers.22.", "value": 0.0},
                            {"filter": "layers.23.", "value": 0.0},
                            {"filter": "layers.24.", "value": 0.0},
                            {"filter": "layers.25.", "value": 0.0},
                            {"filter": "layers.26.", "value": 0.0},
                            {"filter": "layers.27.", "value": 0.0},
                            {"filter": "layers.28.", "value": 0.0},
                            {"value": 0.2},
                        ],
                    },
                },
                {
                    "layer_range": [0, 32],
                    "model": "strawberry-patch",
                    "parameters": {
                        "density": 0.92,
                        "weight": [
                            {"filter": "layers.21.", "value": 0.2},
                            {"filter": "layers.22.", "value": 0.2},
                            {"filter": "layers.23.", "value": 0.2},
                            {"filter": "layers.24.", "value": 0.2},
                            {"filter": "layers.25.", "value": 0.2},
                            {"filter": "layers.26.", "value": 0.2},
                            {"filter": "layers.27.", "value": 0.2},
                            {"filter": "layers.28.", "value": 0.2},
                            {"value": 0.0},
                        ],
                    },
                },
                {
                    "layer_range": [0, 32],
                    "model": "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
                    "parameters": {"density": 0.93, "weight": 0.19},
                },
                {
                    "layer_range": [0, 32],
                    "model": "unsloth/Meta-Llama-3.1-8B-Instruct",
                },
            ]
        },
    ],
    "tokenizer": {
        "source": "union",
        "tokens": {
            "<|begin_of_text|>": {
                "source": "unsloth/Meta-Llama-3.1-8B-Instruct",
                "force": True,
            },
            "<|eot_id|>": {
                "source": "unsloth/Meta-Llama-3.1-8B-Instruct",
                "force": True,
            },
            "<|finetune_right_pad_id|>": {
                "source": "unsloth/Meta-Llama-3.1-8B-Instruct",
                "force": True,
            },
        },
    },
    "parameters": {
        "int8_mask": 1.0,
        "normalize": 1.0,
        "random_seed": 145,
    },
    "name": MODEL_NAME,
}

merge_config = [config_one, config_two]

#print(yaml.dump(config_one, default_flow_style=None, sort_keys=False))

with open("README.md", "w") as readme:
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
    readme.writelines(["\n## Merge Configuration\n\n", "The following YAML configuration was used to produce this model:\n\n", "```yaml\n"])

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
