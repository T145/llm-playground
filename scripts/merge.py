from mergekit.merge import run_merge
from mergekit.options import MergeOptions
from mergekit.config import MergeConfiguration
import torch

torch.cuda.empty_cache()

models = [
    {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "parameters": {
            "density": 1,
            "weight": 1
        }
    }
]
merge_config = {
    "models": models,
    "merge_method": "ties",
    "base_model": "Qwen/Qwen2.5-3B",
    "parameters": {
        "weight": 1,
        "density": 1,
        "normalize": True,
        "int8_mask": True
    },
    "dtype": "bfloat16"
}
options = {
    "cuda": True,
    #"random_seed": 42,
}

run_merge(MergeConfiguration(**merge_config), "output/", MergeOptions(**options))
