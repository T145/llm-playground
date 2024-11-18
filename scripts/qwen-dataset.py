from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi
from nomic import atlas, embed
from nomic.data_inference import NomicTopicOptions


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
dataset = load_dataset("Open-Orca/OpenOrca", split="train")
dataset = dataset.shuffle(seed=42).select(range(1000)) # Only use 1000 samples for quick demo


def format_chat_template(row):

    row_json = [{"role": "system", "content": row["system_prompt"] },
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["response"]}]

    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row


username = "T145"
dataset_name = "qwen_orca_sample"
dataset = dataset.map(format_chat_template)
api = HfApi()

api.create_repo(
    repo_id = f"{username}/{dataset_name}",
    repo_type="dataset",
    exist_ok=True,
)

dataset.push_to_hub(f"{username}/{dataset_name}")

atlas.map_data(
    data=list(dataset),
    indexed_field='text',
    identifier=dataset_name
)
