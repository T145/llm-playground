from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi

custom = load_dataset("json", data_files="scripts/custom.json", split="train")
# tokenizer = AutoTokenizer.from_pretrained("T145/ZEUS-8B-V2")
dataset = load_dataset("mlabonne/orpo-dpo-mix-40k", split="train")
dataset = concatenate_datasets([dataset, custom], split="train")

include = list(
    #set(load_dataset("argilla/Capybara-Preferences", split="train")["source"])
)
include.extend(
    [
        # 'distilabel-math-preference-dpo', # has too many "you're an AI" prompts
        "toxic-dpo-v0.2",
        "truthy_dpo",
        "evol_instruct",
        "T145",
    ]
)
a = dataset["source"]
b = [i for i, x in enumerate(a) if x in include]
dataset = dataset.select(b)

for _ in range(3):
    dataset = dataset.shuffle(seed=42)

# def format_chat_template(row):

#     row_json = [{"role": "system", "content": row["system_prompt"] },
#             {"role": "user", "content": row["question"]},
#             {"role": "assistant", "content": row["response"]}]

#     row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
#     return row


username = "T145"
dataset_name = "clear_orpo_13k"
api = HfApi()

api.create_repo(
    repo_id=f"{username}/{dataset_name}",
    repo_type="dataset",
    exist_ok=True,
)

dataset.push_to_hub(f"{username}/{dataset_name}")

# atlas.map_data(
#     data=list(dataset),
#     indexed_field='text',
#     identifier=dataset_name
# )
