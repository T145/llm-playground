import os

import torch
from transformers import AutoModelForCausalLM, MllamaForConditionalGeneration, MllamaProcessor

# NOTE: Sufficient DRAM is required to load both models at once

multimodal_model_path = "unsloth/Llama-3.2-11B-Vision-Instruct"  # Original Llama vision model (11B or 90B)
text_model_path = "T145/ZEUS-8B-V22"  # Model to be merged (8B or 70B)
save_path = "./models/T145/ZEUS-8B-V22-Vision"

multimodal_model = MllamaForConditionalGeneration.from_pretrained(
    multimodal_model_path, device_map="cpu", torch_dtype=torch.bfloat16
)
multimodal_processor = MllamaProcessor.from_pretrained(multimodal_model_path)
text_model = AutoModelForCausalLM.from_pretrained(text_model_path, device_map="cpu", torch_dtype=torch.bfloat16)
state_dict_multimodal = multimodal_model.state_dict()
state_dict_text = text_model.state_dict()

num_decoder_layers_text = text_model.config.num_hidden_layers
num_decoder_layers_vision = multimodal_model.config.text_config.num_hidden_layers

# Find the list of inserted layers in multimodal Llama
inserted_layers = set()
for key_multimodal in state_dict_multimodal:
    if "language_model" in key_multimodal and "cross_attn" in key_multimodal and ".layers." in key_multimodal:
        layer_num_multimodal = int(key_multimodal.split(".layers.")[1].split(".")[0]) if ".layers." in key_multimodal else None

        if layer_num_multimodal is not None:
            inserted_layers.add(layer_num_multimodal)
# Here are the hard-coded list of layers added:
# inserted_layers = {3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78, 83, 88, 93, 98} $ For 90B
# inserted_layers = {3, 8, 13, 18, 23, 28, 33, 38} $ For 11B

assert len(inserted_layers) == num_decoder_layers_vision - num_decoder_layers_text, "# of added layers do not match"

# Build decoder layer map from multimodal layer# to text layer#, skipping layers listed in inserted_layers
layer_map = dict()
layer_num_multimodal = 0
for layer_num_text in range(num_decoder_layers_text):
    while layer_num_multimodal in inserted_layers:
        layer_num_multimodal += 1  # Increment to skip mismatched layers

    layer_map[layer_num_multimodal] = layer_num_text
    layer_num_multimodal += 1

for key_multimodal in state_dict_multimodal:
    if "language_model" not in key_multimodal:
        continue  # A multi-modal param

    if "cross_attn" in key_multimodal:
        continue  # A multi-modal param

    key_text = key_multimodal.replace("language_model.", "")

    if "embed_tokens.weight" in key_multimodal:  # Handle embed tokens separately
        assert key_text in state_dict_text, f"Key not found: {key_text}"
        extra_tokens = state_dict_multimodal[key_multimodal].shape[0] - state_dict_text[key_text].shape[0]
        state_dict_multimodal[key_multimodal][: state_dict_text[key_text].shape[0], :].copy_(state_dict_text[key_text])
        print(f"Replaced {key_multimodal} with {key_text} (preserving last {extra_tokens} tokens)")
        continue

    if "lm_head" in key_multimodal or "model.norm.weight" in key_multimodal:  # Handle other non-decoder layers separately
        assert key_text in state_dict_text, f"Key not found: {key_text}"
        state_dict_multimodal[key_multimodal].copy_(state_dict_text[key_text])
        print(f"Replaced {key_multimodal} with {key_text}")
        continue

    layer_num_multimodal = int(key_multimodal.split(".layers.")[1].split(".")[0]) if ".layers." in key_multimodal else None

    assert layer_num_multimodal is not None, f"Unknown non-decoder key encountered: {key_multimodal}"

    if layer_num_multimodal in inserted_layers:
        continue  # Skip mismatched layers

    assert layer_num_multimodal in layer_map, f"Layer not found in layer_map: {layer_num_multimodal}"

    layer_num_text = layer_map[layer_num_multimodal]
    key_text = key_text.replace(f".layers.{layer_num_multimodal}.", f".layers.{layer_num_text}.")

    assert key_text in state_dict_text, f"Key not found: {key_text}"

    state_dict_multimodal[key_multimodal].copy_(state_dict_text[key_text])

    print(f"Replaced {key_multimodal} with {key_text}")

print("Merged model successfully. Saving...")
# Apply the changes
multimodal_model.load_state_dict(state_dict_multimodal)

# Create save_path if it does not exist
os.makedirs(save_path, exist_ok=True)
multimodal_model.save_pretrained(save_path, safe_serialization=True, max_shard_size="8192MB")
multimodal_processor.save_pretrained(save_path)
print(f"Model saved to {save_path}")
