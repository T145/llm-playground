import os

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2VLForConditionalGeneration

# NOTE: You need sufficient DRAM to load both models at once (otherwise, need to process layer by layer)

multimodal_model_path = "Qwen/Qwen2.5-VL-7B-Instruct"  # Original Qwen vision model
text_model_path = "Qwen/Qwen3-30B-A3B"  # Text model to be merged
save_path = "./models/Qwen3-30B-A3B-Vision"

print("Loading multimodal model...")
multimodal_model = Qwen2VLForConditionalGeneration.from_pretrained(
    multimodal_model_path, device_map="cpu", torch_dtype=torch.bfloat16
)
multimodal_processor = AutoProcessor.from_pretrained(multimodal_model_path)

print("Loading text model...")
text_model = AutoModelForCausalLM.from_pretrained(
    text_model_path, device_map="cpu", torch_dtype=torch.bfloat16
)

state_dict_multimodal = multimodal_model.state_dict()
state_dict_text = text_model.state_dict()

num_decoder_layers_text = text_model.config.num_hidden_layers
num_decoder_layers_vision = multimodal_model.config.text_config.num_hidden_layers

print(f"Text model layers: {num_decoder_layers_text}")
print(f"Vision model text layers: {num_decoder_layers_vision}")

# Find the list of vision-specific layers in multimodal Qwen
vision_specific_keys = set()
for key_multimodal in state_dict_multimodal:
    if any(component in key_multimodal for component in ["visual", "vision", "cross_attn", "merger"]):
        vision_specific_keys.add(key_multimodal)

print(f"Found {len(vision_specific_keys)} vision-specific parameters")

# Handle size mismatch - we need to map layers appropriately
# Since we're going from 30B to 7B vision model, we'll need to sample layers
if num_decoder_layers_text > num_decoder_layers_vision:
    # Sample layers from the larger text model
    layer_step = num_decoder_layers_text / num_decoder_layers_vision
    layer_map = {}
    for vision_layer in range(num_decoder_layers_vision):
        text_layer = int(vision_layer * layer_step)
        layer_map[vision_layer] = text_layer
    print(f"Mapping {num_decoder_layers_vision} vision layers to sampled layers from {num_decoder_layers_text} text layers")
else:
    # Direct mapping if text model is smaller or same size
    layer_map = {i: i for i in range(min(num_decoder_layers_text, num_decoder_layers_vision))}

print("Layer mapping:", layer_map)

# Replace text components while preserving vision components
replaced_count = 0
skipped_count = 0

for key_multimodal in state_dict_multimodal:
    # Skip vision-specific components
    if key_multimodal in vision_specific_keys:
        print(f"Preserving vision component: {key_multimodal}")
        skipped_count += 1
        continue

    # Handle language model components
    if "language_model" in key_multimodal or "model" in key_multimodal:
        # Extract the corresponding key for the text model
        key_text = key_multimodal
        if "language_model." in key_multimodal:
            key_text = key_multimodal.replace("language_model.", "")
        elif "model." in key_multimodal:
            key_text = key_multimodal.replace("model.", "")

        # Handle embedding tokens - may need to handle vocab size differences
        if "embed_tokens.weight" in key_multimodal:
            if key_text in state_dict_text:
                text_vocab_size = state_dict_text[key_text].shape[0]
                vision_vocab_size = state_dict_multimodal[key_multimodal].shape[0]

                if text_vocab_size <= vision_vocab_size:
                    # Copy text embeddings to the beginning of vision embeddings
                    state_dict_multimodal[key_multimodal][:text_vocab_size, :].copy_(state_dict_text[key_text])
                    print(f"Replaced {key_multimodal} with {key_text} (vocab: {text_vocab_size} -> {vision_vocab_size})")
                    replaced_count += 1
                else:
                    print(f"Warning: Text vocab size ({text_vocab_size}) > Vision vocab size ({vision_vocab_size}), truncating")
                    state_dict_multimodal[key_multimodal].copy_(state_dict_text[key_text][:vision_vocab_size, :])
                    replaced_count += 1
                continue

        # Handle layer-specific parameters
        if ".layers." in key_multimodal:
            # Extract layer number from multimodal model key
            try:
                layer_part = key_multimodal.split(".layers.")[1]
                layer_num_vision = int(layer_part.split(".")[0])

                if layer_num_vision in layer_map:
                    layer_num_text = layer_map[layer_num_vision]
                    # Construct the corresponding text model key
                    key_text_with_layer = key_text.replace(f".layers.{layer_num_vision}.", f".layers.{layer_num_text}.")

                    if key_text_with_layer in state_dict_text:
                        # Check tensor shape compatibility
                        if state_dict_multimodal[key_multimodal].shape == state_dict_text[key_text_with_layer].shape:
                            state_dict_multimodal[key_multimodal].copy_(state_dict_text[key_text_with_layer])
                            print(f"Replaced {key_multimodal} with {key_text_with_layer}")
                            replaced_count += 1
                        else:
                            print(f"Shape mismatch for {key_multimodal}: {state_dict_multimodal[key_multimodal].shape} vs {state_dict_text[key_text_with_layer].shape}")
                            skipped_count += 1
                    else:
                        print(f"Key not found in text model: {key_text_with_layer}")
                        skipped_count += 1
                else:
                    print(f"Layer {layer_num_vision} not in layer map")
                    skipped_count += 1
            except (ValueError, IndexError) as e:
                print(f"Error parsing layer number from {key_multimodal}: {e}")
                skipped_count += 1
            continue

        # Handle other non-layer parameters (norm, lm_head, etc.)
        if key_text in state_dict_text:
            if state_dict_multimodal[key_multimodal].shape == state_dict_text[key_text].shape:
                state_dict_multimodal[key_multimodal].copy_(state_dict_text[key_text])
                print(f"Replaced {key_multimodal} with {key_text}")
                replaced_count += 1
            else:
                print(f"Shape mismatch for {key_multimodal}: {state_dict_multimodal[key_multimodal].shape} vs {state_dict_text[key_text].shape}")
                skipped_count += 1
        else:
            print(f"Key not found in text model: {key_text}")
            skipped_count += 1

print("\nMerge summary:")
print(f"Replaced parameters: {replaced_count}")
print(f"Skipped parameters: {skipped_count}")
print(f"Vision-specific parameters preserved: {len(vision_specific_keys)}")

print("Loading merged state dict into model...")
# Apply the changes
multimodal_model.load_state_dict(state_dict_multimodal)

print("Saving merged model...")
# Create save_path if it does not exist
os.makedirs(save_path, exist_ok=True)
multimodal_model.save_pretrained(save_path, safe_serialization=True, max_shard_size="8192MB")
multimodal_processor.save_pretrained(save_path)
print(f"Model saved to {save_path}")

# Update config to reflect the merge
config_path = os.path.join(save_path, "config.json")
if os.path.exists(config_path):
    import json
    with open(config_path) as f:
        config = json.load(f)

    # Add merge information to config
    config["_merge_info"] = {
        "base_vision_model": multimodal_model_path,
        "merged_text_model": text_model_path,
        "merge_method": "vision_adapter_merge",
        "replaced_parameters": replaced_count,
        "preserved_vision_parameters": len(vision_specific_keys)
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("Updated config with merge information")

print("Merge completed successfully!")
