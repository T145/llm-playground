# Qwen Vision Model Merging

This directory contains scripts for merging text models into Qwen vision models, specifically designed to merge `Qwen/Qwen3-30B-A3B` into `Qwen/Qwen2.5-VL-7B-Instruct`.

## Files

- `merge_qwen_vision_adapters.py` - Core merging script that handles the vision adapter merge
- `../run_qwen_vision_merge.py` - Runner script with multiple merge options
- `../configs/qwen_vision_merge.yml` - Mergekit YAML configuration (experimental)

## Usage

### Option 1: Direct Script Execution

```bash
cd scripts/vision
python merge_qwen_vision_adapters.py
```

### Option 2: Using the Runner Script

```bash
cd scripts
python run_qwen_vision_merge.py --method custom
```

Available methods:
- `custom` (recommended) - Uses the custom vision adapter merging approach
- `mergekit` - Uses mergekit with YAML config (may fail due to architectural differences)
- `both` - Tries both methods

### Option 3: Dry Run

To see what would be done without actually running the merge:

```bash
python run_qwen_vision_merge.py --dry-run
```

## How It Works

### Custom Vision Adapter Merge

The custom approach:

1. **Loads both models** into memory (requires sufficient RAM)
2. **Identifies vision components** in the multimodal model (visual encoders, cross-attention layers, etc.)
3. **Maps layers** between the 30B text model and 7B vision model by sampling layers
4. **Preserves vision components** while replacing text components
5. **Handles dimension mismatches** and vocabulary size differences
6. **Saves the merged model** with proper configuration

### Layer Mapping Strategy

Since we're merging a 30B model into a 7B vision model:
- The script samples layers from the larger text model
- Maps them to the corresponding positions in the vision model
- Preserves all vision-specific components (visual encoders, cross-attention, etc.)

## Requirements

- Sufficient RAM to load both models (recommended: 64GB+ for 30B model)
- PyTorch
- Transformers library
- Access to both source models

## Output

The merged model will be saved to:
- Custom method: `./models/Qwen3-30B-A3B-Vision/`
- Mergekit method: `./models/Qwen3-30B-A3B-Vision-Mergekit/`

## Notes

- This is an experimental approach for cross-architecture merging
- The mergekit YAML approach may not work due to architectural differences
- The custom approach is specifically designed for Qwen vision models
- Results may vary and should be thoroughly tested before use

## Troubleshooting

### Out of Memory
- Reduce batch size or use CPU-only mode
- Consider processing layer by layer (requires code modification)

### Shape Mismatches
- The script handles most common shape mismatches
- Check the console output for specific issues

### Model Loading Issues
- Ensure you have access to both source models
- Check your Hugging Face authentication if needed
