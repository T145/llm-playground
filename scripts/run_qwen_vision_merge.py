#!/usr/bin/env python3
"""
Runner script for merging Qwen3-30B text model into Qwen2.5-VL vision model.
Provides both custom Python implementation and mergekit YAML approaches.
"""

import argparse
import sys
from pathlib import Path


def run_custom_merge():
    """Run the custom Python implementation for vision adapter merging."""
    print("Running custom vision adapter merge...")
    print("This approach handles architectural differences between vision and text models.")

    # Execute the custom merge script
    import subprocess
    script_path = Path(__file__).parent / "vision" / "merge_qwen_vision_adapters.py"
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)

    if result.returncode == 0:
        print("Custom merge completed successfully!")
        print(result.stdout)
    else:
        print(f"Custom merge failed with return code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError("Custom merge failed")

def run_mergekit_yaml():
    """Run mergekit with the YAML configuration."""
    print("Running mergekit with YAML configuration...")
    print("Note: This may not work due to architectural differences between models.")

    import yaml
    from mergekit.config import MergeConfiguration
    from mergekit.merge import run_merge
    from mergekit.options import MergeOptions

    config_path = Path(__file__).parent.parent / "configs" / "qwen_vision_merge.yml"
    output_path = "./models/Qwen3-30B-A3B-Vision-Mergekit"

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    merge_config = MergeConfiguration.model_validate(config_data)
    options = MergeOptions(
        cuda=True,
        copy_tokenizer=True,
        allow_crimes=True,  # Allow experimental merges
        out_shard_size=8192,
        lazy_unpickle=True
    )

    try:
        run_merge(merge_config, output_path, options)
        print(f"Mergekit merge completed! Output saved to: {output_path}")
    except Exception as e:
        print(f"Mergekit merge failed: {e}")
        print("This is expected due to architectural differences. Use the custom approach instead.")

def main():
    parser = argparse.ArgumentParser(description="Merge Qwen3-30B text model into Qwen2.5-VL vision model")
    parser.add_argument(
        "--method",
        choices=["custom", "mergekit", "both"],
        default="custom",
        help="Merge method to use (default: custom)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually running the merge"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN MODE - No actual merging will be performed")
        print(f"Selected method: {args.method}")
        print("\nCustom method:")
        print("  - Loads both models into memory")
        print("  - Preserves vision components from Qwen2.5-VL-7B-Instruct")
        print("  - Replaces text components with Qwen3-30B-A3B weights")
        print("  - Handles layer mapping and dimension mismatches")
        print("  - Output: ./models/Qwen3-30B-A3B-Vision")
        print("\nMergekit method:")
        print("  - Uses mergekit YAML configuration")
        print("  - May fail due to architectural differences")
        print("  - Output: ./models/Qwen3-30B-A3B-Vision-Mergekit")
        return

    print("=== Qwen Vision Model Merge ===")
    print("Source vision model: Qwen/Qwen2.5-VL-7B-Instruct")
    print("Source text model: Qwen/Qwen3-30B-A3B")
    print(f"Method: {args.method}")
    print()

    if args.method in ["custom", "both"]:
        try:
            run_custom_merge()
        except Exception as e:
            print(f"Custom merge failed: {e}")
            if args.method == "custom":
                sys.exit(1)

    if args.method in ["mergekit", "both"]:
        try:
            run_mergekit_yaml()
        except Exception as e:
            print(f"Mergekit merge failed: {e}")
            if args.method == "mergekit":
                sys.exit(1)

    print("\n=== Merge Process Complete ===")
    print("Check the output directories for the merged models.")

if __name__ == "__main__":
    main()
