#!/usr/bin/env python
"""Download WaterGen stage1/stage2 weights from Hugging Face."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download WaterGen checkpoints from Hugging Face")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="JiayiWuLeo/WaterGen",
        help="Hugging Face repo id",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Local directory to store stage1/ and stage2/",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(output_dir),
        allow_patterns=["stage1/*", "stage2/*"],
    )
    print(f"Downloaded weights to {output_dir.resolve()}")
    print(f"  Stage 1 LoRA: {output_dir / 'stage1'}")
    print(f"  Stage 2 decoder: {output_dir / 'stage2' / 'model.pth'}")


if __name__ == "__main__":
    main()
