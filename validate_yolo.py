#!/usr/bin/env python3
"""
Standalone validation script for YOLO models.
Use this to run validation at specific checkpoints (e.g., epoch 25, 50).

Usage:
    python validate_yolo.py --config configs/yolo/seg-enhanced-1.yaml --weights runs-segmentation/experiment_optimized/weights/last.pt
"""

import argparse
import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights file (e.g., best.pt or last.pt)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loading model from: {args.weights}")
    model = YOLO(args.weights)

    print("Running validation...")
    val_results = model.val(
        data=config["data"],
        split=config.get("val_split", "val"),
        save=True,
        **config.get("val_args", {})
    )

    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    print(f"Box mAP50:       {val_results.box.map50:.4f}")
    print(f"Box mAP50-95:    {val_results.box.map:.4f}")
    print(f"Box Precision:   {val_results.box.p:.4f}")
    print(f"Box Recall:      {val_results.box.r:.4f}")
    print()
    print(f"Mask mAP50:      {val_results.seg.map50:.4f}")
    print(f"Mask mAP50-95:   {val_results.seg.map:.4f}")
    print(f"Mask Precision:  {val_results.seg.p:.4f}")
    print(f"Mask Recall:     {val_results.seg.r:.4f}")
    print("="*50)
    print()


if __name__ == "__main__":
    main()
