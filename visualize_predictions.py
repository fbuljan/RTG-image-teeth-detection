import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def visualize_predictions(model_path: str, num_images: int = 5, split: str = "val", conf: float = 0.25, save_dir: str = None):
    """
    Load a trained YOLO model and visualize predictions on random images.

    Args:
        model_path: Path to the trained model weights (.pt file)
        num_images: Number of random images to visualize
        split: Dataset split to use ('train', 'val', or 'test')
        conf: Confidence threshold for predictions
        save_dir: Directory to save visualizations (optional)
    """
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Get dataset path from data.yaml
    data_yaml_path = Path("data.yaml")
    if not data_yaml_path.exists():
        print("Error: data.yaml not found in current directory")
        return

    import yaml
    with open(data_yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    # Get image directory for the split
    if split not in data_config:
        print(f"Error: Split '{split}' not found in data.yaml")
        return

    image_dir = Path(data_config[split])
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return

    # Get all images
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    all_images = []
    for ext in image_extensions:
        all_images.extend(image_dir.glob(f"*{ext}"))
        all_images.extend(image_dir.glob(f"*{ext.upper()}"))

    if not all_images:
        print(f"Error: No images found in {image_dir}")
        return

    # Select random images
    num_images = min(num_images, len(all_images))
    selected_images = random.sample(all_images, num_images)

    print(f"\nSelected {num_images} random images from '{split}' split:")
    for img in selected_images:
        print(f"  - {img.name}")
    print()

    # Setup save directory
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = Path("visualizations")
        save_path.mkdir(parents=True, exist_ok=True)

    # Run predictions and visualize
    for i, img_path in enumerate(selected_images):
        print(f"Processing {i+1}/{num_images}: {img_path.name}")

        # Run prediction
        results = model.predict(
            source=str(img_path),
            conf=conf,
            save=False,
            verbose=False
        )

        result = results[0]

        # Get original image
        orig_img = cv2.imread(str(img_path))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # Get annotated image from YOLO
        annotated_img = result.plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # Print detection info
        num_detections = len(result.boxes) if result.boxes is not None else 0
        print(f"  Detections: {num_detections}")

        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.cpu().numpy()
            print(f"  Confidence scores: {[f'{c:.3f}' for c in confs]}")

        # Create side-by-side comparison
        h, w = orig_img.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = orig_img
        comparison[:, w:] = cv2.resize(annotated_img, (w, h))

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, f"Predictions (conf>{conf})", (w + 10, 30), font, 1, (255, 255, 255), 2)

        # Save
        output_path = save_path / f"pred_{i+1}_{img_path.stem}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {output_path}")
        print()

    print(f"All visualizations saved to: {save_path}")
    print("\nSummary of predictions:")

    # Run on all selected images for summary
    total_detections = 0
    all_confs = []

    for img_path in selected_images:
        results = model.predict(source=str(img_path), conf=conf, save=False, verbose=False)
        result = results[0]
        if result.boxes is not None:
            total_detections += len(result.boxes)
            all_confs.extend(result.boxes.conf.cpu().numpy().tolist())

    print(f"  Total detections across {num_images} images: {total_detections}")
    print(f"  Average detections per image: {total_detections / num_images:.1f}")
    if all_confs:
        print(f"  Confidence range: {min(all_confs):.3f} - {max(all_confs):.3f}")
        print(f"  Average confidence: {np.mean(all_confs):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO model predictions on random images")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model weights (.pt file)")
    parser.add_argument("--num-images", type=int, default=5,
                        help="Number of random images to visualize (default: 5)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Dataset split to use (default: val)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save visualizations (default: ./visualizations)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    visualize_predictions(
        model_path=args.model,
        num_images=args.num_images,
        split=args.split,
        conf=args.conf,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
