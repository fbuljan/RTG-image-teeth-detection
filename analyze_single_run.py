import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

def read_metrics(metrics_path: Path) -> pd.DataFrame:
    df = pd.read_json(metrics_path, lines=True)
    # Drop rows with NaN in key columns
    df = df.dropna(subset=['total_loss', 'lr'])
    return df

def parse_validation(validation_path: Path) -> dict:
    metrics = {}
    lines = validation_path.read_text().splitlines()
    current = None
    for line in lines:
        if 'Evaluate annotation type *bbox*' in line:
            current = 'bbox'
        elif 'Evaluate annotation type *segm*' in line:
            current = 'segm'
        if current and ('Average Precision' in line or 'Average Recall' in line):
            # Extract area (all/small/medium/large)
            area_match = re.search(r'area=\s*([a-z]+)', line)
            iou_match = re.search(r'IoU=([0-9.]+:?[0-9.]*)', line)
            nums = re.findall(r'=\s*([0-9]*\.[0-9]+)', line)
            if area_match and iou_match and nums:
                value = float(nums[-1])  # take last number
                prefix = 'AP' if 'Average Precision' in line else 'AR'
                key = f"{current}_{prefix}_{iou_match.group(1).replace(':','to')}_{area_match.group(1)}"
                metrics[key] = value
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Analyze single-run Detectron2 metrics")
    parser.add_argument("run_dir", type=Path,
                        help="Folder with metrics.json and validation.txt")
    args = parser.parse_args()

    run_dir = args.run_dir
    metrics_file = run_dir / "metrics.json"
    validation_file = run_dir / "validation.txt"
    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Read and clean training metrics
    if metrics_file.exists():
        df = read_metrics(metrics_file)

        # Plot and save figures
        def save_plot(x, y, xlabel, ylabel, title, fname):
            plt.figure()
            plt.plot(x, y)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.savefig(results_dir / fname, bbox_inches='tight')
            plt.close()

        save_plot(df['iteration'], df['total_loss'],
                  'Iteration', 'Total Loss', 'Training Loss Curve',
                  'loss_curve.png')
        save_plot(df['iteration'], df['lr'],
                  'Iteration', 'Learning Rate', 'LR Schedule',
                  'lr_schedule.png')
        if 'loss_mask' in df:
            save_plot(df['iteration'], df['loss_mask'],
                      'Iteration', 'Mask Loss', 'Mask Loss Curve',
                      'mask_loss_curve.png')
        if 'mask_rcnn/accuracy' in df:
            save_plot(df['iteration'], df['mask_rcnn/accuracy'],
                      'Iteration', 'Mask Accuracy', 'Mask R-CNN Accuracy',
                      'mask_accuracy.png')

    # Parse validation metrics
    validation_metrics = {}
    if validation_file.exists():
        validation_metrics = parse_validation(validation_file)
        (results_dir / "validation_raw.txt").write_text(validation_file.read_text())
        with open(results_dir / "validation_summary.txt", 'w') as f:
            for k, v in sorted(validation_metrics.items()):
                f.write(f"{k}: {v}\n")

    # Write results summary
    with open(results_dir / "results.txt", 'w') as f:
        f.write("=== Training Metrics ===\n")
        if metrics_file.exists():
            f.write(f"Min total_loss: {df['total_loss'].min():.4f}\n")
            f.write(f"Final total_loss: {df['total_loss'].iloc[-1]:.4f}\n")
            if 'loss_mask' in df:
                f.write(f"Min mask_loss: {df['loss_mask'].min():.4f}\n")
                f.write(f"Final mask_loss: {df['loss_mask'].iloc[-1]:.4f}\n")
            if 'mask_rcnn/accuracy' in df:
                f.write(f"Max mask_accuracy: {df['mask_rcnn/accuracy'].max():.4f}\n")
                f.write(f"Final mask_accuracy: {df['mask_rcnn/accuracy'].iloc[-1]:.4f}\n")
        if validation_metrics:
            f.write("\n=== Validation Metrics ===\n")
            for k, v in sorted(validation_metrics.items()):
                f.write(f"{k}: {v}\n")

    print(f"Results saved in: {results_dir}")

if __name__ == "__main__":
    main()


