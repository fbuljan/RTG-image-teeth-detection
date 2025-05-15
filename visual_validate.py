import os
import argparse
import random
import torch
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger

setup_logger()

# Paths
COCO_JSON_PATH = "coco_annotations.json"
IMAGE_ROOT    = "dataset_raw"
SPLIT_TXT      = "splits/test.txt"


def register_test_dataset(name="teeth_test"):
    # Load filenames from split
    with open(SPLIT_TXT, "r") as f:
        test_basenames = {os.path.basename(line.strip()) for line in f if line.strip()}

    def loader():
        dicts = load_coco_json(
            COCO_JSON_PATH,
            IMAGE_ROOT,
            dataset_name=name,
            extra_annotation_keys=["segmentation"],
        )
        filtered = []
        for d in dicts:
            base = os.path.basename(d["file_name"])
            if base in test_basenames:
                # Normalize category_id to 0
                for ann in d.get("annotations", []):
                    ann["category_id"] = 0
                filtered.append(d)
        return filtered

    DatasetCatalog.register(name, loader)
    meta = MetadataCatalog.get(name)
    meta.set(
        thing_classes=["tooth"],
        evaluator_type="coco",
        image_root=IMAGE_ROOT,
    )

    data = DatasetCatalog.get(name)
    print(f"[DATASET] `{name}` contains {len(data)} images.")
    return name, meta


def create_side_by_side(gt_img, pred_img):
    return cv2.hconcat([gt_img, pred_img])


def main():
    parser = argparse.ArgumentParser(description="Detectron2 Visual Validation")
    parser.add_argument("--model-path", required=True, help="Path to model_final.pth")
    parser.add_argument("--score-thr", type=float, default=0.05,
                        help="Score threshold (use 0.0 for debug)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of random images to visualize")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save visualizations")
    args = parser.parse_args()

    # Config and weights
    model_dir = os.path.dirname(args.model_path)
    yaml_base = os.path.basename(model_dir) + ".yaml"
    yaml_path = os.path.join("configs", "detectron", yaml_base)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    print(f"[CONFIG] Loading base model config from: {yaml_path}")
    cfg = get_cfg()
    # 1. Load Mask R-CNN base config from model zoo
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    # 2. Merge custom overrides
    cfg.merge_from_file(yaml_path)
    print(f"[CONFIG] After merge: NUM_CLASSES={cfg.MODEL.ROI_HEADS.NUM_CLASSES}")

    cfg.MODEL.WEIGHTS = args.model_path
    cfg.MODEL.DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thr

    print(f"[MODEL] Weights     : {cfg.MODEL.WEIGHTS}")
    print(f"[MODEL] Device      : {cfg.MODEL.DEVICE}")
    print(f"[MODEL] NUM_CLASSES : {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    print(f"[MODEL] SCORE_THRESH: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")

    # Register dataset
    ds_name, metadata = register_test_dataset()
    cfg.DATASETS.TEST = (ds_name,)

    predictor = DefaultPredictor(cfg)

    # Sample images
    all_dicts = DatasetCatalog.get(ds_name)
    n = min(args.n_samples, len(all_dicts))
    print(f"[RUN] Visualizing {n} random samples from '{ds_name}'")
    samples = random.sample(all_dicts, n)

    out_dir = args.output_dir or os.path.join(model_dir, "visual_validation")
    os.makedirs(out_dir, exist_ok=True)

    for idx, d in enumerate(samples, start=1):
        img_path = d["file_name"]
        exists = os.path.exists(img_path)
        print(f"{idx:02d}) {os.path.basename(img_path)} | exists? {exists}")
        if not exists:
            print(f"    [ERROR] File missing: {img_path}")
            continue

        img = cv2.imread(img_path)
        # Ground truth visualization
        v_gt = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        gt_vis = v_gt.draw_dataset_dict(d).get_image()[:, :, ::-1]

        # Prediction
        outputs = predictor(img)
        print(f"    OUTPUT keys: {list(outputs.keys())}")
        inst = outputs["instances"].to("cpu")
        print(f"    Instances: {len(inst)}")
        if len(inst) == 0:
            print("    [WARN] No predictions above threshold.")

        v_pred = Visualizer(
            img[:, :, ::-1], metadata=metadata,
            scale=1.0, instance_mode=ColorMode.IMAGE_BW
        )
        pred_vis = v_pred.draw_instance_predictions(inst).get_image()[:, :, ::-1]

        # Combine and save
        combo = create_side_by_side(gt_vis, pred_vis)
        out_name = f"{idx:02d}_{os.path.splitext(os.path.basename(img_path))[0]}.jpg"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, combo)
        print(f"    [OK] Saved: {out_path}\n")

    print("[DONE] Visual validation complete.")

if __name__ == "__main__":
    main()
