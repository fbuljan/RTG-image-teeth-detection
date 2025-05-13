import os
import argparse
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo
import torch

# === Paths that rarely change ===
COCO_JSON_PATH = "coco_annotations.json"
SPLIT_DIR      = "splits"
IMAGE_ROOT     = "dataset_raw"

def register_split(name, split_txt_path):
    with open(split_txt_path, "r") as f:
        wanted = {os.path.basename(l.strip()) for l in f if l.strip()}

    def loader():
        dicts = load_coco_json(
            COCO_JSON_PATH, IMAGE_ROOT,
            dataset_name=name,
            extra_annotation_keys=["segmentation"],
        )
        return [d for d in dicts if os.path.basename(d["file_name"]) in wanted]

    DatasetCatalog.register(name, loader)
    MetadataCatalog.get(name).set(
        thing_classes=["tooth"],
        evaluator_type="coco",
        image_root=IMAGE_ROOT,
    )

def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        help="path to config yaml",
        required=True
    )
    args = parser.parse_args()

    # 1) Register datasets
    register_split("teeth_train", os.path.join(SPLIT_DIR, "train.txt"))
    register_split("teeth_val",   os.path.join(SPLIT_DIR, "val.txt"))

    # 2) Build config from YAML
    model_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(model_config)
    )
    cfg.merge_from_file(args.config_file)

    # 3) Device override (optional)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 4) Make sure output dir exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 5) Train
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # 6) Final evaluation
    evaluator  = COCOEvaluator("teeth_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "teeth_val")
    print("\n🧪 Running validation…")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(results)

if __name__ == "__main__":
    main()
