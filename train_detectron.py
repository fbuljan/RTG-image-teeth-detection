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
SPLIT_DIR = "splits"
IMAGE_ROOT = "dataset_raw"

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

# === Subclass DefaultTrainer to define build_evaluator ===
class TrainerWithEvaluator(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", help="path to config yaml", required=True)
    args = parser.parse_args()

    config_basename = os.path.splitext(os.path.basename(args.config_file))[0]
    output_dir = os.path.join("detectron2_output", config_basename)

    register_split("teeth_train", os.path.join(SPLIT_DIR, "train.txt"))
    register_split("teeth_val", os.path.join(SPLIT_DIR, "val.txt"))

    model_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = TrainerWithEvaluator(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator("teeth_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "teeth_val")
    print("\n🧪 Running validation…")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(results)

if __name__ == "__main__":
    main()
