import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo
import torch

# === CONFIG ===
COCO_JSON_PATH = "coco_annotations.json"
SPLIT_DIR      = "splits"
OUTPUT_DIR     = "./detectron2_output"
EPOCHS         = 20
BATCH_SIZE     = 2
LEARNING_RATE  = 0.0015

def register_split(name, split_txt_path):
    # Load just the basenames you want in this split
    with open(split_txt_path, "r") as f:
        wanted = {os.path.basename(line.strip()) for line in f if line.strip()}

    image_root = "dataset_raw"  # your top‐level folder

    def loader():
        dicts = load_coco_json(
            COCO_JSON_PATH,
            image_root,
            dataset_name=name,
            extra_annotation_keys=["segmentation"],  # if you still need polygons
        )
        # Keep only records whose basename is in our split
        return [
            d for d in dicts
            if os.path.basename(d["file_name"]) in wanted
        ]

    DatasetCatalog.register(name, loader)
    MetadataCatalog.get(name).set(
        thing_classes=["tooth"],
        evaluator_type="coco",
        image_root=image_root,
    )

def main():
    # 1) Register train & val
    register_split("teeth_train", os.path.join(SPLIT_DIR, "train.txt"))
    register_split("teeth_val",   os.path.join(SPLIT_DIR, "val.txt"))

    # 2) Build config
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN       = ("teeth_train",)
    cfg.DATASETS.TEST        = ("teeth_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.MASK_FORMAT      = "polygon"

    cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
    cfg.SOLVER.BASE_LR       = LEARNING_RATE
    # calculates total iters from your train split size
    num_train = sum(1 for _ in open(os.path.join(SPLIT_DIR, "train.txt")))
    cfg.SOLVER.MAX_ITER     = (num_train // BATCH_SIZE) * EPOCHS
    cfg.SOLVER.STEPS        = []  # no LR decay
    cfg.OUTPUT_DIR          = OUTPUT_DIR
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES         = 1  # just “tooth”
    cfg.MODEL.DEVICE                        = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # 3) Evaluate
    evaluator  = COCOEvaluator("teeth_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "teeth_val")
    print("\n🧪 Running validation…")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(results)

if __name__ == "__main__":
    main()
