import os
import argparse
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import DatasetMapper
from detectron2.data import transforms as T
from detectron2 import model_zoo
import torch

# === Paths that rarely change ===
COCO_JSON_PATH = "coco_annotations.json"
SPLIT_DIR = "splits"
IMAGE_ROOT = "dataset_raw"

def register_split(name, split_txt_path):
    with open(split_txt_path, "r") as f:
        wanted = {os.path.basename(l.strip().replace("\\", "/")) for l in f if l.strip()}

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

# === Subclass DefaultTrainer to define custom augmentations and evaluator ===
class TrainerWithCustomAug(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        aug_list = []

        if cfg.INPUT.AUG.HORIZONTAL_FLIP:
            aug_list.append(T.RandomFlip(horizontal=True, vertical=False))
        if cfg.INPUT.AUG.VERTICAL_FLIP:
            aug_list.append(T.RandomFlip(horizontal=False, vertical=True))
        if cfg.INPUT.AUG.ROTATION_ANGLE > 0:
            angle = cfg.INPUT.AUG.ROTATION_ANGLE
            aug_list.append(T.RandomRotation(angle=[-angle, angle]))
        if cfg.INPUT.AUG.BRIGHTNESS != 0.0:
            brightness_factor = 1 + cfg.INPUT.AUG.BRIGHTNESS
            aug_list.append(T.RandomBrightness(1 / brightness_factor, brightness_factor))
        if cfg.INPUT.AUG.CONTRAST != 0.0:
            contrast_factor = 1 + cfg.INPUT.AUG.CONTRAST
            aug_list.append(T.RandomContrast(1 / contrast_factor, contrast_factor))
        if cfg.INPUT.AUG.SATURATION != 0.0:
            saturation_factor = 1 + cfg.INPUT.AUG.SATURATION
            aug_list.append(T.RandomSaturation(1 / saturation_factor, saturation_factor))
        if cfg.INPUT.AUG.CROP.ENABLED:
            aug_list.append(T.RandomCrop("relative_range", cfg.INPUT.AUG.CROP.RANGE))

        mapper = DatasetMapper(cfg, is_train=True, augmentations=aug_list)
        return build_detection_train_loader(cfg, mapper=mapper)

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
    # add custom keys
    cfg.INPUT.AUG = CN()
    cfg.INPUT.AUG.HORIZONTAL_FLIP = False
    cfg.INPUT.AUG.VERTICAL_FLIP = False
    cfg.INPUT.AUG.ROTATION_ANGLE = 0
    cfg.INPUT.AUG.BRIGHTNESS = 0.0
    cfg.INPUT.AUG.CONTRAST = 0.0
    cfg.INPUT.AUG.SATURATION = 0.0
    cfg.INPUT.AUG.CROP = CN()
    cfg.INPUT.AUG.CROP.ENABLED = False
    cfg.INPUT.AUG.CROP.RANGE = [1.0, 1.0]
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = TrainerWithCustomAug(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()
