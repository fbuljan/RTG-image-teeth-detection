#!/bin/bash
set -e

apt update && apt install -y unzip git wget git-lfs

pip install gdown shapely absl-py termcolor yacs matplotlib cloudpickle tqdm
pip install opencv-python-headless
pip install 'git+https://github.com/facebookresearch/fvcore.git'
pip install 'git+https://github.com/facebookresearch/detectron2.git'

gdown https://drive.google.com/uc?id=1vyPxeKRD9YuUtjLfkdwaVmaPJlr-hoaR
unzip dataset_raw.zip

git clone https://github.com/fbuljan/RTG-image-teeth-detection.git
cd RTG-image-teeth-detection
mv ../dataset_raw .

git lfs install
git lfs pull
