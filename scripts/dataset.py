import os
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def collate_fn(batch):
    """
    batch: list of tuples:
      (image, bboxes, attr_labels, index_values, quad_values, age, sex)
    """
    images, bboxes, attr_labels, index_values, quad_values, ages, sexes = zip(*batch)

    images = torch.stack(images)

    bboxes = list(bboxes)
    attr_labels = list(attr_labels)
    index_values = list(index_values)
    quad_values = list(quad_values)

    ages = torch.tensor(ages, dtype=torch.float32)
    sexes = torch.tensor(sexes, dtype=torch.float32)

    return images, bboxes, attr_labels, index_values, quad_values, ages, sexes


class ToothDetectionDataset(Dataset):
    def __init__(self, xml_file: str, csv_file: str, images_folder: str, transform=None):
        self.images_folder = images_folder

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.attribute_mapping = {
            "ne iznikli zub": 0,
            "iznikli zub": 1,
            "zavrsen rast korijena": 2,
            "nezavrsen rast korijena": 3
        }

        self.metadata = pd.read_csv(csv_file, delimiter=';').set_index("image_name")
        self.annotations = self._parse_xml(xml_file)
        self.image_list = self._get_image_list()

    def _parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        annotations = {}

        for image in root.findall("image"):
            image_name = image.get("name")
            width = int(image.get("width"))
            height = int(image.get("height"))

            boxes = []
            for box in image.findall("box"):
                xtl = float(box.get("xtl"))
                ytl = float(box.get("ytl"))
                xbr = float(box.get("xbr"))
                ybr = float(box.get("ybr"))
                label = box.get("label")

                attributes = {}
                for attr in box.findall("attribute"):
                    attr_name = attr.get("name")
                    attr_value = attr.text.strip()
                    attributes[attr_name] = attr_value

                attr_values = torch.zeros(len(self.attribute_mapping))
                for attr_name, idx in self.attribute_mapping.items():
                    if attributes.get(attr_name, "").lower() == "true":
                        attr_values[idx] = 1.0

                box_index = int(attributes.get("index", 0))
                box_quad  = int(attributes.get("quad", 0))

                boxes.append({
                    "label": label,
                    "bbox": (xtl, ytl, xbr, ybr),
                    "attributes": attr_values,
                    "index": box_index,
                    "quad": box_quad
                })

            annotations[image_name] = {
                "width": width,
                "height": height,
                "boxes": boxes
            }

        return annotations

    def _get_image_list(self):
        valid_imgs = []
        for img in os.listdir(self.images_folder):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                if (img in self.metadata.index) and (img in self.annotations):
                    valid_imgs.append(img)
        return valid_imgs

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.images_folder, image_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        annotation = self.annotations.get(image_name, {"boxes": []})
        width = annotation.get("width", 1)
        height = annotation.get("height", 1)

        metadata_row = self.metadata.loc[image_name] \
            if image_name in self.metadata.index else None

        bboxes = []
        attr_labels = []
        index_values = []
        quad_values = []

        for box in annotation["boxes"]:
            xtl, ytl, xbr, ybr = box["bbox"]

            x_center = ((xtl + xbr) / 2) / width
            y_center = ((ytl + ybr) / 2) / height
            w_norm = (xbr - xtl) / width
            h_norm = (ybr - ytl) / height

            bboxes.append([0, x_center, y_center, w_norm, h_norm])
            attr_labels.append(box["attributes"])
            index_values.append(box["index"])
            quad_values.append(box["quad"])

        if len(bboxes) > 0:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            attr_labels = torch.stack(attr_labels)
            index_values = torch.tensor(index_values, dtype=torch.long)
            quad_values = torch.tensor(quad_values, dtype=torch.long)
        else:
            bboxes = torch.empty((0, 5), dtype=torch.float32)
            attr_labels = torch.empty((0, len(self.attribute_mapping)), dtype=torch.float32)
            index_values = torch.empty((0,), dtype=torch.long)
            quad_values = torch.empty((0,), dtype=torch.long)

        if metadata_row is not None:
            age = torch.tensor(metadata_row['age'], dtype=torch.float32)
            sex_val = str(metadata_row['sex']).lower()
            sex = torch.tensor(1.0 if sex_val == 'male' else 0.0, dtype=torch.float32)
        else:
            age = torch.tensor(0.0, dtype=torch.float32)
            sex = torch.tensor(0.0, dtype=torch.float32)

        return (image, bboxes, attr_labels, index_values, quad_values, age, sex)
