import os
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ToothDetectionDataset(Dataset):
    def __init__(self, xml_file: str, csv_file: str, images_folder: str, transform=None):
        self.images_folder = images_folder
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.metadata = pd.read_csv(csv_file, delimiter=';')
        self.annotations = self._parse_xml(xml_file)
        
        self.attribute_mapping = {
            "ne iznikli zub": 0, "iznikli zub": 1,
            "zavrsen rast korijena": 2, "nezavrsen rast korijena": 3
        }

    def _parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        annotations = {}

        for image in root.findall("image"):
            image_name = image.get("name")
            width, height = int(image.get("width")), int(image.get("height"))
            boxes = []

            for box in image.findall("box"):
                xtl, ytl, xbr, ybr = float(box.get("xtl")), float(box.get("ytl")), float(box.get("xbr")), float(box.get("ybr"))
                label = box.get("label")

                attributes = {attr.get("name"): attr.text.strip() for attr in box.findall("attribute")}
                
                attr_values = torch.zeros(len(self.attribute_mapping))
                for attr_name, attr_value in attributes.items():
                    if attr_value.lower() == "true" and attr_name in self.attribute_mapping:
                        attr_values[self.attribute_mapping[attr_name]] = 1  # One-hot encoding

                boxes.append({
                    "label": label,
                    "bbox": (xtl, ytl, xbr, ybr),
                    "attributes": attr_values,
                    "index": int(attributes.get("index", 0)),
                    "quad": int(attributes.get("quad", 0))
                })

            annotations[image_name] = {"width": width, "height": height, "boxes": boxes}

        return annotations

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_name = row['image_name']
        image_path = os.path.join(self.images_folder, image_name)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        annotation = self.annotations.get(image_name, {"boxes": []})
        boxes = annotation["boxes"]

        bboxes = []
        attr_labels = []
        index_values = []
        quad_values = []

        for box in boxes:
            xtl, ytl, xbr, ybr = box["bbox"]
            width, height = annotation["width"], annotation["height"]

            x_center = ((xtl + xbr) / 2) / width
            y_center = ((ytl + ybr) / 2) / height
            box_width = (xbr - xtl) / width
            box_height = (ybr - ytl) / height

            bboxes.append([0, x_center, y_center, box_width, box_height])  
            attr_labels.append(box["attributes"])
            index_values.append(box["index"])
            quad_values.append(box["quad"])

        bboxes = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.empty((0, 5))
        attr_labels = torch.stack(attr_labels) if attr_labels else torch.empty((0, len(self.attribute_mapping)))
        index_values = torch.tensor(index_values, dtype=torch.long) if index_values else torch.empty((0,))
        quad_values = torch.tensor(quad_values, dtype=torch.long) if quad_values else torch.empty((0,))

        age = torch.tensor(row['age'], dtype=torch.float32)
        sex = 1 if row['sex'] == 'male' else 0

        return image, bboxes, attr_labels, index_values, quad_values, age, sex
