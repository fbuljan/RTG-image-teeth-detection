import os
import xml.etree.ElementTree as ET
import argparse

def convert_annotations(xml_file, dataset_root):
    images_dir = os.path.join(dataset_root, "images")
    subsets = ["train", "val", "test"]

    labels_dir = os.path.join(dataset_root, "labels")
    for subset in subsets:
        subset_label_dir = os.path.join(labels_dir, subset)
        os.makedirs(subset_label_dir, exist_ok=True)

    tree = ET.parse(xml_file)
    root = tree.getroot()

    for image_elem in root.findall('image'):
        image_name = image_elem.get('name')
        width = float(image_elem.get('width'))
        height = float(image_elem.get('height'))

        subset_found = None
        for subset in subsets:
            img_path = os.path.join(images_dir, subset, image_name)
            if os.path.exists(img_path):
                subset_found = subset
                break
        if subset_found is None:
            print(f"WARNING: Image {image_name} not found in any subset (train/val/test). Preskačem.")
            continue

        lines = []
        for box in image_elem.findall('box'):
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            x_center = ((xtl + xbr) / 2.0) / width
            y_center = ((ytl + ybr) / 2.0) / height
            bbox_width = (xbr - xtl) / width
            bbox_height = (ybr - ytl) / height
            line = f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
            lines.append(line)
        
        label_filename = os.path.splitext(image_name)[0] + ".txt"
        label_filepath = os.path.join(labels_dir, subset_found, label_filename)
        with open(label_filepath, "w") as f:
            f.write("\n".join(lines))
        print(f"Processed {image_name} in subset '{subset_found}': {len(lines)} objects.")

def main():
    parser = argparse.ArgumentParser(description="Convert XML annotations to YOLO-format labels for dataset.")
    parser.add_argument("--xml", type=str, required=True, help="Path to the XML annotations file (e.g., dataset/annotations.xml).")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to the dataset root directory (should contain 'images' folder).")
    args = parser.parse_args()

    convert_annotations(args.xml, args.dataset_root)

if __name__ == "__main__":
    main()
