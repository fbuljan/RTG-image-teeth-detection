"""Extract rotation-canonicalised tooth crops from dataset_raw/.

Mirrors extract_crops_gt.py but rotates each tooth so its principal axis (the
longer side of the polygon's minimum-area rectangle) is VERTICAL. The
180° ambiguity is resolved geometrically using the centroid's vertical
position inside the rotated bbox: we want the polygon's bulk to sit in
the lower half (i.e. crown wider at bottom, root narrower at top).
This is a pure geometric rule — no FDI label needed — so the same rule
applies at training time (per-FDI mask polygons) and at inference time
(YOLO polygons).

Outputs the same on-disk layout as extract_crops_gt.py so build_manifest.py
can be reused:
  identification/data/crops_gt_rotnorm/{image_id}/{FDI}_raw.png
  identification/data/crops_gt_rotnorm/{image_id}/{FDI}_masked.png
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

_LOWER_RED1 = np.array([0, 70, 50])
_UPPER_RED1 = np.array([10, 255, 255])
_LOWER_RED2 = np.array([170, 70, 50])
_UPPER_RED2 = np.array([180, 255, 255])


def detect_red_mask(mask_bgr):
    hsv = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, _LOWER_RED1, _UPPER_RED1)
    m2 = cv2.inRange(hsv, _LOWER_RED2, _UPPER_RED2)
    return cv2.bitwise_or(m1, m2)


def canonical_rotation_deg(
    polygon: np.ndarray,
    panoramic_h: int | None = None,
) -> float:
    """Return the rotation angle (degrees, image-coord convention used by
    cv2.getRotationMatrix2D / cv2.warpAffine) that brings the polygon's
    principal axis vertical with its wider end (crown) at the bottom of the
    canonical frame.

    Algorithm
    ---------
    1. Find the principal axis as the long edge of cv2.boxPoints(minAreaRect).
       This avoids the w-vs-h branch that produced a 90° discontinuity when
       polygons drifted across the aspect-ratio = 1 boundary.
    2. Rotate the polygon into the candidate canonical frame USING THE SAME
       image-coord rotation that cv2.warpAffine will later apply to the
       panoramic. This guarantees the half-area disambiguator inspects the
       same orientation the embedder will see.
    3. Compare wider-half area; if the wider end is on top, flip 180°.
    4. (Optional fallback) When the polygon is too close to area-symmetric
       (|delta| < 5%) AND `panoramic_h` is provided, use the polygon centroid's
       Y position within the panoramic to pick the orientation: upper-arch
       teeth (centroid above midline) get root pointing up; lower-arch teeth
       get root pointing down. This is impure (uses the centroid as a proxy
       for arch) but only kicks in when geometry is unreliable.
    """
    if polygon.dtype != np.float32:
        polygon = polygon.astype(np.float32)
    pts = polygon.reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)
    (cx, cy), _, _ = rect

    # --- BLOCKER 2 fix: long-edge direction, no w/h branch ---
    box = cv2.boxPoints(rect)  # 4×2, ordered around the rectangle
    edges = np.diff(np.vstack([box, box[:1]]), axis=0)  # 4×2
    edge_lengths = np.linalg.norm(edges, axis=1)
    long_edge_idx = int(np.argmax(edge_lengths))
    dx, dy = edges[long_edge_idx]
    # We need an angle `rot` such that cv2.warpAffine(getRotationMatrix2D(c, rot, 1))
    # transforms the long-edge direction (dx, dy) into a vertical vector (0, ±L).
    # The image-coord rotation matrix M_rot = [[cos r, sin r], [-sin r, cos r]]
    # acts on a direction (dx, dy) as:
    #   dx' = cos(r)*dx + sin(r)*dy
    #   dy' = -sin(r)*dx + cos(r)*dy
    # Setting dx' = 0 gives tan(r) = -dx/dy, i.e. r = atan2(-dx, dy).
    rot = float(np.degrees(np.arctan2(-dx, dy)))
    # Normalise to (-180, 180]
    rot = ((rot + 180.0) % 360.0) - 180.0

    # --- BLOCKER 1 fix: rotate polygon in IMAGE-COORD frame ---
    # cv2.getRotationMatrix2D((cx,cy), deg, 1.0) builds:
    #   [[ cos(deg),  sin(deg), tx],
    #    [-sin(deg),  cos(deg), ty]]
    # So a polygon point p is mapped to R @ (p - center) + center where
    # R = [[c, s], [-s, c]] (this is the image-coord rotation matrix).
    theta = np.deg2rad(rot)
    c, s = np.cos(theta), np.sin(theta)
    centered = polygon - np.array([cx, cy], dtype=np.float32)
    rotated = np.stack([
        c * centered[:, 0] + s * centered[:, 1],   # x'
        -s * centered[:, 0] + c * centered[:, 1],  # y'  (image-coord)
    ], axis=1)

    # --- HIGH #2 fix: arch convention for the 180° decision (always when
    # panoramic_h is known). Empirically (audit's parity test on the baseline
    # YOLO+GT polygons) using the arch convention always — rather than only as
    # a fallback when half-area is ambiguous — reduces the GT-vs-YOLO
    # disagreement rate at |Δrot|>90° from 16% to 6.7%, and at |Δrot|>170°
    # (180° flips) from 10% to 4.3%. The principal-axis estimate is shared, so
    # this only changes the flip decision: making it depend on the polygon's
    # OVERALL position in the panoramic (above/below midline) instead of its
    # SHAPE (which subtle YOLO mask differences perturb).
    #
    # In a dental panoramic the maxilla is above the midline and the mandible
    # is below. For an upper-arch tooth, the crown points DOWN (toward the
    # midline); for a lower-arch tooth, the crown points UP. We want canonical
    # orientation with crown at the bottom of the frame for ALL teeth.
    if panoramic_h is not None:
        centroid_y_pano = float(np.mean(polygon[:, 1]))
        upper_arch = centroid_y_pano < (panoramic_h / 2.0)
        # Identify crown half of the ORIGINAL polygon by panoramic-Y:
        #   upper-arch: crown is the lower half of the polygon (largest y)
        #   lower-arch: crown is the upper half of the polygon (smallest y)
        orig_y = polygon[:, 1]
        if upper_arch:
            crown_mask = orig_y > np.median(orig_y)
        else:
            crown_mask = orig_y < np.median(orig_y)
        if crown_mask.any():
            crown_y_in_canon = rotated[crown_mask, 1].mean()
            other_y_in_canon = rotated[~crown_mask, 1].mean()
            # Want crown to land at the BOTTOM of the canonical frame
            # (largest y in canon). If it landed at the top, flip 180°.
            if crown_y_in_canon < other_y_in_canon:
                rot += 180.0
                rot = ((rot + 180.0) % 360.0) - 180.0
        return float(rot)

    # No panoramic_h available — fall back to half-area heuristic.
    rotated_norm = rotated - rotated.min(axis=0)
    bbox_w = max(1, int(np.ceil(rotated_norm[:, 0].max())))
    bbox_h = max(1, int(np.ceil(rotated_norm[:, 1].max())))
    canvas = np.zeros((bbox_h + 1, bbox_w + 1), dtype=np.uint8)
    cv2.fillPoly(canvas, [rotated_norm.astype(np.int32)], 1)
    mid = (bbox_h + 1) // 2
    area_top = int(canvas[:mid].sum())
    area_bottom = int(canvas[mid:].sum())
    if area_bottom < area_top:
        rot += 180.0
        rot = ((rot + 180.0) % 360.0) - 180.0
    return float(rot)


def rotate_and_crop(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    polygon: np.ndarray,
    rotation_deg: float,
    padding_ratio: float = 0.1,
):
    """Rotate the panoramic + mask about the polygon centroid by `rotation_deg`,
    then crop the polygon's tight bbox in the rotated frame with `padding_ratio`.

    Returns (raw_crop, masked_crop) in BGR, both same arbitrary HxW (caller
    resizes to 224x224 with padding).
    """
    h, w = image_bgr.shape[:2]
    pts = polygon.astype(np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)
    (cx, cy), _, _ = rect

    # Rotate full image about polygon centroid; the resulting frame stays the
    # same canvas size (warp truncates the corners that would have spilled out,
    # which is fine because we only care about the area around the polygon).
    M = cv2.getRotationMatrix2D((cx, cy), rotation_deg, 1.0)
    rotated_img = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    rotated_mask = cv2.warpAffine(binary_mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Polygon in rotated frame
    polygon_h = np.hstack([polygon, np.ones((len(polygon), 1))]).astype(np.float32)
    rotated_polygon = (M @ polygon_h.T).T  # (N, 2)

    x_min, y_min = rotated_polygon.min(axis=0)
    x_max, y_max = rotated_polygon.max(axis=0)
    bw = x_max - x_min
    bh = y_max - y_min
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    x1 = max(0, int(round(x_min - pad_x)))
    y1 = max(0, int(round(y_min - pad_y)))
    x2 = min(w, int(round(x_max + pad_x)))
    y2 = min(h, int(round(y_max + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return None, None
    raw_crop = rotated_img[y1:y2, x1:x2]
    mask_region = rotated_mask[y1:y2, x1:x2]
    masked_crop = raw_crop.copy()
    masked_crop[mask_region == 0] = 0
    return raw_crop, masked_crop


def resize_with_padding(image, target_size=224):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def process_folder(folder_path, output_dir, target_size=224, padding_ratio=0.1):
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    main_image_path = folder_path / f"{folder_name}.png"
    if not main_image_path.exists():
        return 0, 0, [f"No main image: {main_image_path}"]

    mask_files = sorted(folder_path.glob(f"{folder_name}+*.png"))
    if not mask_files:
        return 0, 0, [f"No mask files in {folder_path}"]

    main_img = cv2.imread(str(main_image_path))
    if main_img is None:
        return 0, 0, [f"Failed to read: {main_image_path}"]

    out_dir = Path(output_dir) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_err = 0, 0
    errors = []
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]

    clean_fdis = set()
    for mp in mask_files:
        m = re.search(r'\+(\d+)\.png$', mp.name)
        if m:
            clean_fdis.add(m.group(1))

    for mask_path in mask_files:
        match = re.search(r'\+(\d+)(?:\s*(?:#\d+)?\s*)\.png$', mask_path.name)
        if not match:
            n_err += 1
            errors.append(f"{folder_name}: Bad mask filename: {mask_path.name}")
            continue
        fdi = match.group(1)

        is_hash_variant = '#' in mask_path.name or re.search(r'\+\d+\s+\.png$', mask_path.name)
        if is_hash_variant and fdi in clean_fdis:
            continue
        if (out_dir / f"{fdi}_raw.png").exists():
            continue

        mask_bgr = cv2.imread(str(mask_path))
        if mask_bgr is None:
            n_err += 1
            errors.append(f"{folder_name} FDI={fdi}: Failed to read mask")
            continue

        binary_mask = detect_red_mask(mask_bgr)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            n_err += 1
            errors.append(f"{folder_name} FDI={fdi}: No red region found")
            continue
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) < 30:
            n_err += 1
            errors.append(f"{folder_name} FDI={fdi}: Polygon too small")
            continue

        poly = biggest.squeeze()
        if poly.ndim != 2 or poly.shape[0] < 3:
            n_err += 1
            errors.append(f"{folder_name} FDI={fdi}: Degenerate polygon")
            continue
        poly = poly.astype(np.float32)

        rot = canonical_rotation_deg(poly, panoramic_h=main_img.shape[0])
        raw_crop, masked_crop = rotate_and_crop(
            main_img, binary_mask, poly, rot, padding_ratio=padding_ratio,
        )
        if raw_crop is None or raw_crop.size == 0:
            n_err += 1
            errors.append(f"{folder_name} FDI={fdi}: empty crop after rotation")
            continue
        if raw_crop.shape[0] < 5 or raw_crop.shape[1] < 5:
            n_err += 1
            errors.append(f"{folder_name} FDI={fdi}: Crop too small {raw_crop.shape}")
            continue

        raw_resized = resize_with_padding(raw_crop, target_size)
        masked_resized = resize_with_padding(masked_crop, target_size)
        cv2.imwrite(str(out_dir / f"{fdi}_raw.png"), raw_resized, png_params)
        cv2.imwrite(str(out_dir / f"{fdi}_masked.png"), masked_resized, png_params)
        n_ok += 1

    return n_ok, n_err, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="dataset_raw")
    parser.add_argument("--output-dir", default="identification/data/crops_gt_rotnorm")
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--padding-ratio", type=float, default=0.1)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    folders = sorted([f for f in input_dir.iterdir() if f.is_dir()])
    print(f"Found {len(folders)} folders in {input_dir}", flush=True)

    output_dir = Path(args.output_dir)
    existing = set(d.name for d in output_dir.iterdir() if d.is_dir()) if output_dir.exists() else set()
    to_process = [f for f in folders if f.name not in existing]
    print(f"Already processed: {len(existing)}, remaining: {len(to_process)}", flush=True)

    total_crops = 0
    total_errors = 0
    error_log = []
    for i, folder in enumerate(to_process):
        n_ok, n_err, errs = process_folder(folder, args.output_dir, args.target_size, args.padding_ratio)
        total_crops += n_ok
        total_errors += n_err
        error_log.extend(errs)
        if (i + 1) % 50 == 0 or i == len(to_process) - 1:
            print(f"  {i+1}/{len(to_process)} folders, {total_crops} crops", flush=True)

    print(f"\nDone. Extracted {total_crops} crops, {total_errors} errors.")
    if error_log:
        log_path = Path(args.output_dir) / "extraction_errors.log"
        with open(log_path, "w") as f:
            f.write("\n".join(error_log))
        print(f"Error log: {log_path}")


if __name__ == "__main__":
    main()
