"""Render annotated overlays on the panoramic for the demo UI.

Two overlays per query:
- yolo_overlay.png: bboxes drawn on the panoramic
- fdi_overlay.png:  bboxes + FDI numbers drawn on the panoramic

Output is written next to the per-query temp directory so the FastAPI
static file route can serve it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

# Each FDI gets a distinct color from this 16-color cycle, picked by FDI % 16.
# Picked to be readable on grayscale dental X-rays.
PALETTE = [
    (231, 76, 60), (46, 204, 113), (52, 152, 219), (241, 196, 15),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (52, 73, 94),
    (192, 57, 43), (39, 174, 96), (41, 128, 185), (243, 156, 18),
    (142, 68, 173), (22, 160, 133), (211, 84, 0), (44, 62, 80),
]


def _load_font(size: int) -> ImageFont.ImageFont:
    """Best-effort font load — fall back to PIL default."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _color_for(key: int) -> tuple[int, int, int]:
    return PALETTE[key % len(PALETTE)]


def render_yolo_overlay(
    panoramic_path: Path,
    bboxes: Iterable[tuple[float, float, float, float]],
    output_path: Path,
) -> None:
    """Save a copy of `panoramic_path` with each bbox drawn as a colored rectangle."""
    img = Image.open(panoramic_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    line_width = max(2, int(min(img.size) * 0.003))
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        color = _color_for(i)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def render_segmentation_overlay(
    panoramic_path: Path,
    polygons: Iterable[Iterable[tuple[float, float]]],
    output_path: Path,
) -> None:
    """Save a copy of `panoramic_path` with each tooth mask drawn as a colored contour."""
    img = Image.open(panoramic_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    line_width = max(3, int(min(img.size) * 0.005))
    for i, polygon in enumerate(polygons):
        color = _color_for(i)
        pts = [(float(x), float(y)) for x, y in polygon]
        if len(pts) < 3:
            continue
        # Close the polygon by repeating the first point.
        draw.line(pts + [pts[0]], fill=color, width=line_width, joint="curve")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def render_fdi_overlay(
    panoramic_path: Path,
    bboxes: Iterable[tuple[float, float, float, float]],
    fdi_labels: Iterable[str | int],
    output_path: Path,
    polygons: Iterable[Iterable[tuple[float, float]]] | None = None,
) -> None:
    """Save a copy of `panoramic_path` with each tooth outline + FDI label.

    If `polygons` is provided, each tooth is outlined by its mask contour
    (matching the segmentation overlay style). Otherwise rectangles are drawn
    from `bboxes` (detection mode).
    """
    img = Image.open(panoramic_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    line_width = max(3, int(min(img.size) * 0.005))
    # Floor/ceiling so tiny crops stay readable and giant panoramics don't get
    # billboard-sized labels. ~38px on a typical 1300x2500 panoramic.
    font_size = max(28, min(72, int(min(img.size) * 0.038)))
    font = _load_font(font_size)

    bbox_list = list(bboxes)
    label_list = list(fdi_labels)
    polygon_list = list(polygons) if polygons is not None else None

    for i, ((x1, y1, x2, y2), fdi) in enumerate(zip(bbox_list, label_list)):
        color = _color_for(int(fdi) if str(fdi).isdigit() else hash(str(fdi)) & 0xFF)

        if polygon_list is not None and i < len(polygon_list):
            pts = [(float(x), float(y)) for x, y in polygon_list[i]]
            if len(pts) >= 3:
                draw.line(pts + [pts[0]], fill=color, width=line_width, joint="curve")
            else:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = str(fdi)
        try:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_w, text_h = font.getsize(label)

        pad = 4
        bg_x1 = x1
        bg_y1 = max(0.0, y1 - text_h - pad * 2)
        bg_x2 = bg_x1 + text_w + pad * 2
        bg_y2 = bg_y1 + text_h + pad * 2
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color)
        draw.text((bg_x1 + pad, bg_y1 + pad), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
