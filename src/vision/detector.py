# FILE: src/vision/detector.py
"""
Label-region detector for the Legal Metrology Checker.

What this file does
-------------------
1. detect_regions(image_bgr)  -> finds label regions (product_name, mrp, ...)
                                 on a pack photo using a YOLOv8 model.
2. draw_regions(image_bgr, regions) -> draws coloured, labelled boxes on a
                                 COPY of the photo (the original is untouched).

The trained model file must be at:  <repository root>/models/best.pt
If the file is missing, or anything goes wrong, detect_regions prints a short
warning and returns an empty list [] - it never crashes the app.
"""

from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Settings (kept in one place so they are easy to find)
# ---------------------------------------------------------------------------

# Repository root = two folders above this file (src/vision/detector.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Where the trained YOLOv8 weights must be placed.
MODEL_PATH = REPO_ROOT / "models" / "best.pt"

CONF_THRESHOLD = 0.25   # ignore detections the model is less than 25% sure about
IMAGE_SIZE = 960        # the model looks at the photo resized to 960 pixels
DEVICE = "cpu"          # run on the normal processor (no graphics card needed)
MAX_PER_LABEL = 2       # keep at most this many boxes for the same label

# One BGR colour (OpenCV uses Blue, Green, Red order) for each of the 6 labels.
LABEL_COLOURS = {
    "product_name": (255, 128, 0),         # blue
    "net_quantity": (0, 200, 0),           # green
    "mrp": (0, 0, 255),                    # red
    "date_block": (0, 200, 255),           # orange
    "manufacturer_block": (255, 0, 255),   # magenta
    "consumer_care_block": (200, 200, 0),  # teal
}

# The model is loaded the first time it is needed and then kept here,
# so we never load the (slow) model file twice.
_model = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _get_model():
    """Return the cached YOLO model, loading it from MODEL_PATH on first use.

    Returns None (and prints a warning) if the file is missing or cannot be
    loaded. A failure is NOT cached, so if you copy best.pt into models/
    later, the next call will pick it up.
    """
    global _model

    # Already loaded earlier? Reuse it.
    if _model is not None:
        return _model

    # File missing? Warn and give up quietly.
    if not MODEL_PATH.exists():
        print(f"[detector] WARNING: model file not found at {MODEL_PATH}")
        return None

    try:
        # Imported here (not at the top) so that importing this file is fast
        # and never fails, even if ultralytics has a problem.
        from ultralytics import YOLO

        _model = YOLO(str(MODEL_PATH))
        return _model
    except Exception as exc:  # noqa: BLE001 - we really want to catch everything
        print(f"[detector] WARNING: could not load model: {exc}")
        return None


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------

def _to_numpy(values):
    """Turn a torch tensor, numpy array or list into a float numpy array."""
    if hasattr(values, "cpu"):       # torch tensors have .cpu()
        values = values.cpu()
    if hasattr(values, "numpy"):     # ...and then .numpy()
        values = values.numpy()
    return np.asarray(values, dtype=float)


def _label_name(names, class_id):
    """Look up the label text for a class number, e.g. 2 -> 'mrp'."""
    try:
        if isinstance(names, (dict, list, tuple)):
            return str(names[class_id])
    except Exception:  # noqa: BLE001
        pass
    return f"class_{class_id}"


def _keep_top_per_label(detections):
    """If a label appears more than MAX_PER_LABEL times, keep the most confident.

    The final list is sorted from most to least confident.
    """
    groups = {}
    for det in detections:
        groups.setdefault(det["label"], []).append(det)

    kept = []
    for items in groups.values():
        items.sort(key=lambda d: d["confidence"], reverse=True)
        kept.extend(items[:MAX_PER_LABEL])

    kept.sort(key=lambda d: d["confidence"], reverse=True)
    return kept


def _colour_for(label):
    """Return a BGR colour tuple for a label (stable: same label, same colour)."""
    if label in LABEL_COLOURS:
        return LABEL_COLOURS[label]
    # Unknown label: build a repeatable colour from its letters.
    seed = sum(ord(ch) for ch in str(label))
    return (50 + (seed * 37) % 206, 50 + (seed * 91) % 206, 50 + (seed * 53) % 206)


def _is_usable_image(image_bgr):
    """True if image_bgr is a non-empty colour image (height x width x channels)."""
    return (
        isinstance(image_bgr, np.ndarray)
        and image_bgr.ndim == 3
        and image_bgr.size > 0
    )


def _detections_from_result(result, model, width, height):
    """Convert one YOLO result into the CONTRACT list (before the 'top 2' rule).

    Kept as its own small function so detect_regions stays short and simple.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    # Pull the numbers out of the model's answer.
    xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
    confidences = _to_numpy(boxes.conf).reshape(-1)
    class_ids = _to_numpy(boxes.cls).reshape(-1)
    names = getattr(result, "names", None) or getattr(model, "names", None) or {}

    detections = []
    for coords, conf, class_id in zip(xyxy, confidences, class_ids):
        x1, y1, x2, y2 = [int(round(float(v))) for v in coords]

        # Keep the box inside the picture.
        x1 = min(max(x1, 0), width - 1)
        x2 = min(max(x2, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        y2 = min(max(y2, 0), height - 1)

        # Skip boxes that have no area.
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "label": _label_name(names, int(class_id)),
                "confidence": round(float(conf), 3),
                "bbox": [x1, y1, x2, y2],
            }
        )
    return detections


# ---------------------------------------------------------------------------
# Public functions (names and shapes come from the CONTRACT)
# ---------------------------------------------------------------------------

def detect_regions(image_bgr):
    """Find label regions in a pack photo.

    Args:
        image_bgr: the photo as a numpy array in OpenCV BGR order (height x width x 3).

    Returns:
        A list like:
        [{'label': 'mrp', 'confidence': 0.912, 'bbox': [x1, y1, x2, y2]}, ...]
        - confidence is rounded to 3 decimals
        - bbox is 4 whole numbers (pixels), kept inside the image
        - the same label appears at most 2 times (the most confident ones)
        - sorted from most to least confident
        Returns [] if the model file is missing or anything goes wrong.
    """
    try:
        # Check the photo is a normal colour image.
        if not _is_usable_image(image_bgr):
            print("[detector] WARNING: image is missing or not a colour image")
            return []

        model = _get_model()
        if model is None:
            return []

        height, width = image_bgr.shape[:2]

        # Ask YOLO to look at the photo. verbose=False keeps the terminal quiet.
        results = model.predict(
            source=image_bgr,
            conf=CONF_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )
        if not results:
            return []

        detections = _detections_from_result(results[0], model, width, height)
        return _keep_top_per_label(detections)

    except Exception as exc:  # noqa: BLE001
        print(f"[detector] WARNING: detection failed: {exc}")
        return []


def _draw_one_region(canvas, region, scale, text_thickness, box_thickness):
    """Draw a single box and its label text onto canvas (changes canvas)."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    label = str(region["label"])
    confidence = float(region["confidence"])
    x1, y1, x2, y2 = [int(round(float(v))) for v in region["bbox"]]
    colour = _colour_for(label)

    # The box itself.
    cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, box_thickness)

    # The label text, on a filled strip so it is easy to read.
    text = f"{label} {confidence:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, text_thickness)
    strip_h = text_h + baseline + 6

    # Put the strip above the box; if there is no room, put it inside.
    top = y1 - strip_h
    if top < 0:
        top = y1
    cv2.rectangle(canvas, (x1, top), (x1 + text_w + 6, top + strip_h), colour, -1)

    # Choose black or white text depending on how bright the colour is.
    brightness = 0.114 * colour[0] + 0.587 * colour[1] + 0.299 * colour[2]
    text_colour = (0, 0, 0) if brightness > 140 else (255, 255, 255)
    cv2.putText(
        canvas,
        text,
        (x1 + 3, top + text_h + 3),
        font,
        scale,
        text_colour,
        text_thickness,
        cv2.LINE_AA,
    )


def draw_regions(image_bgr, regions):
    """Draw each region's box and label text on a COPY of the image.

    Each label gets its own colour. The original image is never changed.

    Args:
        image_bgr: the photo (numpy array, BGR).
        regions: the list returned by detect_regions.

    Returns:
        A new image with boxes drawn. If the input is not a usable image it is
        returned unchanged.
    """
    # Not a usable image? Nothing to draw on - hand it back as it is.
    if not _is_usable_image(image_bgr):
        return image_bgr

    # Work on a copy so the caller's picture stays exactly as it was.
    canvas = np.ascontiguousarray(image_bgr.copy())

    try:
        height, width = canvas.shape[:2]

        # Make line and text sizes grow with the picture size.
        scale = max(0.5, max(height, width) / 1600.0)
        text_thickness = max(1, int(round(scale * 2)))
        box_thickness = max(2, int(round(scale * 3)))

        for region in regions or []:
            try:
                _draw_one_region(canvas, region, scale, text_thickness, box_thickness)
            except Exception as exc:  # noqa: BLE001 - skip one bad region only
                print(f"[detector] WARNING: could not draw a region: {exc}")
                continue

        return canvas

    except Exception as exc:  # noqa: BLE001
        print(f"[detector] WARNING: drawing failed: {exc}")
        return canvas

# END OF FILE: src/vision/detector.py