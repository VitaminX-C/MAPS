# FILE: src/vision/detector.py
"""Label-region detection.

PLACEHOLDER VERSION: it does NOT use YOLO and does NOT look for models/best.pt.
It always returns 6 fake boxes laid out in fixed positions on the photo.
(The real version must return [] when models/best.pt is missing.)
"""

import logging

import cv2

logger = logging.getLogger(__name__)

# The 6 allowed region labels, with a fake position for each box.
# Each position is (left, top, right, bottom) as a FRACTION of the image size.
_FAKE_LAYOUT = {
    "product_name": (0.05, 0.05, 0.95, 0.20),
    "net_quantity": (0.05, 0.22, 0.40, 0.32),
    "mrp": (0.45, 0.22, 0.95, 0.32),
    "date_block": (0.05, 0.35, 0.95, 0.50),
    "manufacturer_block": (0.05, 0.52, 0.95, 0.75),
    "consumer_care_block": (0.05, 0.78, 0.95, 0.92),
}


def detect_regions(image_bgr) -> list:
    """Find label regions in the image.

    Returns:
        A list of {'label': str, 'confidence': float, 'bbox': [x1, y1, x2, y2]}
        with pixel coordinates. Returns [] if anything fails.
    """
    try:
        if image_bgr is None:
            return []
        height, width = image_bgr.shape[:2]
        regions = []
        for label, (left, top, right, bottom) in _FAKE_LAYOUT.items():
            regions.append(
                {
                    "label": label,
                    "confidence": 0.9,
                    "bbox": [
                        int(left * width),
                        int(top * height),
                        int(right * width),
                        int(bottom * height),
                    ],
                }
            )
        return regions
    except Exception:
        logger.exception("detect_regions failed")
        return []


def draw_regions(image_bgr, regions):
    """Return a COPY of the image with a green labelled box around each region.

    If anything goes wrong, the original image is returned unchanged.
    """
    try:
        if image_bgr is None:
            return image_bgr
        annotated = image_bgr.copy()
        for region in regions or []:
            try:
                x1, y1, x2, y2 = [int(v) for v in region["bbox"]]
                caption = f"{region['label']} {float(region['confidence']):.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(
                    annotated,
                    caption,
                    (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 200, 0),
                    1,
                    cv2.LINE_AA,
                )
            except Exception:
                # One bad region must not stop the others being drawn.
                logger.exception("could not draw region %s", region)
        return annotated
    except Exception:
        logger.exception("draw_regions failed")
        return image_bgr