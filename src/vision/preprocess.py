# FILE: src/vision/preprocess.py
"""Image loading and cleaning.

PLACEHOLDER VERSION: it only loads the photo and reads its size.
There is no blur / brightness / resolution checking yet, so 'warnings' is
always empty for a readable image.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_and_preprocess(image_path: str) -> dict:
    """Load an image file and return it with its size and quality warnings.

    Returns:
        {'image_bgr': numpy array, 'width': int, 'height': int, 'warnings': [str]}
        If the file cannot be read, returns the safe default
        {'image_bgr': None, 'width': 0, 'height': 0, 'warnings': ['unreadable_image']}.
    """
    # The safe answer we return whenever anything goes wrong.
    failure = {
        "image_bgr": None,
        "width": 0,
        "height": 0,
        "warnings": ["unreadable_image"],
    }
    try:
        # cv2.imread returns None (it does not raise) when the file is bad.
        image = cv2.imread(str(image_path))
        if image is None or image.size == 0:
            return failure
        height, width = image.shape[:2]
        return {
            "image_bgr": image,
            "width": int(width),
            "height": int(height),
            "warnings": [],  # PLACEHOLDER: real checks come later
        }
    except Exception:
        logger.exception("load_and_preprocess failed for %s", image_path)
        return failure


def prepare_for_ocr(crop_bgr) -> np.ndarray:
    """Turn a cropped colour image into a grayscale image ready for OCR.

    PLACEHOLDER VERSION: converts to grayscale and doubles the size.
    Real contrast enhancement comes later.
    On any problem it returns a tiny 1x1 black grayscale image.
    """
    safe_default = np.zeros((1, 1), dtype=np.uint8)
    try:
        if crop_bgr is None or crop_bgr.size == 0:
            return safe_default
        # A colour image has 3 dimensions; a grayscale one has 2.
        if crop_bgr.ndim == 3:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop_bgr
        height, width = gray.shape[:2]
        # Make the picture 2x bigger so small print is easier to read.
        return cv2.resize(
            gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC
        )
    except Exception:
        logger.exception("prepare_for_ocr failed")
        return safe_default