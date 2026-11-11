# FILE: src/vision/preprocess.py
"""
Image pre-processing for the Legal Metrology Checker.

Two public functions (exactly as in the CONTRACT):

    load_and_preprocess(image_path) -> dict
        Reads a photo, fixes phone rotation, shrinks it if huge, checks quality
        (resolution, blur, brightness) and applies a light colour-preserving
        clean-up so that YOLO can detect label regions.

    prepare_for_ocr(crop_bgr) -> grayscale numpy array
        Takes a cropped label region and makes it easier for EasyOCR to read
        (upscale if small, grayscale, contrast boost).

Neither function ever raises an error. If something goes wrong they return the
safe default described in the CONTRACT.
"""

import math
import os

import cv2
import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Constants you may want to tune later (kept at the top on purpose).
# ---------------------------------------------------------------------------

# Blur check: variance of the Laplacian of the grey image.
# A value BELOW this number means the photo is considered 'blurry'.
BLUR_THRESHOLD = 100.0

# Brightness check on the mean grey value (0 = black, 255 = white).
DARK_THRESHOLD = 60.0     # mean below this  -> 'too_dark'
BRIGHT_THRESHOLD = 200.0  # mean above this  -> 'too_bright'

# Size limits (in pixels) for the longest side of the photo.
MAX_SIDE_PX = 2000   # bigger photos are shrunk to this
MIN_SIDE_PX = 800    # smaller photos get the 'low_resolution' warning

# prepare_for_ocr: crops shorter than this are enlarged.
MIN_CROP_HEIGHT_PX = 64

# CLAHE (local contrast boost) settings.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


def _failure_result():
    """Return the failure result required by the CONTRACT."""
    return {
        'image_bgr': None,
        'width': 0,
        'height': 0,
        'warnings': ['unreadable_image'],
    }


def _read_image_as_bgr(image_path):
    """
    Read an image file with Pillow, fix its rotation using the EXIF tag,
    and return it as a BGR numpy array (the colour order OpenCV uses).

    Returns None if the file is missing, corrupt or not an image.
    """
    try:
        # A missing file or a wrong type of path is not readable.
        if not isinstance(image_path, (str, os.PathLike)):
            return None
        if not os.path.isfile(image_path):
            return None

        with Image.open(image_path) as pil_img:
            # load() forces Pillow to really decode the file, so a corrupt
            # file fails here (inside try) and not later.
            pil_img.load()

            # Phones store "which way is up" in the EXIF tag instead of
            # rotating the pixels. exif_transpose rotates the pixels for us.
            oriented = pil_img
            try:
                fixed = ImageOps.exif_transpose(pil_img)
                if fixed is not None:
                    oriented = fixed
            except Exception:
                # If the EXIF data is damaged we simply keep the image as is.
                oriented = pil_img

            # Force 3 colour channels (handles grayscale, PNG with
            # transparency, palette images and so on).
            rgb_array = np.array(oriented.convert('RGB'), dtype=np.uint8)

        if rgb_array.ndim != 3 or rgb_array.size == 0:
            return None

        # Pillow gives RGB, OpenCV wants BGR.
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(bgr_array)
    except Exception:
        return None


def _shrink_if_needed(image_bgr):
    """
    If the longest side is over MAX_SIDE_PX, shrink the image so that the
    longest side equals MAX_SIDE_PX, keeping the aspect ratio.
    Otherwise return the image unchanged.
    """
    height, width = image_bgr.shape[:2]
    longest = max(height, width)
    if longest <= MAX_SIDE_PX:
        return image_bgr

    scale = MAX_SIDE_PX / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    # INTER_AREA is the best OpenCV method for making pictures smaller.
    return cv2.resize(image_bgr, (new_width, new_height),
                      interpolation=cv2.INTER_AREA)


def _quality_warnings(image_bgr):
    """
    Check blur and brightness on the grey version of the image.
    Returns a list containing any of: 'blurry', 'too_dark', 'too_bright'.
    If the check itself fails, returns an empty list (no false alarms).
    """
    found = []
    try:
        grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Blur: a sharp photo has strong edges, so the Laplacian (an edge
        # detector) has a high variance. A blurry photo has a low variance.
        blur_score = cv2.Laplacian(grey, cv2.CV_64F).var()
        if blur_score < BLUR_THRESHOLD:
            found.append('blurry')

        # Brightness: the average grey value of all pixels.
        mean_brightness = float(grey.mean())
        if mean_brightness < DARK_THRESHOLD:
            found.append('too_dark')
        elif mean_brightness > BRIGHT_THRESHOLD:
            found.append('too_bright')
    except Exception:
        return []
    return found


def _light_cleanup(image_bgr):
    """
    Light clean-up that KEEPS COLOUR (YOLO needs colour):
      1. gentle edge-preserving denoise
      2. CLAHE contrast boost on the L (lightness) channel only, in LAB colour

    If anything fails, the original image is returned unchanged.
    """
    try:
        # Bilateral filter smooths noise but keeps edges (text) sharp.
        # The small numbers make it "gentle" and fast.
        denoised = cv2.bilateralFilter(image_bgr, d=5,
                                       sigmaColor=25, sigmaSpace=25)

        # LAB colour space separates lightness (L) from colour (A and B).
        # Boosting only L improves contrast without changing the colours.
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                tileGridSize=CLAHE_TILE_GRID)
        l_boosted = clahe.apply(l_channel)
        merged = cv2.merge((l_boosted, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        return image_bgr


def load_and_preprocess(image_path: str) -> dict:
    """
    Load a product photo and prepare it for the YOLO detector.

    Steps:
      1. Read the file (missing/corrupt -> 'unreadable_image' failure result).
      2. Fix phone rotation (EXIF) and convert to a BGR numpy array.
      3. Shrink if the longest side is over 2000 px; add 'low_resolution'
         if the longest side is under 800 px.
      4. Blur check ('blurry').
      5. Brightness check ('too_dark' / 'too_bright').
      6. Light clean-up (gentle denoise + CLAHE on the L channel). The image
         stays in colour.

    The blur and brightness checks are done BEFORE the clean-up, so they
    measure the real photo and not our improved version of it.

    Returns:
        {'image_bgr': numpy array, 'width': int, 'height': int,
         'warnings': [str]}
        width and height describe the returned image (after any shrinking).
        On failure: {'image_bgr': None, 'width': 0, 'height': 0,
                     'warnings': ['unreadable_image']}
    """
    try:
        # Steps 1 and 2: read the file and fix rotation.
        image_bgr = _read_image_as_bgr(image_path)
        if image_bgr is None:
            return _failure_result()

        warnings = []

        # Step 3: shrink if huge, warn if tiny.
        image_bgr = _shrink_if_needed(image_bgr)
        height, width = image_bgr.shape[:2]
        if max(height, width) < MIN_SIDE_PX:
            warnings.append('low_resolution')

        # Steps 4 and 5: blur and brightness checks.
        warnings.extend(_quality_warnings(image_bgr))

        # Step 6: light clean-up (colour is kept).
        image_bgr = _light_cleanup(image_bgr)

        return {
            'image_bgr': image_bgr,
            'width': int(width),
            'height': int(height),
            'warnings': warnings,
        }
    except Exception:
        return _failure_result()


def prepare_for_ocr(crop_bgr):
    """
    Make a cropped label region easier for EasyOCR to read.

    Steps:
      1. If the crop is shorter than 64 px, enlarge it with INTER_CUBIC so
         it is exactly 64 px tall (width grows by the same factor).
      2. Convert to grayscale.
      3. Apply CLAHE (local contrast boost).

    Returns:
        A grayscale uint8 numpy array (2 dimensions).
        If the input is None, empty or invalid, returns an EMPTY array of
        shape (0, 0). Callers can test this with `result.size == 0`.
    """
    empty = np.zeros((0, 0), dtype=np.uint8)
    try:
        if crop_bgr is None:
            return empty

        crop = np.asarray(crop_bgr)
        if crop.size == 0 or crop.ndim not in (2, 3):
            return empty
        if crop.dtype != np.uint8:
            crop = np.clip(crop, 0, 255).astype(np.uint8)
        crop = np.ascontiguousarray(crop)

        # Step 1: enlarge small crops (small text is hard for OCR).
        height, width = crop.shape[:2]
        if height < MIN_CROP_HEIGHT_PX:
            scale = MIN_CROP_HEIGHT_PX / float(height)
            new_width = max(1, int(math.ceil(width * scale)))
            crop = cv2.resize(crop, (new_width, MIN_CROP_HEIGHT_PX),
                              interpolation=cv2.INTER_CUBIC)

        # Step 2: convert to grayscale (also accepts already-grey input).
        if crop.ndim == 2:
            grey = crop
        elif crop.shape[2] == 1:
            grey = np.ascontiguousarray(crop[:, :, 0])
        elif crop.shape[2] == 4:
            grey = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
        else:
            grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Step 3: boost local contrast.
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                tileGridSize=CLAHE_TILE_GRID)
        return clahe.apply(grey)
    except Exception:
        return empty