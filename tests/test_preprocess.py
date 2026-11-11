# FILE: tests/test_preprocess.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Tests for src/vision/preprocess.py.

All images are tiny fake pictures built in memory with numpy/OpenCV and saved
into pytest's temporary folder, so nothing needs to be downloaded.
"""

import cv2
import numpy as np
from PIL import Image

from src.vision.preprocess import load_and_preprocess, prepare_for_ocr

FAILURE = {'image_bgr': None, 'width': 0, 'height': 0,
           'warnings': ['unreadable_image']}


def _checkerboard(width, height, square=8, dark=(80, 80, 80),
                  light=(180, 180, 180)):
    """Make a sharp black-and-white style checkerboard (BGR, uint8)."""
    ys, xs = np.indices((height, width))
    mask = ((ys // square + xs // square) % 2).astype(bool)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[~mask] = dark
    img[mask] = light
    return img


def _save(tmp_path, name, img):
    """Save an image as PNG (lossless) and return its path as a string."""
    path = str(tmp_path / name)
    assert cv2.imwrite(path, img)
    return path


def test_sharp_image_has_no_warnings(tmp_path):
    path = _save(tmp_path, 'sharp.png', _checkerboard(1000, 800))
    result = load_and_preprocess(path)
    assert result['warnings'] == []
    assert result['width'] == 1000
    assert result['height'] == 800
    assert result['image_bgr'].shape == (800, 1000, 3)


def test_blurred_image_is_flagged_blurry(tmp_path):
    blurred = cv2.GaussianBlur(_checkerboard(1000, 800), (0, 0), 8)
    path = _save(tmp_path, 'blurred.png', blurred)
    result = load_and_preprocess(path)
    assert 'blurry' in result['warnings']


def test_dark_image_is_flagged_too_dark(tmp_path):
    dark_img = _checkerboard(1000, 800, dark=(10, 10, 10), light=(50, 50, 50))
    path = _save(tmp_path, 'dark.png', dark_img)
    result = load_and_preprocess(path)
    assert 'too_dark' in result['warnings']
    assert 'too_bright' not in result['warnings']


def test_bright_image_is_flagged_too_bright(tmp_path):
    bright_img = _checkerboard(1000, 800, dark=(190, 190, 190),
                               light=(250, 250, 250))
    path = _save(tmp_path, 'bright.png', bright_img)
    result = load_and_preprocess(path)
    assert 'too_bright' in result['warnings']
    assert 'too_dark' not in result['warnings']


def test_tiny_image_is_flagged_low_resolution(tmp_path):
    path = _save(tmp_path, 'tiny.png', _checkerboard(300, 200, square=4))
    result = load_and_preprocess(path)
    assert 'low_resolution' in result['warnings']
    assert 'unreadable_image' not in result['warnings']
    assert result['image_bgr'] is not None


def test_missing_file_returns_failure_result(tmp_path):
    result = load_and_preprocess(str(tmp_path / 'does_not_exist.jpg'))
    assert result == FAILURE


def test_corrupt_file_returns_failure_result(tmp_path):
    bad = tmp_path / 'corrupt.jpg'
    bad.write_bytes(b'this is definitely not a picture')
    result = load_and_preprocess(str(bad))
    assert result == FAILURE


def test_big_image_is_shrunk_to_2000_px(tmp_path):
    path = _save(tmp_path, 'big.png', _checkerboard(3000, 1000))
    result = load_and_preprocess(path)
    assert result['width'] == 2000
    assert result['height'] == 667
    assert result['image_bgr'].shape == (667, 2000, 3)
    assert 'low_resolution' not in result['warnings']


def test_colour_is_kept(tmp_path):
    # Reddish checkerboard in BGR order: red channel is much larger than blue.
    red_img = _checkerboard(1000, 800, dark=(40, 40, 200),
                            light=(120, 120, 230))
    path = _save(tmp_path, 'red.png', red_img)
    result = load_and_preprocess(path)
    out = result['image_bgr']
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.dtype == np.uint8
    blue_mean = float(out[:, :, 0].mean())
    red_mean = float(out[:, :, 2].mean())
    assert red_mean - blue_mean > 30


def test_exif_rotation_is_fixed(tmp_path):
    # A 1000 wide x 800 tall picture tagged "rotate me" (EXIF orientation 6)
    # must come out 800 wide x 1000 tall.
    rgb = cv2.cvtColor(_checkerboard(1000, 800), cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    exif = Image.Exif()
    exif[0x0112] = 6  # 0x0112 is the EXIF "Orientation" tag
    path = str(tmp_path / 'rotated.jpg')
    pil_img.save(path, exif=exif)

    result = load_and_preprocess(path)
    assert result['width'] == 800
    assert result['height'] == 1000


def test_prepare_for_ocr_upscales_short_crop():
    crop = np.full((30, 100, 3), 128, dtype=np.uint8)
    out = prepare_for_ocr(crop)
    assert out.ndim == 2                 # grayscale
    assert out.dtype == np.uint8
    assert out.shape[0] == 64            # now 64 px tall
    assert out.shape[1] == 214           # 100 * 64 / 30, rounded up


def test_prepare_for_ocr_keeps_tall_crop_size():
    crop = np.full((100, 200, 3), 128, dtype=np.uint8)
    out = prepare_for_ocr(crop)
    assert out.shape == (100, 200)


def test_prepare_for_ocr_handles_bad_input():
    assert prepare_for_ocr(None).size == 0
    assert prepare_for_ocr(np.zeros((0, 0, 3), dtype=np.uint8)).size == 0