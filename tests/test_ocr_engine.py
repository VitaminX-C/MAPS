# FILE: tests/test_ocr_engine.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests for src/ocr/ocr_engine.py
#   - Fast tests use a FAKE reader (no model download, run in ~1 second).
#   - Tests marked @pytest.mark.slow use the REAL EasyOCR.
#     Skip them with:  python -m pytest tests/test_ocr_engine.py -m "not slow"

import cv2
import numpy as np
import pytest

from src.ocr import ocr_engine
from src.ocr.ocr_engine import run_ocr

EMPTY = {'full_text': '', 'lines': [], 'region_texts': []}
DEVANAGARI_DIGITS = '०१२३४५६७८९'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def quad(x1, y1, x2, y2):
    """Make an EasyOCR-style 4-corner box from simple x1,y1,x2,y2 numbers."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class FakeReader:
    """Pretends to be easyocr.Reader. `handler(image)` decides what to return."""

    def __init__(self, handler):
        self.handler = handler

    def readtext(self, image, **kwargs):
        return self.handler(image)


def use_fake_reader(monkeypatch, handler):
    """Make run_ocr use our FakeReader instead of loading EasyOCR."""
    monkeypatch.setattr(ocr_engine, '_get_reader', lambda: FakeReader(handler))


def fake_prepare_for_ocr(crop):
    """Stand-in for prepare_for_ocr: grayscale and exactly 2x bigger."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, None, fx=2.0, fy=2.0)


# ---------------------------------------------------------------------------
# Fast tests (fake reader)
# ---------------------------------------------------------------------------
def test_none_image_returns_empty_result():
    assert run_ocr(None, []) == EMPTY


def test_wrong_input_types_return_empty_result():
    for bad_input in (np.zeros((0, 0, 3), dtype=np.uint8), 'photo.jpg', 42):
        assert run_ocr(bad_input, []) == EMPTY


def test_box_to_xyxy_converts_four_corners_to_ints():
    box = [[10.4, 20.6], [110.2, 20.9], [110.7, 50.1], [10.1, 50.4]]
    assert ocr_engine._box_to_xyxy(box) == [10, 21, 111, 50]


def test_devanagari_digits_are_converted_to_ascii():
    # Pure Devanagari, mixed with normal digits, and text without digits.
    assert ocr_engine._normalise_digits('२०० g') == '200 g'
    assert ocr_engine._normalise_digits('२0०') == '200'
    assert ocr_engine._normalise_digits('MRP Rs. ४५.००') == 'MRP Rs. 45.00'
    assert ocr_engine._normalise_digits('Net Wt. 200 g') == 'Net Wt. 200 g'
    # Hindi letters and superscripts are NOT digits and must stay unchanged.
    assert ocr_engine._normalise_digits('मूल्य') == 'मूल्य'
    assert ocr_engine._normalise_digits('m²') == 'm²'


def test_devanagari_digits_are_fixed_inside_run_ocr(monkeypatch):
    # This is exactly what the real Hindi model returned on your computer.
    words = [
        (quad(10, 100, 90, 130), 'Net', 0.90),
        (quad(100, 101, 180, 131), 'Wt.', 0.90),
        (quad(190, 100, 280, 130), '२०० 9', 0.80),
    ]
    use_fake_reader(monkeypatch, lambda image: words)
    image = np.zeros((200, 400, 3), dtype=np.uint8)

    result = run_ocr(image, [])

    assert result['full_text'] == 'Net Wt. 200 9'
    assert not any(ch in result['full_text'] for ch in DEVANAGARI_DIGITS)


def test_words_on_same_row_are_merged_and_sorted(monkeypatch):
    # Words are given in a jumbled order on purpose.
    words = [
        (quad(200, 100, 300, 130), '45.00', 0.90),
        (quad(10, 102, 90, 132), 'MRP', 0.95),
        (quad(100, 98, 180, 128), 'Rs.', 0.90),
        (quad(100, 201, 200, 231), 'Wt. 200 g', 0.80),
        (quad(10, 200, 90, 230), 'Net', 0.90),
    ]
    use_fake_reader(monkeypatch, lambda image: words)
    image = np.zeros((300, 400, 3), dtype=np.uint8)

    result = run_ocr(image, [])

    assert result['full_text'] == 'MRP Rs. 45.00\nNet Wt. 200 g'
    assert len(result['lines']) == 2
    assert result['lines'][0]['bbox'] == [10, 98, 300, 132]
    assert result['lines'][0]['confidence'] == pytest.approx(0.9167, abs=1e-3)
    assert result['region_texts'] == []


def test_low_confidence_text_is_dropped(monkeypatch):
    words = [
        (quad(10, 10, 90, 40), 'noise', 0.10),   # dropped (below 0.2)
        (quad(10, 60, 90, 90), 'Keep', 0.20),    # kept (0.2 is not below 0.2)
    ]
    use_fake_reader(monkeypatch, lambda image: words)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = run_ocr(image, [])

    assert result['full_text'] == 'Keep'
    assert len(result['lines']) == 1


def test_reader_creation_failure_returns_empty_result(monkeypatch):
    def broken_reader():
        raise RuntimeError('model could not be loaded')

    monkeypatch.setattr(ocr_engine, '_get_reader', broken_reader)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    assert run_ocr(image, []) == EMPTY


def test_readtext_failure_returns_empty_result(monkeypatch):
    def broken_handler(image):
        raise RuntimeError('EasyOCR crashed')

    use_fake_reader(monkeypatch, broken_handler)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    assert run_ocr(image, []) == EMPTY


def test_region_boxes_are_mapped_back_to_full_image(monkeypatch):
    monkeypatch.setattr(ocr_engine, 'prepare_for_ocr', fake_prepare_for_ocr)

    # The crop is 120x70 and is enlarged 2x to 240x140. A word found at
    # (20,20)-(100,40) in the enlarged crop must land at
    # x: 90 + 20*0.5 = 100 ... 90 + 100*0.5 = 140
    # y: 40 + 20*0.5 = 50  ... 40 + 40*0.5  = 60
    def handler(image):
        return [(quad(20, 20, 100, 40), 'MRP', 0.9)]

    image = np.zeros((200, 400, 3), dtype=np.uint8)
    words = ocr_engine._read_region_words(
        FakeReader(handler), image, [90, 40, 210, 110]
    )

    assert len(words) == 1
    assert words[0]['text'] == 'MRP'
    assert words[0]['bbox'] == [100, 50, 140, 60]


def test_region_texts_built_and_bad_regions_skipped(monkeypatch):
    monkeypatch.setattr(ocr_engine, 'prepare_for_ocr', fake_prepare_for_ocr)

    def handler(image):
        if image.ndim == 3:        # the full colour image: nothing found
            return []
        return [                   # a cropped (gray) region
            (quad(10, 10, 200, 40), 'Net Wt. 200 g', 0.90),
            (quad(10, 60, 50, 80), 'x', 0.10),   # too unsure, dropped
        ]

    use_fake_reader(monkeypatch, handler)
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    regions = [
        {'label': 'net_quantity', 'confidence': 0.9, 'bbox': [100, 50, 200, 100]},
        'this is not a dict',                                # skipped
        {'label': 'mrp'},                                    # no bbox: skipped
        {'label': 'date_block', 'bbox': [1, 2]},             # wrong shape: skipped
        {'label': 'product_name', 'bbox': [500, 500, 600, 600]},  # outside image
    ]

    result = run_ocr(image, regions)

    assert result['full_text'] == ''
    assert result['lines'] == []
    assert len(result['region_texts']) == 1
    entry = result['region_texts'][0]
    assert entry['label'] == 'net_quantity'
    assert entry['text'] == 'Net Wt. 200 g'
    assert entry['confidence'] == pytest.approx(0.9)
    assert entry['bbox'] == [100, 50, 200, 100]


# ---------------------------------------------------------------------------
# Slow tests (REAL EasyOCR - first run downloads models, needs internet)
# ---------------------------------------------------------------------------
def make_label_image():
    """White image with two lines of black text drawn by OpenCV."""
    image = np.full((320, 640, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'MRP Rs. 45.00', (30, 100), font, 1.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, 'Net Wt. 200 g', (30, 240), font, 1.6, (0, 0, 0), 4, cv2.LINE_AA)
    return image


@pytest.mark.slow
def test_real_ocr_reads_mrp_and_net_quantity():
    result = run_ocr(make_label_image(), [])

    assert '45' in result['full_text']
    assert '200' in result['full_text']
    assert not any(ch in result['full_text'] for ch in DEVANAGARI_DIGITS)
    assert len(result['lines']) >= 2
    # 'MRP Rs. 45.00' must stay together on ONE line.
    mrp_lines = [line['text'] for line in result['lines'] if '45' in line['text']]
    assert mrp_lines, 'no line contains 45'
    assert 'MR' in mrp_lines[0].upper()


@pytest.mark.slow
def test_real_ocr_reads_a_region():
    regions = [{'label': 'net_quantity', 'confidence': 0.9, 'bbox': [20, 195, 480, 255]}]

    result = run_ocr(make_label_image(), regions)

    assert len(result['region_texts']) == 1
    assert result['region_texts'][0]['label'] == 'net_quantity'
    assert '200' in result['region_texts'][0]['text']