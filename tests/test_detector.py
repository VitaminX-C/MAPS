# FILE: tests/test_detector.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Tests for src/vision/detector.py.
# They use a FAKE model, so no model file or big download is needed.

import numpy as np
import pytest

from src.vision import detector


# ---------------------------------------------------------------------------
# Fake pieces that imitate what YOLO returns
# ---------------------------------------------------------------------------

class FakeBoxes:
    """Imitates result.boxes (xyxy, conf, cls as numpy arrays)."""

    def __init__(self, rows):
        # rows = list of (x1, y1, x2, y2, confidence, class_id)
        if rows:
            data = np.array(rows, dtype=float)
            self.xyxy = data[:, 0:4]
            self.conf = data[:, 4]
            self.cls = data[:, 5]
        else:
            self.xyxy = np.zeros((0, 4))
            self.conf = np.zeros((0,))
            self.cls = np.zeros((0,))


class FakeResult:
    """Imitates one YOLO result."""

    def __init__(self, rows, names):
        self.boxes = FakeBoxes(rows)
        self.names = names


class FakeModel:
    """Imitates the YOLO model object. Remembers how it was called."""

    def __init__(self, rows, names=None):
        self.rows = rows
        self.names = names or {
            0: "product_name", 1: "net_quantity", 2: "mrp",
            3: "date_block", 4: "manufacturer_block", 5: "consumer_care_block",
        }
        self.calls = []

    def predict(self, source=None, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult(self.rows, self.names)]


class CrashingModel:
    """A model whose predict always fails."""

    names = {}

    def predict(self, *args, **kwargs):
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def reset_cached_model(monkeypatch):
    """Before every test, forget any cached model (restored afterwards)."""
    monkeypatch.setattr(detector, "_model", None)


@pytest.fixture
def blank_image():
    """A black 400 (high) x 600 (wide) colour image."""
    return np.zeros((400, 600, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_empty_list_when_model_file_missing(monkeypatch, tmp_path, blank_image, capsys):
    """(a) No model file -> [] and a short warning, no crash."""
    monkeypatch.setattr(detector, "MODEL_PATH", tmp_path / "does_not_exist.pt")
    assert detector.detect_regions(blank_image) == []
    assert "warning" in capsys.readouterr().out.lower()


def test_output_has_contract_shape_with_fake_model(monkeypatch, blank_image):
    """(b) Each result has label (str), confidence (3 decimals), bbox (4 ints)."""
    fake = FakeModel([(10.4, 20.6, 100.2, 80.9, 0.91234, 2)])
    monkeypatch.setattr(detector, "_model", fake)

    regions = detector.detect_regions(blank_image)

    assert len(regions) == 1
    region = regions[0]
    assert set(region.keys()) == {"label", "confidence", "bbox"}
    assert region["label"] == "mrp"
    assert region["confidence"] == 0.912
    assert region["bbox"] == [10, 21, 100, 81]
    assert len(region["bbox"]) == 4
    assert all(type(v) is int for v in region["bbox"])


def test_predict_is_called_with_required_settings(monkeypatch, blank_image):
    """The model must be asked for conf=0.25, imgsz=960, on the CPU."""
    fake = FakeModel([])
    monkeypatch.setattr(detector, "_model", fake)

    assert detector.detect_regions(blank_image) == []
    assert len(fake.calls) == 1
    assert fake.calls[0]["conf"] == 0.25
    assert fake.calls[0]["imgsz"] == 960
    assert fake.calls[0]["device"] == "cpu"


def test_keeps_only_two_most_confident_per_label(monkeypatch, blank_image):
    """Four 'mrp' boxes -> only the 2 best stay. Other labels are untouched."""
    rows = [
        (10, 10, 50, 50, 0.90, 2),
        (60, 10, 100, 50, 0.50, 2),
        (110, 10, 150, 50, 0.80, 2),
        (160, 10, 200, 50, 0.30, 2),
        (10, 100, 90, 140, 0.70, 0),   # one product_name box
    ]
    monkeypatch.setattr(detector, "_model", FakeModel(rows))

    regions = detector.detect_regions(blank_image)

    mrp_confidences = sorted(r["confidence"] for r in regions if r["label"] == "mrp")
    assert mrp_confidences == [0.8, 0.9]
    assert len([r for r in regions if r["label"] == "product_name"]) == 1
    assert len(regions) == 3


def test_returns_empty_list_when_prediction_crashes(monkeypatch, blank_image):
    """If the model raises an error, we return [] instead of crashing."""
    monkeypatch.setattr(detector, "_model", CrashingModel())
    assert detector.detect_regions(blank_image) == []


def test_returns_empty_list_for_invalid_image(monkeypatch):
    """None or a grey (2-D) image is not usable -> []."""
    monkeypatch.setattr(detector, "_model", FakeModel([(1, 1, 5, 5, 0.9, 0)]))
    assert detector.detect_regions(None) == []
    assert detector.detect_regions(np.zeros((10, 10), dtype=np.uint8)) == []


def test_uses_cached_model_and_does_not_reload(monkeypatch, tmp_path, blank_image):
    """Once a model is cached, the (missing) file is not looked at again."""
    monkeypatch.setattr(detector, "MODEL_PATH", tmp_path / "does_not_exist.pt")
    fake = FakeModel([(10, 10, 60, 60, 0.9, 1)])
    monkeypatch.setattr(detector, "_model", fake)

    first = detector.detect_regions(blank_image)
    second = detector.detect_regions(blank_image)

    assert first == second
    assert first[0]["label"] == "net_quantity"
    assert len(fake.calls) == 2   # same model object used both times


def test_boxes_are_clamped_inside_image(monkeypatch, blank_image):
    """A box sticking out of the picture is trimmed to the picture edges."""
    fake = FakeModel([(-20, -5, 5000, 5000, 0.8, 3)])
    monkeypatch.setattr(detector, "_model", fake)

    regions = detector.detect_regions(blank_image)   # image is 600 wide, 400 high
    assert regions[0]["bbox"] == [0, 0, 599, 399]


def test_draw_regions_does_not_change_original_image(blank_image):
    """(c) The original stays untouched; the returned copy has boxes drawn."""
    original_before = blank_image.copy()
    regions = [{"label": "mrp", "confidence": 0.9, "bbox": [50, 50, 200, 150]}]

    drawn = detector.draw_regions(blank_image, regions)

    assert np.array_equal(blank_image, original_before)   # original unchanged
    assert drawn is not blank_image                        # a different object
    assert drawn.shape == blank_image.shape
    assert drawn.sum() > 0                                 # something was drawn


def test_draw_regions_handles_empty_and_bad_input(blank_image):
    """No regions -> identical copy. Bad image -> returned as is. Bad region -> skipped."""
    same = detector.draw_regions(blank_image, [])
    assert np.array_equal(same, blank_image)

    assert detector.draw_regions(None, []) is None

    bad_region = [{"label": "mrp"}]   # missing confidence and bbox
    result = detector.draw_regions(blank_image, bad_region)
    assert np.array_equal(result, blank_image)


def test_each_label_gets_its_own_colour():
    """The 6 official labels must all have different colours."""
    labels = [
        "product_name", "net_quantity", "mrp",
        "date_block", "manufacturer_block", "consumer_care_block",
    ]
    colours = {detector._colour_for(label) for label in labels}
    assert len(colours) == 6
