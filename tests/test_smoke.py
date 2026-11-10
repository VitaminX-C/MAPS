# FILE: tests/test_smoke.py
"""Smoke test: run the whole pipeline on a tiny blank image and check that the
report has every key the CONTRACT promises."""

from PIL import Image

from src import pipeline

EXPECTED_REPORT_KEYS = {
    "inspection_id",
    "timestamp",
    "image_filename",
    "context",
    "warnings",
    "regions",
    "ocr_text",
    "fields",
    "validation",
    "annotated_image_path",
    "processing_seconds",
    "is_demo",
}

EXPECTED_FIELD_NAMES = {
    "product_name",
    "net_quantity",
    "mrp",
    "mfg_date",
    "expiry_date",
    "manufacturer",
    "consumer_care",
    "country_of_origin",
    "unit_sale_price",
}

EXPECTED_FIELD_KEYS = {"found", "raw_text", "confidence", "parsed"}
EXPECTED_VALIDATION_KEYS = {"overall_status", "checks", "summary"}
EXPECTED_SUMMARY_KEYS = {"passed", "failed", "review", "not_applicable"}


def test_pipeline_returns_every_contract_key(tmp_path, monkeypatch):
    # Send the annotated image to a temporary folder, not the real outputs/ folder.
    monkeypatch.setattr(pipeline, "ANNOTATED_DIR", tmp_path / "annotated")

    # Create a tiny blank white image in the temporary folder.
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (64, 64), color="white").save(image_path)

    context = {
        "category": "food",
        "is_imported": False,
        "pdp_width_cm": None,
        "pdp_height_cm": None,
        "ocr_quality": "good",
    }

    report = pipeline.analyze_image(str(image_path), context)

    # The report itself.
    assert set(report.keys()) == EXPECTED_REPORT_KEYS
    assert report["image_filename"] == "blank.png"

    # The 9 fields, each with the 4 required keys.
    assert set(report["fields"].keys()) == EXPECTED_FIELD_NAMES
    for field in report["fields"].values():
        assert set(field.keys()) == EXPECTED_FIELD_KEYS

    # The validation block.
    assert set(report["validation"].keys()) == EXPECTED_VALIDATION_KEYS
    assert set(report["validation"]["summary"].keys()) == EXPECTED_SUMMARY_KEYS
    assert report["validation"]["overall_status"] in {"PASS", "FAIL", "REVIEW"}
