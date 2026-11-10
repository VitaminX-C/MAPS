# FILE: src/pipeline.py
"""The pipeline: runs every step in order and builds the report.

Order: preprocess -> detect regions -> OCR -> extract fields -> validate.
It never raises an error. If something breaks it returns a complete report
with safe defaults and adds 'pipeline_error' to the warnings.

NOTE: this version does not save to the database yet, so 'inspection_id' is
always None. The app (or a later version of this file) will call
save_inspection and fill it in.
"""

import datetime
import logging
import os
import re
import time
from pathlib import Path

import cv2

from src.extraction.extractor import extract_fields
from src.ocr.ocr_engine import run_ocr
from src.rules.validator import validate
from src.vision.detector import detect_regions, draw_regions
from src.vision.preprocess import load_and_preprocess

logger = logging.getLogger(__name__)

# Change this to False when ALL placeholder modules have been replaced.
USING_PLACEHOLDERS = True

# The repository root folder (this file lives in <root>/src/pipeline.py).
REPO_ROOT = Path(__file__).resolve().parent.parent

# Annotated photos (with boxes drawn) are saved here. Git ignores outputs/.
ANNOTATED_DIR = REPO_ROOT / "outputs" / "annotated"

# Used for any context value the caller did not supply.
DEFAULT_CONTEXT = {
    "category": "food",
    "is_imported": False,
    "pdp_width_cm": None,
    "pdp_height_cm": None,
    "ocr_quality": "good",
}

# An empty 'parsed' dict for each of the 9 fields (used for safe defaults).
_EMPTY_PARSED = {
    "product_name": {"name": None},
    "net_quantity": {"amount": None, "unit": None},
    "mrp": {"amount": None, "currency_ok": False, "inclusive_of_taxes": False},
    "mfg_date": {"day": None, "month": None, "year": None, "label_type": None},
    "expiry_date": {"day": None, "month": None, "year": None, "label_type": None},
    "manufacturer": {"name": None, "address": None, "pincode": None, "role": None},
    "consumer_care": {"phone": None, "email": None, "has_name_or_address": False},
    "country_of_origin": {"country": None},
    "unit_sale_price": {"amount": None, "per_unit": None},
}


def _empty_fields() -> dict:
    """Return all 9 fields marked as 'not found' (the safe default)."""
    return {
        name: {
            "found": False,
            "raw_text": None,
            "confidence": 0.0,
            "parsed": dict(parsed),
        }
        for name, parsed in _EMPTY_PARSED.items()
    }


def _safe_validation() -> dict:
    """Return a 'REVIEW' result with no checks (the safe default)."""
    return {
        "overall_status": "REVIEW",
        "checks": [],
        "summary": {"passed": 0, "failed": 0, "review": 0, "not_applicable": 0},
    }


def _merge_context(context) -> dict:
    """Fill any missing context values from DEFAULT_CONTEXT."""
    merged = dict(DEFAULT_CONTEXT)
    if isinstance(context, dict):
        merged.update(context)
    return merged


def _save_annotated_image(image_bgr, image_filename: str):
    """Save the annotated image as a PNG. Returns its path, or None on failure."""
    try:
        ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
        # Keep only safe characters from the original file name.
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(image_filename).stem) or "image"
        file_name = f"{stem}_{int(time.time() * 1000)}_annotated.png"
        out_path = ANNOTATED_DIR / file_name
        written = cv2.imwrite(str(out_path), image_bgr)
        return str(out_path) if written else None
    except Exception:
        logger.exception("could not save annotated image")
        return None


def analyze_image(image_path: str, context: dict) -> dict:
    """Analyse one product photo and return the full report dict.

    Report keys: inspection_id, timestamp, image_filename, context, warnings,
    regions, ocr_text, fields, validation, annotated_image_path,
    processing_seconds, is_demo.
    """
    started = time.perf_counter()
    merged_context = _merge_context(context)

    # Start from a complete, safe report. Each step below overwrites its part.
    report = {
        "inspection_id": None,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "image_filename": os.path.basename(str(image_path)),
        "context": merged_context,
        "warnings": [],
        "regions": [],
        "ocr_text": "",
        "fields": _empty_fields(),
        "validation": _safe_validation(),
        "annotated_image_path": None,
        "processing_seconds": 0.0,
        "is_demo": USING_PLACEHOLDERS,
    }

    try:
        # Step 1: load and clean the image.
        prepared = load_and_preprocess(image_path)
        report["warnings"] = list(prepared.get("warnings", []))
        image_bgr = prepared.get("image_bgr")

        if image_bgr is None:
            # Unreadable photo: skip vision and OCR, and mark OCR quality poor.
            # (merged_context is the same object stored in the report.)
            regions = []
            ocr_result = {"full_text": "", "lines": [], "region_texts": []}
            merged_context["ocr_quality"] = "poor"
        else:
            # Step 2: find label regions.
            regions = detect_regions(image_bgr)
            # Step 3: read the text.
            ocr_result = run_ocr(image_bgr, regions)
            # Save a copy of the photo with the boxes drawn on it.
            annotated = draw_regions(image_bgr, regions)
            report["annotated_image_path"] = _save_annotated_image(
                annotated, report["image_filename"]
            )

        report["regions"] = regions
        report["ocr_text"] = ocr_result.get("full_text", "")

        # Step 4: extract the 9 fields.
        report["fields"] = extract_fields(ocr_result)

        # Step 5: validate against the rules.
        report["validation"] = validate(report["fields"], merged_context)
    except Exception:
        logger.exception("analyze_image failed")
        report["warnings"].append("pipeline_error")

    report["processing_seconds"] = round(time.perf_counter() - started, 3)
    return report
