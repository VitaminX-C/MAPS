# FILE: src/analytics/db.py
"""SQLite storage for inspection reports.

PLACEHOLDER VERSION: nothing is stored. There is no database file yet.
The real version will keep data/inspections.db and create the data/ folder.
"""

import logging

from src.extraction.extractor import extract_fields
from src.rules.validator import validate

logger = logging.getLogger(__name__)


def _fake_report(inspection_id: int) -> dict:
    """Build a fake report in the correct shape (re-using the placeholder modules)."""
    context = {
        "category": "food",
        "is_imported": False,
        "pdp_width_cm": None,
        "pdp_height_cm": None,
        "ocr_quality": "good",
    }
    fields = extract_fields({"full_text": "", "lines": [], "region_texts": []})
    return {
        "inspection_id": inspection_id,
        "timestamp": "2026-11-09T10:30:00",
        "image_filename": "fake_pack.jpg",
        "context": context,
        "warnings": [],
        "regions": [],
        "ocr_text": "",
        "fields": fields,
        "validation": validate(fields, context),
        "annotated_image_path": None,
        "processing_seconds": 0.0,
        "is_demo": True,
    }


def init_db() -> None:
    """Create the database and its table if they do not exist (placeholder: does nothing)."""
    return None


def save_inspection(report: dict) -> int:
    """Store one report and return its new inspection id (placeholder: always 1)."""
    return 1


def get_all_inspections() -> list:
    """Return every stored report as a list of dicts (placeholder: one fake report)."""
    try:
        return [_fake_report(1)]
    except Exception:
        logger.exception("get_all_inspections failed")
        return []


def get_inspection(id: int) -> dict:
    """Return one stored report by id (placeholder: a fake report with that id)."""
    try:
        return _fake_report(int(id))
    except Exception:
        logger.exception("get_inspection failed")
        return {}


def delete_all() -> None:
    """Delete every stored inspection (placeholder: does nothing)."""
    return None
