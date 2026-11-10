# FILE: src/ocr/ocr_engine.py
"""Text reading (OCR).

PLACEHOLDER VERSION: it does NOT run EasyOCR. It ignores the pixels and
returns fixed fake text for a pretend biscuit packet.
"""

import logging

logger = logging.getLogger(__name__)

# Fake text lines, as if printed on a pack.
_FAKE_LINES = [
    "Sample Tea Biscuits",
    "Net Qty: 250 g",
    "MRP Rs. 45.00 (Incl. of all taxes)",
    "Mfg Date: 09/2026",
    "Best Before: 03/2027",
    "Mfd. by: Sample Foods Pvt Ltd, 12 Industrial Area, Bengaluru 560001",
    "Customer Care: 1800-123-4567, care@samplefoods.example",
    "Made in India",
]

# Fake text to report for each region label.
_FAKE_REGION_TEXT = {
    "product_name": "Sample Tea Biscuits",
    "net_quantity": "Net Qty: 250 g",
    "mrp": "MRP Rs. 45.00 (Incl. of all taxes)",
    "date_block": "Mfg Date: 09/2026 Best Before: 03/2027",
    "manufacturer_block": "Mfd. by: Sample Foods Pvt Ltd, 12 Industrial Area, Bengaluru 560001",
    "consumer_care_block": "Customer Care: 1800-123-4567, care@samplefoods.example",
}


def run_ocr(image_bgr, regions: list) -> dict:
    """Read the text in the image.

    Returns:
        {'full_text': str,
         'lines': [{'text': str, 'confidence': float, 'bbox': [x1, y1, x2, y2]}],
         'region_texts': [{'label': str, 'text': str, 'confidence': float, 'bbox': [..]}]}
        On any failure returns empty text and empty lists.
    """
    empty = {"full_text": "", "lines": [], "region_texts": []}
    try:
        # Fake lines stacked from top to bottom, 30 pixels apart.
        lines = []
        for index, text in enumerate(_FAKE_LINES):
            top = 10 + 30 * index
            lines.append(
                {"text": text, "confidence": 0.9, "bbox": [10, top, 400, top + 25]}
            )

        # One fake text entry for every region we were given.
        region_texts = []
        for region in regions or []:
            try:
                label = region["label"]
                region_texts.append(
                    {
                        "label": label,
                        "text": _FAKE_REGION_TEXT.get(label, ""),
                        "confidence": 0.9,
                        "bbox": [int(v) for v in region["bbox"]],
                    }
                )
            except Exception:
                logger.exception("skipping bad region %s", region)

        return {
            "full_text": "\n".join(_FAKE_LINES),
            "lines": lines,
            "region_texts": region_texts,
        }
    except Exception:
        logger.exception("run_ocr failed")
        return empty