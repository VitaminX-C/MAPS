# FILE: src/extraction/extractor.py
"""Field extraction.

PLACEHOLDER VERSION: it ignores the OCR text and always returns the same fake
values for the 9 fields. (One field, unit_sale_price, is deliberately 'not
found' so that later screens have a missing field to show.)
The real version will use regex plus a spaCy EntityRuler.
"""


def _field(found: bool, raw_text, confidence: float, parsed: dict) -> dict:
    """Build one field entry in the exact shape the CONTRACT requires."""
    return {
        "found": found,
        "raw_text": raw_text,
        "confidence": confidence,
        "parsed": parsed,
    }


def extract_fields(ocr_result: dict) -> dict:
    """Pull the 9 legal-metrology fields out of the OCR result.

    Returns a dict with exactly these keys: product_name, net_quantity, mrp,
    mfg_date, expiry_date, manufacturer, consumer_care, country_of_origin,
    unit_sale_price. Every value is
    {'found': bool, 'raw_text': str|None, 'confidence': float, 'parsed': dict}.
    """
    return {
        "product_name": _field(
            True, "Sample Tea Biscuits", 0.9, {"name": "Sample Tea Biscuits"}
        ),
        "net_quantity": _field(
            True, "Net Qty: 250 g", 0.9, {"amount": 250.0, "unit": "g"}
        ),
        "mrp": _field(
            True,
            "MRP Rs. 45.00 (Incl. of all taxes)",
            0.9,
            {"amount": 45.0, "currency_ok": True, "inclusive_of_taxes": True},
        ),
        "mfg_date": _field(
            True,
            "Mfg Date: 09/2026",
            0.9,
            {"day": None, "month": 9, "year": 2026, "label_type": "mfg"},
        ),
        "expiry_date": _field(
            True,
            "Best Before: 03/2027",
            0.9,
            {"day": None, "month": 3, "year": 2027, "label_type": "best_before"},
        ),
        "manufacturer": _field(
            True,
            "Mfd. by: Sample Foods Pvt Ltd, 12 Industrial Area, Bengaluru 560001",
            0.9,
            {
                "name": "Sample Foods Pvt Ltd",
                "address": "12 Industrial Area, Bengaluru",
                "pincode": "560001",
                "role": "manufacturer",
            },
        ),
        "consumer_care": _field(
            True,
            "Customer Care: 1800-123-4567, care@samplefoods.example",
            0.9,
            {
                "phone": "1800-123-4567",
                "email": "care@samplefoods.example",
                "has_name_or_address": True,
            },
        ),
        "country_of_origin": _field(
            True, "Made in India", 0.9, {"country": "India"}
        ),
        "unit_sale_price": _field(
            False, None, 0.0, {"amount": None, "per_unit": None}
        ),
    }