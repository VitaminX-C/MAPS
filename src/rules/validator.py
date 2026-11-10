# FILE: src/rules/validator.py
"""Rule checking against the Legal Metrology (Packaged Commodities) Rules, 2011.

PLACEHOLDER VERSION: it does NOT read src/rules/rule_book.json and does NOT
look at the fields. It always returns the same 5 fake checks (one for each
status) so that reports and dashboards have something to display.
"""

import logging

logger = logging.getLogger(__name__)

_FAKE_REFERENCE = "Legal Metrology (Packaged Commodities) Rules, 2011 (placeholder reference)"


def _check(rule_id: str, title: str, status: str, message: str, field: str) -> dict:
    """Build one check entry in the exact shape the CONTRACT requires."""
    return {
        "rule_id": rule_id,
        "title": title,
        "status": status,
        "message": message,
        "legal_reference": _FAKE_REFERENCE,
        "field": field,
    }


def validate(fields: dict, context: dict) -> dict:
    """Check the extracted fields against the rule book.

    Returns:
        {'overall_status': 'PASS'|'FAIL'|'REVIEW',
         'checks': [...],
         'summary': {'passed': int, 'failed': int, 'review': int, 'not_applicable': int}}
        On any failure returns overall_status 'REVIEW' with no checks.
    """
    safe_default = {
        "overall_status": "REVIEW",
        "checks": [],
        "summary": {"passed": 0, "failed": 0, "review": 0, "not_applicable": 0},
    }
    try:
        checks = [
            _check("LM-001", "Net quantity declared", "pass",
                   "Placeholder: net quantity found.", "net_quantity"),
            _check("LM-002", "MRP declared as inclusive of all taxes", "pass",
                   "Placeholder: MRP found with 'Incl. of all taxes'.", "mrp"),
            _check("LM-003", "Consumer care details present", "review",
                   "Placeholder: a person should confirm the phone number.", "consumer_care"),
            _check("LM-004", "Unit sale price declared", "fail",
                   "Placeholder: unit sale price was not found.", "unit_sale_price"),
            _check("LM-005", "Country of origin (imported goods only)", "not_applicable",
                   "Placeholder: product is not marked as imported.", "country_of_origin"),
        ]

        # Count how many checks have each status.
        summary = {"passed": 0, "failed": 0, "review": 0, "not_applicable": 0}
        status_to_key = {
            "pass": "passed",
            "fail": "failed",
            "review": "review",
            "not_applicable": "not_applicable",
        }
        for check in checks:
            summary[status_to_key[check["status"]]] += 1

        # Overall result: any fail -> FAIL; otherwise any review -> REVIEW; else PASS.
        if summary["failed"] > 0:
            overall = "FAIL"
        elif summary["review"] > 0:
            overall = "REVIEW"
        else:
            overall = "PASS"

        return {"overall_status": overall, "checks": checks, "summary": summary}
    except Exception:
        logger.exception("validate failed")
        return safe_default
