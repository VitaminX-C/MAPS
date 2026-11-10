# FILE: src/analytics/stats.py
"""Statistics for the dashboard.

PLACEHOLDER VERSION: returns fixed fake numbers. The numbers add up:
4 pass + 4 fail + 2 review = 10 total.
pass_rate values are PERCENTAGES from 0 to 100.
"""


def get_stats() -> dict:
    """Return summary statistics about all stored inspections.

    Returns:
        {'total': int, 'pass': int, 'fail': int, 'review': int,
         'pass_rate': float,
         'top_violations': [{'rule_id': str, 'title': str, 'count': int}],
         'by_day': [{'date': str, 'pass': int, 'fail': int, 'review': int}],
         'by_category': [{'category': str, 'count': int, 'pass_rate': float}]}
    """
    return {
        "total": 10,
        "pass": 4,
        "fail": 4,
        "review": 2,
        "pass_rate": 40.0,
        "top_violations": [
            {"rule_id": "LM-004", "title": "Unit sale price declared", "count": 4},
            {"rule_id": "LM-003", "title": "Consumer care details present", "count": 2},
        ],
        "by_day": [
            {"date": "2026-11-08", "pass": 1, "fail": 2, "review": 0},
            {"date": "2026-11-09", "pass": 3, "fail": 2, "review": 2},
        ],
        "by_category": [
            {"category": "food", "count": 7, "pass_rate": 42.9},
            {"category": "non_food", "count": 3, "pass_rate": 33.3},
        ],
    }
