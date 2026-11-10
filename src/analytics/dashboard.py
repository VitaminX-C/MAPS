# FILE: src/analytics/dashboard.py
"""The analytics dashboard, drawn inside a Streamlit app.

PLACEHOLDER VERSION: shows the (fake) numbers from get_stats() with simple
metrics, a table and bar charts.
"""

import pandas as pd
import streamlit as st

from src.analytics.stats import get_stats


def render_dashboard() -> None:
    """Draw the dashboard. Shows a red error box instead of crashing."""
    try:
        stats = get_stats()

        st.header("Analytics dashboard")

        # Four headline numbers side by side.
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total inspections", stats["total"])
        col2.metric("Passed", stats["pass"])
        col3.metric("Failed", stats["fail"])
        col4.metric("Pass rate (%)", stats["pass_rate"])

        st.subheader("Top violations")
        if stats["top_violations"]:
            st.dataframe(pd.DataFrame(stats["top_violations"]), hide_index=True)
        else:
            st.info("No violations recorded yet.")

        st.subheader("Results per day")
        if stats["by_day"]:
            by_day = pd.DataFrame(stats["by_day"]).set_index("date")
            st.bar_chart(by_day[["pass", "fail", "review"]])
        else:
            st.info("No inspections recorded yet.")

        st.subheader("By product category")
        if stats["by_category"]:
            st.dataframe(pd.DataFrame(stats["by_category"]), hide_index=True)
        else:
            st.info("No inspections recorded yet.")
    except Exception as error:
        st.error(f"The dashboard could not be drawn: {error}")
