# FILE: app/main.py
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Streamlit page: upload a photo, click Analyze, see the report as JSON.
# Start it from the repository root with:  streamlit run app/main.py
# The two lines above put the repository root on Python's search path so that
# 'from src...' imports work.
#
# IMPORTANT: do not put a bare text string (a docstring) anywhere above
# st.set_page_config. Streamlit would show it on the page and then refuse to
# run set_page_config, which must be the first Streamlit command.

import os
import tempfile

import streamlit as st

from src.pipeline import analyze_image

# The inspection context. Fixed for now; real input controls come later.
DEFAULT_CONTEXT = {
    "category": "food",
    "is_imported": False,
    "pdp_width_cm": None,
    "pdp_height_cm": None,
    "ocr_quality": "good",
}

# This MUST be the first Streamlit command in the file.
st.set_page_config(page_title="Legal Metrology Checker", layout="wide")
st.title("Legal Metrology Checker")
st.caption("Upload a photo of a packaged product and click Analyze.")

uploaded = st.file_uploader("Photo of the product pack", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, caption="Uploaded photo", width=400)

# The button is greyed out until a photo has been chosen.
if st.button("Analyze", type="primary", disabled=uploaded is None):
    with st.spinner("Analyzing..."):
        try:
            # Save the upload to a temporary folder (deleted automatically),
            # keeping its original file name so the report shows it.
            with tempfile.TemporaryDirectory() as temp_dir:
                safe_name = os.path.basename(uploaded.name) or "upload.jpg"
                temp_path = os.path.join(temp_dir, safe_name)
                with open(temp_path, "wb") as temp_file:
                    temp_file.write(uploaded.getvalue())
                st.session_state["report"] = analyze_image(temp_path, dict(DEFAULT_CONTEXT))
        except Exception as error:
            st.session_state["report"] = None
            st.error(f"Something went wrong: {error}")

# Show the last report (kept in session_state so it survives page refreshes).
report = st.session_state.get("report")
if report:
    if report.get("is_demo"):
        st.warning("DEMO MODE: the modules are placeholders, so these results are FAKE.")
    for warning_text in report.get("warnings", []):
        st.warning(f"Warning: {warning_text}")

    annotated_path = report.get("annotated_image_path")
    if annotated_path and os.path.exists(annotated_path):
        st.image(annotated_path, caption="Detected regions", width=400)

    st.subheader("Report (JSON)")
    st.json(report)
