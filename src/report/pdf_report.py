# FILE: src/report/pdf_report.py
"""PDF report builder.

PLACEHOLDER VERSION: writes a real but very small one-page PDF that shows the
file name, the overall status and a 'demo' notice. The real layout comes later.
"""

import logging
import os

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)


def generate_pdf(report: dict, output_path: str) -> str:
    """Write a PDF for this report and return the path that was written.

    Returns an empty string '' if the PDF could not be written.
    """
    try:
        # Make sure the destination folder exists.
        folder = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(folder, exist_ok=True)

        pdf = canvas.Canvas(str(output_path), pagesize=A4)
        _, page_height = A4

        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(50, page_height - 60, "Legal Metrology Checker - PLACEHOLDER REPORT")

        pdf.setFont("Helvetica", 11)
        status = report.get("validation", {}).get("overall_status", "UNKNOWN")
        pdf.drawString(50, page_height - 100, f"Image: {report.get('image_filename', '')}")
        pdf.drawString(50, page_height - 120, f"Overall status: {status}")
        pdf.drawString(50, page_height - 140, f"Demo data: {report.get('is_demo', False)}")

        pdf.showPage()
        pdf.save()
        return str(output_path)
    except Exception:
        logger.exception("generate_pdf failed")
        return ""
