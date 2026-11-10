# Offline Compliance Scanner for Legal Metrology

Manual inspection and compliance checking of packaged commodities by enforcement agencies is currently slow, error-prone, and resource-intensive. Common violations such as **missing declarations**, **incorrect font sizes**, and **improper MRP declarations** frequently go unnoticed or take too long to audit.

Our solution is an **AI-Powered Offline Compliance Scanner** designed to automate label scanning, extract critical text declarations, and validate them against the Legal Metrology Rules, 2011—cutting inspection time from **5 minutes down to 3 seconds** using standard smartphones.

## Key Innovations

1. **Offline-First Architecture:**
   Equipped with lightweight, on-device OCR (PaddleOCR) and a local SQLite queue, enabling field officers to conduct basic validations completely offline in basements, warehouses, or remote godowns without internet connectivity.

2. **Decoupled Rule Engine:**
   Compliance rules are maintained in independent `JSON`/`YAML` configuration files. Regulatory updates made by administrators require zero code re-compilation or app store redeployment.

## Compliance Workflow

1. **Web & Mobile Upload:** Field officers capture or upload product label images via the mobile or web app.

2. **Smart Scan & Extract:** Computer Vision and OCR models automatically detect Regions of Interest (ROIs) and extract textual details.

3. **Rules-Based Validation:** Extracted text and font metrics are parsed and validated against the Legal Metrology (Packaged Commodities) Rules, 2011.

4. **Compliance Reporting:** Generates instant audit-ready PDFs and geo-tagged evidence records.

5. **Monitoring Dashboard:** A unified web portal with Role-Based Access Control (RBAC) to track violations, compliance scores, and regional trend maps.

## Tech Stack

| Layer | Technologies Used | 
| ----- | ----- | 
| **Frontend** | React Native (Mobile App), React.js (Web Dashboard) | 
| **Backend** | Python, FastAPI | 
| **AI / Computer Vision** | YOLOv8 (ROI Detection), OpenCV (Image Preprocessing & Cropping), PaddleOCR | 
| **NLP** | Regex, spaCy (Entity extraction for Net Weight, MRP, Dates, etc.) | 
| **Database** | PostgreSQL (Cloud), SQLite (Local Offline Queue) | 
| **Cloud & Deployment** | AWS (EC2 + S3), Docker | 
| **Authentication** | JWT + Role-Based Access Control (RBAC) | 

## Implementation Roadmap

* **Phase 1: Data & Rules (Weeks 1–2)**
  * Digitize Legal Metrology rules into structured `JSON`/`YAML` formats.
  * Collect 500+ diverse product label images and annotate Regions of Interest (ROIs).

* **Phase 2: AI Pipeline (Weeks 3–5)**
  * Train YOLOv8 for declaration block detection.
  * Implement OpenCV cropping and cylindrical unwrapping.
  * Integrate OCR + NLP pipelines for extracting net weight, MRP, manufacturer details, and dates.

* **Phase 3: App & Dashboard (Weeks 6–8)**
  * Develop the React Native mobile scanning app with on-device offline OCR.
  * Build the React.js web monitoring dashboard with RBAC.

* **Phase 4: System Integration (Weeks 9–10)**
  * Connect Mobile App, FastAPI Backend, and Decoupled Rule Engine.
  * Dockerize the application and deploy to AWS Cloud (EC2 & S3).

## Technical Challenges & Mitigations

* **Challenge 1: Curved and reflective packaging surfaces distort scans.**
  * *Solution:* Implement camera-level AR guidance to prompt officers to capture flat label areas, and apply cylindrical surface unwrapping via OpenCV prior to running font-size measurements.

* **Challenge 2: On-device OCR struggles with sub-millimeter print under poor lighting.**
  * *Solution:* Limit offline scans to binary presence checks, reserving detailed millimeter font-height measurements for when the device syncs with the cloud processing pipeline.

## Impact & Benefits

* **Faster Inspections:** Reduces manual checking time from ~5 minutes to ~3 seconds per product.
* **Better Consumer Protection:** Accurately detects missing, misleading, or non-compliant declarations.
* **Stronger Enforcement:** Empowers officials with geo-tagged, audit-ready digital evidence to prioritize inspections.
* **Reduced Compliance Costs:** Assists businesses and manufacturers in pre-validating packaging designs before incurring penalties.
* **Data-Driven Governance:** Creates digital records, compliance scores, and violation trend maps (aligning with the National Legal Metrology Portal - *eMaap*).
* **Paperless & Scalable:** Eliminates manual paperwork while enabling large-scale nationwide monitoring.

## References & Research

1. **Legal Metrology Overview:** Department of Consumer Affairs – [consumeraffairs.gov.in](https://consumeraffairs.gov.in/pages/legal-metrology-overview)
2. **Packaged Commodities Rules, 2011:** Mandatory declarations (MRP, Net Quantity, Date of Manufacture, Manufacturer details) – [wbconsumers.gov.in](https://wbconsumers.gov.in)
3. **Object & Font Measurement with OpenCV:** PyImageSearch – [Measuring size of objects in an image with OpenCV](https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/)
4. **National Legal Metrology Portal (eMaap):** PIB, Govt. of India (Dec 2024) – Unified digital enforcement portal (PRID 2080446).
5. **National Consumer Helpline (NCH):** Consumer grievance and complaint data – [consumerhelpline.gov.in](https://consumerhelpline.gov.in)
6. **GovTech & Economic Impact:** World Bank Group – [GovTech and the Future of the Public Sector](https://worldbank.org/en/topic/governance/brief/govtech)
