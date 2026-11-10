<!-- FILE: docs/CONTRACT.md -->
CONTRACT (boxes are pixel lists [x1, y1, x2, y2]; everything is plain Python dicts and lists):
REGION LABELS (exactly 6): product_name, net_quantity, mrp, date_block, manufacturer_block, consumer_care_block
FIELDS (exactly 9): product_name, net_quantity, mrp, mfg_date, expiry_date, manufacturer, consumer_care, country_of_origin, unit_sale_price

src/vision/preprocess.py
  load_and_preprocess(image_path: str) -> {'image_bgr': numpy array, 'width': int, 'height': int, 'warnings': [str]}
    warnings use: 'blurry', 'too_dark', 'too_bright', 'low_resolution', 'unreadable_image'.
    On failure return {'image_bgr': None, 'width': 0, 'height': 0, 'warnings': ['unreadable_image']}
  prepare_for_ocr(crop_bgr) -> grayscale numpy array (upscaled and contrast-enhanced)
src/vision/detector.py
  detect_regions(image_bgr) -> [{'label': str, 'confidence': float, 'bbox': [x1,y1,x2,y2]}]   (return [] if models/best.pt is missing or anything fails)
  draw_regions(image_bgr, regions) -> image_bgr with labelled boxes drawn
src/ocr/ocr_engine.py
  run_ocr(image_bgr, regions: list) -> {'full_text': str, 'lines': [{'text': str, 'confidence': float, 'bbox': [..]}], 'region_texts': [{'label': str, 'text': str, 'confidence': float, 'bbox': [..]}]}
  (on failure return empty text and empty lists)
src/extraction/extractor.py
  extract_fields(ocr_result: dict) -> dict with the 9 FIELDS as keys. Every value is {'found': bool, 'raw_text': str or None, 'confidence': float 0 to 1, 'parsed': dict}
  parsed per field:
    product_name: {'name': str|None}
    net_quantity: {'amount': float|None, 'unit': str|None}   unit normalised to g, kg, ml, l, cm, m, nos; anything else (oz, lb...) kept as written in lower case
    mrp: {'amount': float|None, 'currency_ok': bool, 'inclusive_of_taxes': bool}
    mfg_date and expiry_date: {'day': int|None, 'month': int|None, 'year': int|None, 'label_type': 'mfg'|'packed'|'best_before'|'use_by'|'expiry'|None}
    manufacturer: {'name': str|None, 'address': str|None, 'pincode': str|None, 'role': 'manufacturer'|'packer'|'importer'|'marketer'|None}
    consumer_care: {'phone': str|None, 'email': str|None, 'has_name_or_address': bool}
    country_of_origin: {'country': str|None}
    unit_sale_price: {'amount': float|None, 'per_unit': str|None}
src/rules/validator.py   (rules live in src/rules/rule_book.json)
  validate(fields: dict, context: dict) -> {'overall_status': 'PASS'|'FAIL'|'REVIEW', 'checks': [{'rule_id': str, 'title': str, 'status': 'pass'|'fail'|'review'|'not_applicable', 'message': str, 'legal_reference': str, 'field': str}], 'summary': {'passed': int, 'failed': int, 'review': int, 'not_applicable': int}}
  context = {'category': 'food'|'non_food', 'is_imported': bool, 'pdp_width_cm': float|None, 'pdp_height_cm': float|None, 'ocr_quality': 'good'|'poor'}
src/pipeline.py
  analyze_image(image_path: str, context: dict) -> report = {'inspection_id': int|None, 'timestamp': ISO string, 'image_filename': str, 'context': dict, 'warnings': [str], 'regions': list, 'ocr_text': str, 'fields': dict, 'validation': dict, 'annotated_image_path': str|None, 'processing_seconds': float, 'is_demo': bool}
src/report/pdf_report.py
  generate_pdf(report: dict, output_path: str) -> str   (the path written)
src/analytics/db.py
  init_db(); save_inspection(report: dict) -> int; get_all_inspections() -> list[dict]; get_inspection(id: int) -> dict; delete_all()
src/analytics/stats.py
  get_stats() -> {'total': int, 'pass': int, 'fail': int, 'review': int, 'pass_rate': float, 'top_violations': [{'rule_id': str, 'title': str, 'count': int}], 'by_day': [{'date': str, 'pass': int, 'fail': int, 'review': int}], 'by_category': [{'category': str, 'count': int, 'pass_rate': float}]}
src/analytics/dashboard.py
  render_dashboard() -> None   (draws the dashboard inside a Streamlit app)
app/main.py: the Streamlit app, started with: streamlit run app/main.py
