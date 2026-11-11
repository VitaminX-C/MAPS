# FILE: src/ocr/ocr_engine.py
"""
OCR engine for the Legal Metrology Checker.

Public function (from the CONTRACT):
    run_ocr(image_bgr, regions) -> {'full_text', 'lines', 'region_texts'}

What it does, in plain English:
  1. Loads the EasyOCR reader ONE time (it is slow to load) and reuses it.
  2. Reads ALL the text on the whole image.
  3. Joins words that sit on the same visual row into one line
     (so "MRP", "Rs.", "45.00" become the single line "MRP Rs. 45.00").
  4. Re-reads each detected region (name, MRP, dates...) on its own,
     enlarged and sharpened, which usually gives better text.
  5. Throws away any text EasyOCR is less than 20% sure about.
  6. Converts Hindi/Arabic-style digits (e.g. २०० or ٢٠٠) into normal
     digits (200), because the Hindi model sometimes reads English
     digits that way.
  7. Never crashes: on any failure it returns empty text and empty lists.

All boxes are [x1, y1, x2, y2] in pixels of the FULL image.
"""

import threading
import time
import unicodedata

import numpy as np

from src.vision.preprocess import prepare_for_ocr

# ---------------------------------------------------------------------------
# Settings (change them here, nowhere else)
# ---------------------------------------------------------------------------
OCR_LANGUAGES = ['en', 'hi']   # English + Hindi
MIN_CONFIDENCE = 0.2           # text below this is dropped
CROP_PADDING_PX = 10           # extra pixels around each region before cropping
SAME_ROW_OVERLAP = 0.5         # words overlapping more than 50% vertically = same row

# The EasyOCR reader lives here once created. None means "not created yet".
_reader = None
_reader_lock = threading.Lock()  # stops two threads creating it twice


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _empty_result():
    """Return the safe default result required by the CONTRACT."""
    return {'full_text': '', 'lines': [], 'region_texts': []}


def _log_time(step_name, started_at):
    """Print how many seconds a step took, e.g. '[ocr] full-image OCR: 2.31 s'."""
    print(f"[ocr] {step_name}: {time.perf_counter() - started_at:.2f} s")


def _is_valid_image(image):
    """True only for a non-empty numpy array that is 2-D (gray) or 3-D (colour)."""
    return (
        isinstance(image, np.ndarray)
        and image.ndim in (2, 3)
        and image.size > 0
    )


def _normalise_digits(text):
    """
    Turn digits from other scripts into normal 0-9 digits.

    Example: 'Net Wt. २०० g' -> 'Net Wt. 200 g'
             (Devanagari digits)  (normal digits)

    Why: the Hindi model often reads English digits as Devanagari digits,
    and the rest of the app only understands 0-9.
    Letters, punctuation and normal digits are left exactly as they are.
    """
    changed = []
    for character in text:
        if character.isascii():
            changed.append(character)
            continue
        # unicodedata.decimal gives the number a digit stands for (0-9),
        # or None if the character is not a real decimal digit.
        value = unicodedata.decimal(character, None)
        changed.append(str(value) if value is not None else character)
    return ''.join(changed)


def _get_reader():
    """
    Create the EasyOCR reader the first time it is needed, then reuse it.

    The import of easyocr is done here (not at the top of the file) because
    importing it is slow, and we only want to pay that cost when OCR is
    really needed. The first ever call may also download the language
    models (internet needed once).
    """
    global _reader
    if _reader is not None:
        return _reader
    with _reader_lock:
        if _reader is None:
            import easyocr  # local import on purpose (slow import)
            _reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
    return _reader


def _box_to_xyxy(box, offset_x=0.0, offset_y=0.0, scale_x=1.0, scale_y=1.0):
    """
    Convert an EasyOCR 4-corner box [[x,y],[x,y],[x,y],[x,y]] into
    [x1, y1, x2, y2] whole numbers.

    offset_* and scale_* are used to move a box from a crop back into
    full-image coordinates: full = offset + crop_value * scale.
    """
    xs = [offset_x + float(point[0]) * scale_x for point in box]
    ys = [offset_y + float(point[1]) * scale_y for point in box]
    return [
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs))),
        int(round(max(ys))),
    ]


def _read_words(reader, image, offset_x=0.0, offset_y=0.0, scale_x=1.0, scale_y=1.0):
    """
    Run EasyOCR on one image and return a list of word dicts:
        {'text': str, 'confidence': float, 'bbox': [x1, y1, x2, y2]}

    - Digits from other scripts are converted to 0-9.
    - Text with confidence below MIN_CONFIDENCE is dropped.
    - Empty / malformed items are skipped.
    - If EasyOCR itself raises an error, the error is passed up to the caller.
    """
    raw_results = reader.readtext(image)
    words = []
    for item in raw_results:
        try:
            box = item[0]
            text = _normalise_digits(str(item[1]).strip())
            confidence = float(item[2])
            if not text:
                continue
            if not (confidence >= MIN_CONFIDENCE):  # also rejects NaN
                continue
            bbox = _box_to_xyxy(box, offset_x, offset_y, scale_x, scale_y)
        except (TypeError, ValueError, IndexError):
            continue  # one broken item must not stop the rest
        words.append({
            'text': text,
            'confidence': round(confidence, 4),
            'bbox': bbox,
        })
    return words


def _vertical_overlap_ratio(box_a, box_b):
    """
    How much two boxes overlap vertically, as a fraction (0 to 1) of the
    SHORTER box's height. 1.0 means the shorter box is fully inside the
    other one's vertical range.
    """
    top = max(box_a[1], box_b[1])
    bottom = min(box_a[3], box_b[3])
    overlap = max(0, bottom - top)
    shorter_height = max(1, min(box_a[3] - box_a[1], box_b[3] - box_b[1]))
    return overlap / shorter_height


def _union_box(box_a, box_b):
    """Smallest box that contains both boxes."""
    return [
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3]),
    ]


def _merge_into_lines(words):
    """
    Group words into lines.

    Steps:
      1. Sort words top-to-bottom (by vertical centre), then left-to-right.
      2. Walk through them; a word joins the current row if it overlaps
         the row vertically by more than 50%, otherwise it starts a new row.
      3. Inside each row sort words left-to-right and join them with spaces.
      4. Line confidence = average of its word confidences.
    Returns a list of {'text', 'confidence', 'bbox'} sorted top-to-bottom.
    """
    if not words:
        return []

    ordered = sorted(
        words,
        key=lambda w: ((w['bbox'][1] + w['bbox'][3]) / 2.0, w['bbox'][0]),
    )

    rows = []
    for word in ordered:
        if rows and _vertical_overlap_ratio(rows[-1]['box'], word['bbox']) > SAME_ROW_OVERLAP:
            rows[-1]['words'].append(word)
            rows[-1]['box'] = _union_box(rows[-1]['box'], word['bbox'])
        else:
            rows.append({'box': list(word['bbox']), 'words': [word]})

    lines = []
    for row in rows:
        row_words = sorted(row['words'], key=lambda w: w['bbox'][0])
        text = ' '.join(w['text'] for w in row_words)
        average = sum(w['confidence'] for w in row_words) / len(row_words)
        lines.append({
            'text': text,
            'confidence': round(average, 4),
            'bbox': row['box'],
        })

    lines.sort(key=lambda line: (line['bbox'][1], line['bbox'][0]))
    return lines


def _clamp_box(bbox, width, height, padding):
    """
    Add `padding` pixels around bbox and keep the result inside the image.
    Returns [x1, y1, x2, y2] as ints, or None if the box is unusable
    (wrong shape, not numbers, or smaller than 2 pixels after clamping).
    """
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return None
    x1 = min(max(x1 - padding, 0), width)
    y1 = min(max(y1 - padding, 0), height)
    x2 = min(max(x2 + padding, 0), width)
    y2 = min(max(y2 + padding, 0), height)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [x1, y1, x2, y2]


def _read_region_words(reader, image, padded_box):
    """
    OCR one cropped region and return its words in FULL-IMAGE coordinates.

    prepare_for_ocr() enlarges the crop, so the boxes EasyOCR returns are in
    "enlarged crop" pixels. We work out the shrink factor from the two image
    sizes and move each box back to where it sits on the original image.
    """
    x1, y1, x2, y2 = padded_box
    crop = image[y1:y2, x1:x2]
    prepared = prepare_for_ocr(crop)
    if not _is_valid_image(prepared):
        raise ValueError('prepare_for_ocr did not return a usable image')

    crop_h, crop_w = crop.shape[:2]
    prepared_h, prepared_w = prepared.shape[:2]
    scale_x = crop_w / prepared_w
    scale_y = crop_h / prepared_h
    return _read_words(reader, prepared, x1, y1, scale_x, scale_y)


def _process_region(reader, image, region):
    """
    Build one region_texts entry for one detected region, or return None
    if the region is malformed or no text was found in it.
    """
    if not isinstance(region, dict):
        return None

    height, width = image.shape[:2]
    bbox = region.get('bbox')
    padded_box = _clamp_box(bbox, width, height, CROP_PADDING_PX)
    if padded_box is None:
        return None
    tight_box = _clamp_box(bbox, width, height, 0) or padded_box

    words = _read_region_words(reader, image, padded_box)
    if not words:
        return None

    lines = _merge_into_lines(words)
    average = sum(w['confidence'] for w in words) / len(words)
    return {
        'label': str(region.get('label', '')),
        'text': '\n'.join(line['text'] for line in lines),
        'confidence': round(average, 4),
        'bbox': tight_box,
    }


# ---------------------------------------------------------------------------
# The public function
# ---------------------------------------------------------------------------
def run_ocr(image_bgr, regions):
    """
    Read text from a product-pack image.

    Args:
        image_bgr: the image as a numpy array (OpenCV BGR), or None.
        regions: list of {'label', 'confidence', 'bbox'} from the detector
                 (may be an empty list - then only the full image is read).

    Returns:
        {
          'full_text': str,
          'lines': [{'text', 'confidence', 'bbox'}],
          'region_texts': [{'label', 'text', 'confidence', 'bbox'}]
        }
        On ANY failure returns {'full_text': '', 'lines': [], 'region_texts': []}.
    """
    try:
        if not _is_valid_image(image_bgr):
            print('[ocr] no usable image given - returning empty result')
            return _empty_result()

        # Step 1: get (or create) the reader
        started = time.perf_counter()
        reader = _get_reader()
        _log_time('load EasyOCR reader', started)

        # Step 2: OCR the whole image
        started = time.perf_counter()
        words = _read_words(reader, image_bgr)
        _log_time(f'full-image OCR ({len(words)} words)', started)

        # Step 3: join words into lines
        started = time.perf_counter()
        lines = _merge_into_lines(words)
        full_text = '\n'.join(line['text'] for line in lines)
        _log_time(f'merge into lines ({len(lines)} lines)', started)

        # Step 4: OCR each region separately
        started = time.perf_counter()
        region_texts = []
        for region in (regions or []):
            region_started = time.perf_counter()
            label = region.get('label', '?') if isinstance(region, dict) else '?'
            try:
                entry = _process_region(reader, image_bgr, region)
            except Exception as error:  # one bad region must not stop the others
                print(f"[ocr] region '{label}' failed: {error}")
                continue
            if entry is not None:
                region_texts.append(entry)
            _log_time(f"region '{label}'", region_started)
        _log_time(f'all regions ({len(region_texts)} with text)', started)

        return {
            'full_text': full_text,
            'lines': lines,
            'region_texts': region_texts,
        }

    except Exception as error:
        print(f'[ocr] ERROR - returning empty result: {error}')
        return _empty_result()