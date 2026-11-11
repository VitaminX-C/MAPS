# FILE: scripts/try_ocr.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try the OCR on one photo (no detector needed).
#
# Usage (from the repository root):
#     python scripts/try_ocr.py path/to/photo.jpg
#
# It prints the text it found and saves the full result as JSON in
#     tests/fixtures/ocr_<photo name>.json

import json

from src.ocr.ocr_engine import run_ocr
from src.vision.preprocess import load_and_preprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv):
    """Run preprocess + OCR on argv[1]. Returns 0 on success, 1 on a problem."""
    # Windows terminals can choke on Hindi letters; ask for UTF-8 output.
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    if len(argv) != 2:
        print('Usage: python scripts/try_ocr.py path/to/photo.jpg')
        return 1

    image_path = argv[1]
    if not os.path.isfile(image_path):
        print(f'File not found: {image_path}')
        return 1

    # Step 1: clean the image
    pre = load_and_preprocess(image_path)
    image_bgr = pre.get('image_bgr')
    if image_bgr is None:
        print(f"Could not read the image. Warnings: {pre.get('warnings')}")
        return 1
    print(f"Image size: {pre.get('width')} x {pre.get('height')}")
    print(f"Image warnings: {pre.get('warnings')}")

    # Step 2: OCR (no regions, so only the full image is read)
    result = run_ocr(image_bgr, [])

    print('\n===== FULL TEXT =====')
    print(result['full_text'] if result['full_text'] else '(no text found)')
    print('=====================')
    print(f"Lines found: {len(result['lines'])}")

    # Step 3: save the whole result as JSON
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(REPO_ROOT, 'tests', 'fixtures')
    out_path = os.path.join(out_dir, f'ocr_{stem}.json')
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f'Saved: {out_path}')
    except OSError as error:
        print(f'Could not save JSON: {error}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))