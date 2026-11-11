# FILE: scripts/try_detector.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Try the detector on one photo.
#
# Usage (from the repository root):
#     python scripts/try_detector.py path/to/photo.jpg
#
# Steps: clean the photo -> detect regions -> print them -> save the annotated
# picture to outputs/annotated_try.jpg

import cv2

from src.vision.preprocess import load_and_preprocess
from src.vision.detector import detect_regions, draw_regions

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    """Run the try-out. Returns 0 on success, 1 on a problem."""
    # 1. Read the image path typed after the script name.
    if len(sys.argv) != 2:
        print("Usage: python scripts/try_detector.py path/to/photo.jpg")
        return 1
    image_path = sys.argv[1]

    try:
        # 2. Load and clean the photo.
        pre = load_and_preprocess(image_path)
        print(f"Image size: {pre['width']} x {pre['height']}")
        print(f"Warnings: {pre['warnings'] if pre['warnings'] else 'none'}")

        if pre["image_bgr"] is None:
            print("Could not read the image. Check the path and try again.")
            return 1

        # 3. Detect the label regions.
        regions = detect_regions(pre["image_bgr"])
        print(f"Regions found: {len(regions)}")
        for region in regions:
            print(
                f"  - {region['label']:<20} "
                f"confidence={region['confidence']:.3f}  bbox={region['bbox']}"
            )
        if not regions:
            print("  (No regions. Is models/best.pt present? See any WARNING above.)")

        # 4. Draw the boxes and save the picture.
        annotated = draw_regions(pre["image_bgr"], regions)
        out_dir = REPO_ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "annotated_try.jpg"
        if cv2.imwrite(str(out_path), annotated):
            print(f"Saved annotated picture to: {out_path}")
        else:
            print(f"Could not save the picture to: {out_path}")
            return 1
        return 0

    except Exception as exc:  # noqa: BLE001
        print(f"Something went wrong: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
