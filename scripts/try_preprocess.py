# FILE: scripts/try_preprocess.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Try the preprocessing step on one photo.

Usage (from the repository root):
    python scripts/try_preprocess.py path/to/photo.jpg

It prints the warnings and the image size, and saves the cleaned image to
outputs/clean_try.jpg so you can open it and look at it.
"""

import cv2

from src.vision.preprocess import load_and_preprocess

OUTPUT_PATH = os.path.join('outputs', 'clean_try.jpg')


def main():
    """Run load_and_preprocess on the image given on the command line."""
    if len(sys.argv) != 2:
        print('Usage: python scripts/try_preprocess.py path/to/photo.jpg')
        return 1

    image_path = sys.argv[1]
    result = load_and_preprocess(image_path)

    print('Image path :', image_path)
    print('Warnings   :', result['warnings'] if result['warnings'] else 'none')
    print('Size       : %d x %d pixels (width x height)'
          % (result['width'], result['height']))

    if result['image_bgr'] is None:
        print('Could not read this image, so nothing was saved.')
        return 1

    # Make sure the outputs folder exists, then save the cleaned image.
    os.makedirs('outputs', exist_ok=True)
    saved = cv2.imwrite(OUTPUT_PATH, result['image_bgr'],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not saved:
        print('Could not save the cleaned image to', OUTPUT_PATH)
        return 1

    print('Saved cleaned image to', OUTPUT_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main())