"""Preprocessing of Sec. 3.2: brain-region extraction, intensity normalisation and resizing.

(1) Convert to grayscale, threshold the foreground, erode and dilate to remove small artefacts, take the largest
    contour and crop the image at its extreme top, bottom, left and right points with a small margin.
(2) Rescale intensities to [0, 255].
(3) Resize to 224 x 224 (the model repeats the grayscale channel to 224 x 224 x 3).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

SIZE = 224


def crop_brain(gray: np.ndarray, thresh: int = 45, margin: int = 5) -> np.ndarray:
    """Crop to the bounding extreme points of the largest foreground contour. Returns the input if none is found."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    c = max(contours, key=cv2.contourArea)
    left = int(c[c[:, :, 0].argmin()][0][0])
    right = int(c[c[:, :, 0].argmax()][0][0])
    top = int(c[c[:, :, 1].argmin()][0][1])
    bottom = int(c[c[:, :, 1].argmax()][0][1])
    h, w = gray.shape
    y0, y1 = max(top - margin, 0), min(bottom + margin, h)
    x0, x1 = max(left - margin, 0), min(right + margin, w)
    if y1 - y0 < 16 or x1 - x0 < 16:
        return gray
    return gray[y0:y1, x0:x1]


def normalise(gray: np.ndarray) -> np.ndarray:
    lo, hi = float(gray.min()), float(gray.max())
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.uint8)
    return ((gray.astype(np.float32) - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def preprocess_array(gray: np.ndarray, size: int = SIZE) -> np.ndarray:
    return cv2.resize(normalise(crop_brain(gray)), (size, size), interpolation=cv2.INTER_AREA)


def preprocess_file(path: str | Path, size: int = SIZE) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)  # unicode-safe on Windows
    if img is None:
        raise ValueError(f"could not read {path}")
    return preprocess_array(img, size)


def build_cache(paths: list[str], cache_file: str | Path, size: int = SIZE, workers: int = 8) -> np.ndarray:
    """Preprocess every image once and store an (N, size, size) uint8 array; reuse it when it already matches."""
    cache_file = Path(cache_file)
    index_file = cache_file.with_suffix(".paths.txt")
    if cache_file.exists() and index_file.exists() and index_file.read_text(encoding="utf-8").splitlines() == list(paths):
        return np.load(cache_file)
    with ThreadPoolExecutor(max_workers=workers) as ex:  # OpenCV releases the GIL
        arr = np.stack(list(ex.map(lambda p: preprocess_file(p, size), paths)))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, arr)
    index_file.write_text("\n".join(paths), encoding="utf-8")
    return arr
