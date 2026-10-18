"""Synthetic degradations applied to clean tiles.

Test (writes 3 degraded examples to work/debug/ so you can look at them):
    python -m src.degrade work/original/<some_tile>.png
"""
import sys

import cv2
import numpy as np

import config


def _read(path):
    img = cv2.imread(str(path))
    if img is None:
        raise Exception(f"Could not read image: {path}")
    return img


def add_gaussian_noise(input_path, output_path, seed):
    img = _read(input_path)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, config.NOISE_SIGMA, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), noisy)
    return output_path


def gaussian_blur(input_path, output_path):
    img = _read(input_path)
    k = config.BLUR_KSIZE
    blurred = cv2.GaussianBlur(img, (k, k), config.BLUR_SIGMA)
    cv2.imwrite(str(output_path), blurred)
    return output_path


def low_contrast(input_path, output_path):
    img = _read(input_path)
    out = cv2.convertScaleAbs(img, alpha=config.CONTRAST_ALPHA,
                              beta=config.CONTRAST_BETA)
    cv2.imwrite(str(output_path), out)
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.degrade <tile.png>")
        sys.exit()
    src = sys.argv[1]
    dbg = config.WORK_DIR / "debug"
    dbg.mkdir(parents=True, exist_ok=True)
    add_gaussian_noise(src, dbg / "test_noisy.png", config.SEED)
    gaussian_blur(src, dbg / "test_blurred.png")
    low_contrast(src, dbg / "test_lowcon.png")
    print("Wrote test_noisy.png, test_blurred.png, test_lowcon.png to", dbg)