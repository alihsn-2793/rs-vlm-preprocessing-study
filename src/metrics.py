"""Computes PSNR and SSIM of every generated version against the clean original.

Run after run_experiment (or after run_experiment --build-only):
    python -m src.metrics
"""
import cv2
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import config


def main():
    q = pd.read_csv(config.ROOT / "data" / "questions.csv")
    tiles = q["tile_id"].drop_duplicates().tolist()
    conds = [c for c in config.CONDITIONS if c != "original"]
    conds += config.CLEAN_FIX_CONDITIONS

    rows = []
    for t in tiles:
        ref_path = config.WORK_DIR / "original" / f"{t}.png"
        ref = cv2.imread(str(ref_path))
        if ref is None:
            continue
        for c in conds:
            p = config.WORK_DIR / c / f"{t}.png"
            if not p.exists():
                continue
            img = cv2.imread(str(p))
            psnr = peak_signal_noise_ratio(ref, img, data_range=255)
            ssim = structural_similarity(ref, img, channel_axis=2,
                                         data_range=255)
            rows.append({"tile_id": t, "condition": c,
                         "psnr": round(float(psnr), 4),
                         "ssim": round(float(ssim), 4)})

    out = config.RESULTS_DIR / "image_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    if len(df):
        print(df.groupby("condition")[["psnr", "ssim"]].mean().round(3))


if __name__ == "__main__":
    main()