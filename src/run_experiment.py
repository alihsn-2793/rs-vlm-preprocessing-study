"""Builds all image versions and asks the VLM every question. Resumable.

    python -m src.run_experiment --build-only --limit 3   # only make images
    python -m src.run_experiment --limit 2                # tiny real test
    python -m src.run_experiment                          # full run

Only SUCCESSFUL answers are written to results/answers.csv. Failed calls
(server errors, out-of-memory, empty output) are retried and, if they still
fail, logged to results/logs/errors.log and NOT saved, so a rerun asks them
again. The run stops if several calls in a row fail.
"""
import argparse
import csv
import random
import sys
import time
import zlib

import pandas as pd

import config
from src import degrade, image_ops, vlm

ANSWERS = config.RESULTS_DIR / "answers.csv"
ERRLOG = config.RESULTS_DIR / "logs" / "errors.log"
FIELDS = ["tile_id", "bucket", "n_aircraft", "condition", "qtype", "question",
          "ground_truth", "raw_answer", "seconds", "error", "model"]

RETRIES = 2                    # extra attempts for a failed call
RETRY_WAIT = 10                # seconds between attempts
MAX_CONSECUTIVE_FAILURES = 4   # stop the run after this many failures in a row


def cond_path(cond, tile_id):
    return config.WORK_DIR / cond / f"{tile_id}.png"


# (new condition, parent condition, function(src, dst, seed))
def _noise(s, d, seed): degrade.add_gaussian_noise(s, d, seed)
def _blur(s, d, seed): degrade.gaussian_blur(s, d)
def _lowcon(s, d, seed): degrade.low_contrast(s, d)
def _denoise(s, d, seed): image_ops.denoise_image(str(s), str(d))
def _sharpen(s, d, seed): image_ops.sharpen_image(str(s), str(d))
def _enhance(s, d, seed): image_ops.enhance_image(str(s), str(d))


RECIPES = [
    ("noisy", "original", _noise),
    ("noisy_denoised", "noisy", _denoise),
    ("blurred", "original", _blur),
    ("blurred_sharpened", "blurred", _sharpen),
    ("lowcon", "original", _lowcon),
    ("lowcon_enhanced", "lowcon", _enhance),
]
CLEAN_RECIPES = [
    ("clean_denoised", "original", _denoise),
    ("clean_sharpened", "original", _sharpen),
    ("clean_enhanced", "original", _enhance),
]


def build_versions(tile_id):
    seed = config.SEED + zlib.crc32(tile_id.encode())
    recipes = RECIPES + (CLEAN_RECIPES if config.INCLUDE_FIXES_ON_CLEAN else [])
    for cond, parent, fn in recipes:
        dst = cond_path(cond, tile_id)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        fn(cond_path(parent, tile_id), dst, seed)


def ask_with_retry(image_path, question):
    res = None
    for attempt in range(RETRIES + 1):
        res = vlm.ask(image_path, question)
        if res["raw"] and not res["error"]:
            return res
        if attempt < RETRIES:
            time.sleep(RETRY_WAIT)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N tiles")
    ap.add_argument("--build-only", action="store_true",
                    help="make image versions, do not call the VLM")
    args = ap.parse_args()

    q = pd.read_csv(config.ROOT / "data" / "questions.csv",
                    dtype={"ground_truth": str})
    tiles = q["tile_id"].drop_duplicates().tolist()
    random.Random(config.RANDOM_SEED).shuffle(tiles)
    if args.limit:
        tiles = tiles[:args.limit]

    conds = list(config.CONDITIONS)
    if config.INCLUDE_FIXES_ON_CLEAN:
        conds += config.CLEAN_FIX_CONDITIONS

    print(f"Building image versions for {len(tiles)} tiles ...")
    for t in tiles:
        build_versions(t)
    print("Image versions ready.")
    if args.build_only:
        return

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ERRLOG.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if ANSWERS.exists() and ANSWERS.stat().st_size > 0:
        prev = pd.read_csv(ANSWERS, dtype=str, keep_default_na=False)
        ok = prev[(prev.raw_answer != "") & (prev.error == "")]
        bad = len(prev) - len(ok)
        if bad:
            print(f"STOP: answers.csv contains {bad} empty or failed rows. "
                  "Clean the file first (keep only rows with a non-empty "
                  "raw_answer and an empty error column).")
            sys.exit(1)
        done = set(zip(ok.tile_id, ok.condition, ok.qtype))

    jobs = []
    for t in tiles:
        rows = q[q.tile_id == t]
        for c in conds:
            for _, r in rows.iterrows():
                if (t, c, r.qtype) not in done:
                    jobs.append((t, c, r))
    print(f"{len(jobs)} questions to ask ({len(done)} already done).")

    new_file = not ANSWERS.exists() or ANSWERS.stat().st_size == 0
    start = time.time()
    fails = 0
    with open(ANSWERS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, (t, c, r) in enumerate(jobs, 1):
            res = ask_with_retry(cond_path(c, t), r.question)
            if not res["raw"] or res["error"]:
                fails += 1
                with open(ERRLOG, "a", encoding="utf-8") as ef:
                    ef.write(f"{t}|{c}|{r.qtype}|{res['error'] or 'empty answer'}\n")
                print(f"[{i}/{len(jobs)}] FAILED {t} {c} {r.qtype}: "
                      f"{(res['error'] or 'empty answer')[:120]}")
                if fails >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\nStopping: {fails} failures in a row. Likely out of "
                          "memory or Ollama not running. Close other programs, "
                          "restart Ollama (or the laptop), then run the same "
                          "command again. Nothing saved so far is lost.")
                    sys.exit(1)
                continue
            fails = 0
            w.writerow({
                "tile_id": t, "bucket": r.bucket, "n_aircraft": r.n_aircraft,
                "condition": c, "qtype": r.qtype, "question": r.question,
                "ground_truth": r.ground_truth, "raw_answer": res["raw"],
                "seconds": res["seconds"], "error": res["error"],
                "model": config.MODEL,
            })
            f.flush()
            elapsed = time.time() - start
            eta = elapsed / i * (len(jobs) - i) / 60
            print(f"[{i}/{len(jobs)}] {t} {c} {r.qtype}: "
                  f"{res['raw']!r} (truth {r.ground_truth}) "
                  f"{res['seconds']}s | ETA {eta:.0f} min")
    print("Done.")


if __name__ == "__main__":
    main()