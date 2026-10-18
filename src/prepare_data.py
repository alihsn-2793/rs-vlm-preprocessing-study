"""Builds the experiment sample from the Airbus aircraft dataset.

Run from the project root:
    python -m src.prepare_data

Expects:
    data/raw/annotations.csv   (columns: id, image_id, geometry, class)
    data/raw/ ... image files  (directly in raw/ or in raw/images/)

Produces:
    work/original/<tile_id>.png      (512x512 tiles, lossless PNG)
    data/questions.csv               (2 questions per tile, with ground truth)
    work/debug/*.png                 (3 tiles with boxes drawn, for a sanity check)
"""
import ast
import random
from collections import defaultdict

import pandas as pd
from PIL import Image, ImageDraw

import config

TILE = config.TILE_SIZE
ANN_PATH = config.RAW_DIR / "annotations.csv"
TILE_DIR = config.WORK_DIR / "original"
DEBUG_DIR = config.WORK_DIR / "debug"
QUESTIONS_CSV = config.ROOT / "data" / "questions.csv"

# bucket name, min aircraft, max aircraft
BUCKETS = [("0", 0, 0), ("1-2", 1, 2), ("3-5", 3, 5), ("6+", 6, 10**9)]

Q_PRESENCE = "Is there an aircraft in this image? Answer only Yes or No."
Q_COUNT = "How many aircraft are in this image? Answer with a single number."


def parse_box(geometry):
    pts = ast.literal_eval(geometry)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def index_images():
    index = {}
    for p in config.RAW_DIR.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            index[p.name] = p
    return index


def count_in_tile(boxes, x0, y0):
    """Number of aircraft fully inside the tile, or None if any aircraft
    is cut by the tile border (ambiguous ground truth -> tile rejected)."""
    x1, y1 = x0 + TILE, y0 + TILE
    inside = 0
    for bx0, by0, bx1, by1 in boxes:
        overlaps = bx0 < x1 and bx1 > x0 and by0 < y1 and by1 > y0
        if not overlaps:
            continue
        fully = bx0 >= x0 and by0 >= y0 and bx1 <= x1 and by1 <= y1
        if not fully:
            return None
        inside += 1
    return inside


def bucket_of(n):
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    return None


def main():
    ann = pd.read_csv(ANN_PATH)
    ann = ann[ann["class"].str.lower() == "airplane"].copy()
    ann["box"] = ann["geometry"].apply(parse_box)
    boxes_by_image = defaultdict(list)
    for image_id, box in zip(ann["image_id"], ann["box"]):
        boxes_by_image[image_id].append(box)

    files = index_images()
    missing = [i for i in boxes_by_image if i not in files]
    print(f"Annotated images: {len(boxes_by_image)} | found on disk: "
          f"{len(boxes_by_image) - len(missing)} | missing: {len(missing)}")
    if missing:
        print("First missing:", missing[:3])

    # 1. enumerate every valid candidate tile
    candidates = defaultdict(list)  # bucket -> list of dicts
    for image_id, boxes in boxes_by_image.items():
        if image_id not in files:
            continue
        with Image.open(files[image_id]) as im:
            w, h = im.size
        for y0 in range(0, h - TILE + 1, TILE):
            for x0 in range(0, w - TILE + 1, TILE):
                n = count_in_tile(boxes, x0, y0)
                if n is None:
                    continue
                candidates[bucket_of(n)].append(
                    {"image_id": image_id, "x0": x0, "y0": y0, "n": n})

    print("\nValid candidate tiles per bucket:")
    for name, _, _ in BUCKETS:
        print(f"  {name:>4}: {len(candidates[name])}")

    # 2. balanced sample, capped per source image
    rng = random.Random(config.RANDOM_SEED)
    per_bucket = config.N_IMAGES // len(BUCKETS)
    selected = []
    for name, _, _ in BUCKETS:
        pool = candidates[name][:]
        rng.shuffle(pool)
        used = defaultdict(int)
        chosen = []
        for c in pool:
            if used[c["image_id"]] >= config.MAX_PER_IMAGE_PER_BUCKET:
                continue
            chosen.append(c)
            used[c["image_id"]] += 1
            if len(chosen) == per_bucket:
                break
        if len(chosen) < per_bucket:
            print(f"WARNING: bucket {name} has only {len(chosen)} tiles "
                  f"(wanted {per_bucket})")
        for c in chosen:
            c["bucket"] = name
        selected.extend(chosen)

    # 3. save tiles (open each source image once)
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    by_image = defaultdict(list)
    for c in selected:
        by_image[c["image_id"]].append(c)

    rows = []
    debug_done = 0
    for image_id, tiles in by_image.items():
        with Image.open(files[image_id]) as im:
            im = im.convert("RGB")
            for c in tiles:
                tile_id = f"{image_id[:8]}_{c['x0']}_{c['y0']}"
                c["tile_id"] = tile_id
                tile = im.crop((c["x0"], c["y0"],
                                c["x0"] + TILE, c["y0"] + TILE))
                tile.save(TILE_DIR / f"{tile_id}.png")

                if debug_done < 3 and c["n"] > 0:
                    dbg = tile.copy()
                    d = ImageDraw.Draw(dbg)
                    for bx0, by0, bx1, by1 in boxes_by_image[image_id]:
                        if (bx0 >= c["x0"] and by0 >= c["y0"] and
                                bx1 <= c["x0"] + TILE and by1 <= c["y0"] + TILE):
                            d.rectangle([bx0 - c["x0"], by0 - c["y0"],
                                         bx1 - c["x0"], by1 - c["y0"]],
                                        outline="red", width=3)
                    dbg.save(DEBUG_DIR / f"{tile_id}_boxes.png")
                    debug_done += 1

                base = {"tile_id": tile_id, "image_id": image_id,
                        "x0": c["x0"], "y0": c["y0"],
                        "n_aircraft": c["n"], "bucket": c["bucket"]}
                rows.append({**base, "qtype": "presence",
                             "question": Q_PRESENCE,
                             "ground_truth": "yes" if c["n"] > 0 else "no"})
                rows.append({**base, "qtype": "count",
                             "question": Q_COUNT,
                             "ground_truth": str(c["n"])})

    QUESTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(QUESTIONS_CSV, index=False)
    print(f"\nSaved {len(selected)} tiles and {len(rows)} questions")
    print(f"Questions: {QUESTIONS_CSV}")
    print(f"Check the red boxes in: {DEBUG_DIR}")


if __name__ == "__main__":
    main()