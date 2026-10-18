"""Extra analysis for the paper: size and direction of counting errors, and
how stable each tile's answer is across the image conditions.

Run after run_experiment and analyze:
    python -W ignore -m src.extra_analysis

Outputs (results/tables):
    count_error_by_condition.csv   MAE, signed error, under/over-count shares
    count_abs_error_paired.csv     paired change in absolute error, CI, Wilcoxon p
    count_mae_by_bucket.csv        MAE by true aircraft group and condition
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config
from src.analyze import parse_count, DAMAGE, FIXES, CLEAN

TAB = config.RESULTS_DIR / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def paired_abs(df, a, b):
    pv = df.pivot_table(index="tile_id", columns="condition",
                        values="abs_err", aggfunc="first")
    if a not in pv or b not in pv:
        return None
    pv = pv[[a, b]].dropna()
    n = len(pv)
    if n == 0:
        return None
    d = (pv[b] - pv[a]).to_numpy()          # positive = larger error in b
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, (2000, n))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    try:
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return {"a": a, "b": b, "n": n,
            "mae_a": round(pv[a].mean(), 3), "mae_b": round(pv[b].mean(), 3),
            "diff_mae": round(d.mean(), 3), "ci_low": round(lo, 3),
            "ci_high": round(hi, 3), "tiles_worse": int((d > 0).sum()),
            "tiles_better": int((d < 0).sum()), "p_wilcoxon": round(p, 4)}


def main():
    df = pd.read_csv(config.RESULTS_DIR / "answers.csv",
                     dtype={"ground_truth": str})
    df = df[df.qtype == "count"].copy()
    df["truth"] = df.ground_truth.astype(int)
    df["pred"] = pd.to_numeric(df.raw_answer.apply(parse_count),
                               errors="coerce")
    df["err"] = df["pred"] - df["truth"]              # negative = undercount
    df["abs_err"] = df.err.abs()

    order = [c for c in config.CONDITIONS + config.CLEAN_FIX_CONDITIONS
             if c in set(df.condition)]

    # ---- error size and direction per condition ----
    rows = []
    for c in order:
        s = df[df.condition == c]
        rows.append({"condition": c, "n": len(s),
                     "exact": round((s.err == 0).mean(), 3),
                     "MAE": round(s.abs_err.mean(), 3),
                     "median_abs_error": s.abs_err.median(),
                     "mean_signed_error": round(s.err.mean(), 3),
                     "undercount_share": round((s.err < 0).mean(), 3),
                     "overcount_share": round((s.err > 0).mean(), 3)})
    t1 = pd.DataFrame(rows)
    t1.to_csv(TAB / "count_error_by_condition.csv", index=False)
    print("\n=== Count error by condition ===")
    print(t1.to_string(index=False))

    # ---- paired change in absolute error ----
    rows = []
    for kind, pairs in [("damage", DAMAGE), ("fix", FIXES),
                        ("fix_on_clean", CLEAN)]:
        for a, b in pairs:
            r = paired_abs(df, a, b)
            if r:
                r["kind"] = kind
                rows.append(r)
    t2 = pd.DataFrame(rows)
    t2.to_csv(TAB / "count_abs_error_paired.csv", index=False)
    print("\n=== Paired change in absolute count error ===")
    print(t2.to_string(index=False))

    # ---- MAE by true count group ----
    t3 = (df.groupby(["bucket", "condition"])["abs_err"].mean().round(3)
            .unstack("condition"))
    t3 = t3[[c for c in order if c in t3.columns]]
    t3.to_csv(TAB / "count_mae_by_bucket.csv")
    print("\n=== MAE by true count group ===")
    print(t3.to_string())

    # ---- stability of the answer across conditions ----
    pred = df.pivot_table(index="tile_id", columns="condition",
                          values="pred", aggfunc="first")
    core = [c for c in config.CONDITIONS if c in pred]
    sub = pred[core].dropna()
    gt = df.groupby("tile_id")["truth"].first().reindex(sub.index)
    distinct = sub.nunique(axis=1)
    same_all = distinct == 1
    orig_wrong = sub["original"] != gt
    persistent = orig_wrong & same_all
    print("\n=== Stability across the 7 image versions ===")
    print(f"Tiles with the identical count in all versions: "
          f"{int(same_all.sum())} of {len(sub)}")
    print(f"Tiles counted wrong in the original: {int(orig_wrong.sum())}")
    print(f"  of those, wrong with the same answer in every version: "
          f"{int(persistent.sum())}")
    unstable = sub[distinct > 1].copy()
    unstable.insert(0, "true", gt[unstable.index])
    print("\nTiles whose count changed between versions:")
    print(unstable.to_string())


if __name__ == "__main__":
    main()