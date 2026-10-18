"""Scores answers and produces the tables and figures for the paper.

Run after run_experiment and metrics:
    python -m src.analyze

Outputs (results/tables and results/figures):
    accuracy_by_condition.csv      accuracy per condition and question type
    paired_comparisons.csv         damage effect, fix effect, with CI and McNemar p
    rq3_quality_vs_correctness.csv does image-quality gain predict answer gain?
    count_accuracy_by_bucket.csv   count accuracy by true aircraft number
    fig_accuracy.png, fig_quality_vs_answer.png
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

import config

TAB = config.RESULTS_DIR / "tables"
FIG = config.RESULTS_DIR / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

NUM_WORDS = {"zero": 0, "none": 0, "no": 0, "one": 1, "two": 2, "three": 3,
             "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
             "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
             "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20}

DAMAGE = [("original", "noisy"), ("original", "blurred"), ("original", "lowcon")]
FIXES = [("noisy", "noisy_denoised"), ("blurred", "blurred_sharpened"),
         ("lowcon", "lowcon_enhanced")]
CLEAN = [("original", "clean_denoised"), ("original", "clean_sharpened"),
         ("original", "clean_enhanced")]


def parse_presence(text):
    m = re.search(r"\b(yes|no)\b", str(text).lower())
    return m.group(1) if m else None


def parse_count(text):
    t = str(text).lower()
    m = re.search(r"\d+", t)
    if m:
        return int(m.group())
    m = re.search(r"\b(" + "|".join(NUM_WORDS) + r")\b", t)
    return NUM_WORDS[m.group(1)] if m else None


def score_row(r):
    raw = "" if pd.isna(r.raw_answer) else r.raw_answer
    if r.qtype == "presence":
        pred = parse_presence(raw)
        return pd.Series({"correct": int(pred == r.ground_truth),
                          "within1": np.nan, "invalid": int(pred is None)})
    pred = parse_count(raw)
    gt = int(r.ground_truth)
    if pred is None:
        return pd.Series({"correct": 0, "within1": 0, "invalid": 1})
    return pd.Series({"correct": int(pred == gt),
                      "within1": int(abs(pred - gt) <= 1), "invalid": 0})


def paired(df, a, b, qtype, col):
    sub = df[df.qtype == qtype]
    pv = sub.pivot_table(index="tile_id", columns="condition",
                         values=col, aggfunc="first")
    if a not in pv or b not in pv:
        return None
    pv = pv[[a, b]].dropna()
    n = len(pv)
    if n == 0:
        return None
    x, y = pv[a].to_numpy(), pv[b].to_numpy()
    diff = y - x
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, (2000, n))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    better = int(((x == 0) & (y == 1)).sum())
    worse = int(((x == 1) & (y == 0)).sum())
    p = binomtest(better, better + worse, 0.5).pvalue if better + worse else 1.0
    return {"a": a, "b": b, "qtype": qtype, "metric": col, "n": n,
            "acc_a": round(x.mean(), 3), "acc_b": round(y.mean(), 3),
            "diff": round(diff.mean(), 3), "ci_low": round(lo, 3),
            "ci_high": round(hi, 3), "improved": better, "worsened": worse,
            "p_mcnemar": round(p, 4)}


def main():
    df = pd.read_csv(config.RESULTS_DIR / "answers.csv",
                     dtype={"ground_truth": str})
    df = pd.concat([df, df.apply(score_row, axis=1)], axis=1)

    order = config.CONDITIONS + config.CLEAN_FIX_CONDITIONS
    df["condition"] = pd.Categorical(df["condition"],
                                     [c for c in order if c in set(df.condition)],
                                     ordered=True)

    # ---- Table 1: accuracy per condition ----
    acc = (df.groupby(["qtype", "condition"], observed=True)
             .agg(n=("correct", "size"), accuracy=("correct", "mean"),
                  within1=("within1", "mean"), invalid_rate=("invalid", "mean"),
                  mean_seconds=("seconds", "mean"))
             .round(3).reset_index())
    acc.to_csv(TAB / "accuracy_by_condition.csv", index=False)
    print("\n=== Accuracy by condition ===")
    print(acc.to_string(index=False))

    # ---- Table 2: paired comparisons ----
    rows = []
    for kind, pairs in [("damage", DAMAGE), ("fix", FIXES), ("fix_on_clean", CLEAN)]:
        for a, b in pairs:
            for qt, col in [("presence", "correct"), ("count", "correct"),
                            ("count", "within1")]:
                r = paired(df, a, b, qt, col)
                if r:
                    r["kind"] = kind
                    rows.append(r)
    pc = pd.DataFrame(rows)
    pc.to_csv(TAB / "paired_comparisons.csv", index=False)
    print("\n=== Paired comparisons ===")
    print(pc.to_string(index=False))

    # ---- Table 3: count accuracy by true aircraft number ----
    cnt = df[df.qtype == "count"]
    bk = (cnt.groupby(["bucket", "condition"], observed=True)["correct"]
             .agg(["size", "mean"]).round(3).reset_index())
    bk.to_csv(TAB / "count_accuracy_by_bucket.csv", index=False)

    # ---- Table 4: image quality vs answer change (RQ3) ----
    mpath = config.RESULTS_DIR / "image_metrics.csv"
    scatter_data = []
    if mpath.exists():
        m = pd.read_csv(mpath)
        ssim = m.pivot(index="tile_id", columns="condition", values="ssim")
        psnr = m.pivot(index="tile_id", columns="condition", values="psnr")
        rows = []
        for a, b in FIXES:
            for qt in ["presence", "count"]:
                pv = (df[df.qtype == qt]
                      .pivot_table(index="tile_id", columns="condition",
                                   values="correct", aggfunc="first"))
                if a not in pv or b not in pv or a not in ssim or b not in ssim:
                    continue
                d = pd.DataFrame({
                    "d_correct": pv[b] - pv[a],
                    "d_ssim": ssim[b] - ssim[a],
                    "d_psnr": psnr[b] - psnr[a]}).dropna()
                if d.empty:
                    continue
                d["pair"] = f"{a}->{b}"
                d["qtype"] = qt
                scatter_data.append(d)
                up = d[d.d_ssim > 0]
                try:
                    rho, p = spearmanr(d.d_ssim, d.d_correct)
                except Exception:
                    rho, p = np.nan, np.nan
                rows.append({
                    "pair": f"{a}->{b}", "qtype": qt, "n": len(d),
                    "mean_dSSIM": round(d.d_ssim.mean(), 3),
                    "mean_dPSNR": round(d.d_psnr.mean(), 2),
                    "mean_dCorrect": round(d.d_correct.mean(), 3),
                    "spearman_rho": round(rho, 3) if rho == rho else np.nan,
                    "spearman_p": round(p, 4) if p == p else np.nan,
                    "ssim_up_n": len(up),
                    "ssim_up_answer_worse": int((up.d_correct < 0).sum()),
                    "ssim_up_answer_better": int((up.d_correct > 0).sum())})
        rq3 = pd.DataFrame(rows)
        rq3.to_csv(TAB / "rq3_quality_vs_correctness.csv", index=False)
        print("\n=== RQ3: image quality vs answer correctness ===")
        print(rq3.to_string(index=False))
    else:
        print("\n(No image_metrics.csv: run `python -m src.metrics` first.)")

    # ---- Figure 1: accuracy per condition ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, qt, title in zip(axes, ["presence", "count"],
                             ["Presence (Yes/No)", "Count (exact match)"]):
        s = acc[acc.qtype == qt]
        ax.bar(s.condition.astype(str), s.accuracy)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(FIG / "fig_accuracy.png", dpi=200)
    plt.close()

    # ---- Figure 2: change in SSIM vs change in correctness ----
    if scatter_data:
        sd = pd.concat(scatter_data)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        rng = np.random.default_rng(1)
        for ax, qt in zip(axes, ["presence", "count"]):
            for pair, g in sd[sd.qtype == qt].groupby("pair"):
                ax.scatter(g.d_ssim,
                           g.d_correct + rng.normal(0, 0.03, len(g)),
                           label=pair, alpha=0.7)
            ax.axhline(0, color="grey", lw=0.8)
            ax.axvline(0, color="grey", lw=0.8)
            ax.set_xlabel("Change in SSIM after fix")
            ax.set_ylabel("Change in correctness (jittered)")
            ax.set_title(qt)
            ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(FIG / "fig_quality_vs_answer.png", dpi=200)
        plt.close()

    print("\nTables:", TAB, "\nFigures:", FIG)


if __name__ == "__main__":
    main()