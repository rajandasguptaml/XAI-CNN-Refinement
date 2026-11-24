"""Summarise results.json across seeds and draw all figures from the saved models (no retraining).

Usage:
    python scripts/make_figures.py --results results --config configs/default.yaml
Writes results/summary.md, results/summary.json, results/fig_accuracy.png and, per protocol, the relevance chart and
the explanation panel of the first seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xairefine import CLASSES  # noqa: E402
from xairefine.data import load_metadata, make_splits  # noqa: E402
from xairefine.explain import explanation_panel, relevance_chart  # noqa: E402
from xairefine.model import XCNN  # noqa: E402
from xairefine.preprocess import build_cache  # noqa: E402
from xairefine.refine import threshold  # noqa: E402

SETS = [("test", "D1 test (full)"), ("test_leakfree", "D1 test (unseen)"), ("d2_all", "D2 (full)"),
        ("d2_unseen", "D2 (unseen)")]
PAPER = {  # Table 4 of the paper, Dataset-1 and Dataset-2 full sets
    ("baseline", "test"): 97.10, ("refined", "test"): 98.21, ("baseline", "d2_all"): 92.33, ("refined", "d2_all"): 94.72,
}


def summarise(rows: list[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        for key, _ in SETS:
            recs.append({"protocol": r["protocol"], "model": r["model"], "seed": r["seed"], "set": key,
                         "n": r[key]["n"], "accuracy": 100 * r[key]["accuracy"], "f1": 100 * r[key]["f1"],
                         "params": r["params"], "ms": r["ms_per_image"]})
    df = pd.DataFrame(recs)
    g = df.groupby(["protocol", "model", "set"])
    out = g.agg(n=("n", "first"), seeds=("seed", "nunique"), acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                f1_mean=("f1", "mean"), f1_std=("f1", "std"), params=("params", "mean"), ms=("ms", "mean")).reset_index()
    return out.fillna(0.0)


def md_table(s: pd.DataFrame) -> str:
    lines = ["| Protocol | Model | " + " | ".join(label for _, label in SETS) + " | Params | ms / image |",
             "|---|---|" + "---|" * (len(SETS) + 2)]
    for (proto, model), g in s.groupby(["protocol", "model"], sort=False):
        cells = []
        for key, _ in SETS:
            r = g[g["set"] == key].iloc[0]
            cells.append(f"{r.acc_mean:.2f} ± {r.acc_std:.2f} (n={int(r.n)})")
        r0 = g.iloc[0]
        lines.append(f"| {proto} | {model} | " + " | ".join(cells) + f" | {int(r0.params):,} | {r0.ms:.3f} |")
    return "\n".join(lines)


def accuracy_figure(s: pd.DataFrame, out_file: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, proto in zip(axes, ["paper", "clean"]):
        sub = s[s["protocol"] == proto]
        x = np.arange(len(SETS))
        for j, (model, color) in enumerate([("baseline", "#4c72b0"), ("refined", "#2f9e6e")]):
            vals = [sub[(sub.model == model) & (sub.set == k)] for k, _ in SETS]
            means = [v.acc_mean.iloc[0] if len(v) else np.nan for v in vals]
            stds = [v.acc_std.iloc[0] if len(v) else 0 for v in vals]
            ax.bar(x + (j - 0.5) * 0.36, means, 0.36, yerr=stds, capsize=3, color=color, label=model)
            for xi, m, sd in zip(x, means, stds):
                if np.isfinite(m):
                    ax.text(xi + (j - 0.5) * 0.36, m + (sd if np.isfinite(sd) else 0) + 1.2, f"{m:.1f}", ha="center",
                            fontsize=8)
        if proto == "paper":
            for j, model in enumerate(["baseline", "refined"]):
                for k, (key, _) in enumerate(SETS):
                    if (model, key) in PAPER:
                        ax.plot(k + (j - 0.5) * 0.36, PAPER[(model, key)], marker="_", ms=22, mew=2.5, color="#c0392b",
                                label="reported in paper" if (j, k) == (0, 0) else None)
        ax.set_xticks(x, [label for _, label in SETS])
        ax.set_title({"paper": "Paper protocol: random split by image", "clean": "Leak-free protocol: group-aware split"}[proto])
        ax.set_ylim(30, 104)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy (%), mean ± SD over seeds")
    axes[0].legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_file, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-panels", action="store_true")
    args = ap.parse_args()
    res = Path(args.results)
    rows = json.loads((res / "results.json").read_text())
    s = summarise(rows)
    (res / "summary.json").write_text(s.to_json(orient="records", indent=1))
    table = md_table(s)
    (res / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    accuracy_figure(s, res / "fig_accuracy.png")

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for r in rows:
        if r["model"] == "refined":
            run_dir = res / f"{r['protocol']}_seed{r['seed']}"
            relevance_chart(r["relevance"], cfg["model"]["filters"], r["kept_blocks"], r["tau"],
                            str(run_dir / "relevance.png"))
    if args.no_panels:
        return
    meta = load_metadata(cfg["data"]["index_csv"], cfg["data"]["image_root"])
    cache = torch.from_numpy(build_cache(meta["path"].tolist(), Path(cfg["cache_dir"]) / "images224.npy"))
    pos = {p: i for i, p in enumerate(meta["path"])}
    by_key = {(r["protocol"], r["seed"], r["model"]): r for r in rows}
    for proto in sorted({r["protocol"] for r in rows}):
        seed = min(r["seed"] for r in rows if r["protocol"] == proto)
        if (proto, seed, "refined") not in by_key:
            continue
        run_dir = res / f"{proto}_seed{seed}"
        models = {}
        for name in ("baseline", "refined"):
            m = XCNN(by_key[(proto, seed, name)]["filters"])
            m.load_state_dict(torch.load(run_dir / f"{name}.pt", map_location="cpu"))
            models[name] = m.to(device).eval()
        sp = make_splits(meta, proto, seed)
        te = sp["test_leakfree"]
        rng = np.random.default_rng(7)
        pick = [int(rng.choice(te.index[te["y"] == c])) for c in range(len(CLASSES)) if (te["y"] == c).any()]
        imgs = cache[[pos[p] for p in te.loc[pick, "path"]]]
        tr_idx = [pos[p] for p in sp["train"]["path"]]
        bg = cache[np.random.default_rng(0).choice(tr_idx, 40, replace=False)]
        explanation_panel(models["baseline"], models["refined"], imgs, te.loc[pick, "y"].to_numpy(), bg,
                          str(run_dir / "explanations.png"), device=device)
        print("panel", run_dir / "explanations.png")


if __name__ == "__main__":
    main()
