"""Run the full pipeline: preprocess, train the baseline, measure Grad-CAM relevance, remove low-relevance blocks,
retrain, evaluate on all test sets, and draw the explanation figures.

Usage:
    python scripts/run_experiments.py --config configs/default.yaml
    python scripts/run_experiments.py --config configs/default.yaml --protocols clean --seeds 0 --epochs 2   # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xairefine import CLASSES  # noqa: E402
from xairefine.data import check_images, describe, load_metadata, make_splits  # noqa: E402
from xairefine.explain import explanation_panel, relevance_chart  # noqa: E402
from xairefine.gradcam import layer_relevance  # noqa: E402
from xairefine.metrics import scores  # noqa: E402
from xairefine.model import XCNN  # noqa: E402
from xairefine.preprocess import build_cache  # noqa: E402
from xairefine.refine import blocks_to_keep, refined_filters, threshold  # noqa: E402
from xairefine.train import fit, predict  # noqa: E402

EVAL_SETS = ["test", "test_leakfree", "d2_all", "d2_unseen"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--protocols", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    protocols = args.protocols or cfg["protocols"]
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    epochs = args.epochs or cfg["train"]["epochs"]
    out = Path(args.out or cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_file = open(out / "log.txt", "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"=== run {time.strftime('%Y-%m-%d %H:%M')} | device {device} | torch {torch.__version__} | epochs {epochs}")
    meta = load_metadata(cfg["data"]["index_csv"], cfg["data"]["image_root"])
    check_images(meta)
    t = time.time()
    cache = build_cache(meta["path"].tolist(), Path(cfg["cache_dir"]) / "images224.npy")
    log(f"preprocessed {len(meta)} images in {time.time() - t:.0f}s")
    pos = {p: i for i, p in enumerate(meta["path"])}
    all_imgs = torch.from_numpy(cache)

    def tensors(df):
        idx = [pos[p] for p in df["path"]]
        return all_imgs[idx], torch.tensor(df["y"].to_numpy(), dtype=torch.long)

    results = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else []
    done = {(r["protocol"], r["seed"], r["model"]) for r in results}

    for protocol in protocols:
        for seed in seeds:
            splits = make_splits(meta, protocol, seed)
            log(f"\n--- protocol {protocol} | seed {seed}\n" + describe(splits).to_string(index=False))
            data = {k: tensors(v) for k, v in splits.items()}
            run_dir = out / f"{protocol}_seed{seed}"
            run_dir.mkdir(exist_ok=True)

            def train_and_eval(name: str, filters: list[int], extra: dict) -> XCNN:
                model = XCNN(filters)
                if (protocol, seed, name) in done and (run_dir / f"{name}.pt").exists():
                    model.load_state_dict(torch.load(run_dir / f"{name}.pt", map_location="cpu"))
                    return model.to(device)
                log(f"{name}: filters {filters}, {model.n_params():,} parameters")
                info = fit(model, *data["train"], *data["val"], epochs=epochs, batch_size=cfg["train"]["batch_size"],
                           lr=cfg["train"]["lr"], seed=seed, device=device, log=log)
                torch.save(model.state_dict(), run_dir / f"{name}.pt")
                t0 = time.time()
                predict(model, data["test"][0], device=device)
                ms_per_image = (time.time() - t0) * 1000 / max(len(data["test"][0]), 1)
                row = {"protocol": protocol, "seed": seed, "model": name, "filters": filters,
                       "params": model.n_params(), "train_seconds": round(info["seconds"], 1),
                       "best_epoch": info["best_epoch"], "val_acc": info["best_val_acc"],
                       "ms_per_image": round(ms_per_image, 3), **extra}
                for s in EVAL_SETS:
                    x, y = data[s]
                    row[s] = scores(y.numpy(), predict(model, x, device=device))
                    log(f"  {s:<14} n={row[s]['n']:>4} acc {row[s]['accuracy']:.4f} f1 {row[s]['f1']:.4f}")
                results.append(row)
                (out / "results.json").write_text(json.dumps(results, indent=1))
                (run_dir / f"{name}_history.json").write_text(json.dumps(info["history"]))
                return model

            base_filters = cfg["model"]["filters"]
            baseline = train_and_eval("baseline", base_filters, {})

            xv, yv = data["val"]
            rel = layer_relevance(baseline, xv, yv, device=device)
            p = cfg["refine"]["percentile"]
            tau = threshold(rel, p)
            keep = blocks_to_keep(rel, p, cfg["refine"]["min_keep"])
            log(f"relevance m_l: {np.round(rel, 5).tolist()} | tau(p={p}) = {tau:.5g} | keep blocks {keep}")
            relevance_chart(rel.tolist(), base_filters, keep, tau, str(run_dir / "relevance.png"))
            refined = train_and_eval("refined", refined_filters(base_filters, keep),
                                     {"relevance": rel.tolist(), "tau": tau, "kept_blocks": keep})

            if seed == seeds[0]:
                te = splits["test_leakfree"]
                pick = [te.index[te["y"] == c][0] for c in range(len(CLASSES)) if (te["y"] == c).any()]
                imgs = data["test_leakfree"][0][pick]
                bg = data["train"][0][torch.randperm(len(data["train"][0]))[:40]]
                try:
                    explanation_panel(baseline, refined, imgs, te.loc[pick, "y"].to_numpy(), bg,
                                      str(run_dir / "explanations.png"), device=device)
                except Exception as e:  # explanation figures must not lose the trained results
                    log(f"explanation panel failed: {e!r}")
    log("done")


if __name__ == "__main__":
    main()
