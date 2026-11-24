"""Explanation panels (Sec. 3.6, 4.2): Grad-CAM of the baseline and the refined model, SHAP and LIME of the refined
model, for one test image per class."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import CLASSES
from .gradcam import heatmap
from .model import XCNN, to_input


def _shap_map(model: XCNN, img: torch.Tensor, background: torch.Tensor, target: int, device: str) -> np.ndarray:
    import shap

    model.eval()
    expl = shap.GradientExplainer(model, to_input(background.to(device)))
    vals = expl.shap_values(to_input(img[None].to(device)), nsamples=64)
    v = np.asarray(vals)
    # shap returns either a list per class of (1, 3, H, W) or one array (1, 3, H, W, C)
    if v.ndim == 5 and v.shape[-1] == len(CLASSES):
        v = v[0, ..., target]
    else:
        v = np.asarray(vals[target])[0]
    v = v.sum(0)
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(v, sigma=3)  # pixel attributions are too sparse to read unsmoothed


def _lime_map(model: XCNN, img: torch.Tensor, target: int, device: str, samples: int = 300) -> np.ndarray:
    from lime import lime_image

    def f(batch_rgb: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(batch_rgb[..., 0].astype(np.uint8))
        with torch.no_grad():
            return torch.softmax(model(to_input(x.to(device))), 1).cpu().numpy()

    rgb = np.repeat(img.numpy()[..., None], 3, axis=2)
    import contextlib
    import io

    with contextlib.redirect_stderr(io.StringIO()):  # silence LIME's progress bar
        exp = lime_image.LimeImageExplainer(random_state=0).explain_instance(
            rgb, f, labels=(target,), top_labels=None, hide_color=0, num_samples=samples)
    weights = dict(exp.local_exp[target])
    seg = exp.segments
    out = np.zeros(seg.shape, dtype=float)
    for s, w in weights.items():
        out[seg == s] = w
    return out


def explanation_panel(baseline: XCNN, refined: XCNN, images: torch.Tensor, labels: np.ndarray,
                      background: torch.Tensor, out_file: str, device: str = "cpu") -> None:
    rows = len(images)
    fig, axes = plt.subplots(rows, 5, figsize=(14, 2.9 * rows))
    axes = np.atleast_2d(axes)
    titles = ["MRI", "Grad-CAM baseline", "Grad-CAM refined", "SHAP refined", "LIME refined"]
    for r in range(rows):
        img, y = images[r], int(labels[r])
        gb = heatmap(baseline, img, -1, y, device)
        gr = heatmap(refined, img, -1, y, device)
        sh = _shap_map(refined, img, background, y, device)
        li = _lime_map(refined, img, y, device)
        base = img.numpy()
        panels = [None, gb, gr, sh, li]
        for c in range(5):
            ax = axes[r, c]
            ax.imshow(base, cmap="gray")
            if c in (1, 2):
                ax.imshow(panels[c], cmap="jet", alpha=0.45)
            elif c in (3, 4):
                lim = float(np.percentile(np.abs(panels[c]), 99.5)) or 1.0
                ax.imshow(panels[c], cmap="bwr", alpha=0.55, vmin=-lim, vmax=lim)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
        axes[r, 0].set_ylabel(CLASSES[y], fontsize=11)
    plt.tight_layout()
    fig.savefig(out_file, dpi=110)
    plt.close(fig)


def relevance_chart(relevance: list[float], filters: list[int], keep: list[int], tau: float, out_file: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = ["#2f7d5b" if i in keep else "#c0504d" for i in range(len(relevance))]
    ax.bar([f"block {i + 1}\n({f})" for i, f in enumerate(filters)], relevance, color=colors)
    ax.set_yscale("log")
    ax.axhline(tau, color="#333", ls="--", lw=1)
    ax.text(len(relevance) - 0.5, tau, f" tau = {tau:.3g}", va="bottom", ha="right", fontsize=9)
    ax.set_ylabel("Grad-CAM relevance m_l (log scale)")
    ax.set_title("Layer-wise relevance of the baseline (red = removed)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_file, dpi=130)
    plt.close(fig)
