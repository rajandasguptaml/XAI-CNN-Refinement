import numpy as np
import pandas as pd
import torch

from xairefine.data import make_splits
from xairefine.gradcam import heatmap, layer_relevance
from xairefine.metrics import scores
from xairefine.model import XCNN, to_input
from xairefine.preprocess import crop_brain, preprocess_array
from xairefine.refine import blocks_to_keep, refined_filters, threshold
from xairefine.train import fit, predict


def _phantom(h=300, w=260):
    img = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    img[((yy - 150) / 100) ** 2 + ((xx - 140) / 80) ** 2 <= 1] = 160  # "brain"
    img[3:6, 3:6] = 255  # small artefact removed by erosion
    return img


def test_crop_finds_largest_region_and_resizes():
    c = crop_brain(_phantom())
    assert 195 <= c.shape[0] <= 215 and 155 <= c.shape[1] <= 175
    out = preprocess_array(_phantom())
    assert out.shape == (224, 224) and out.dtype == np.uint8 and out.max() == 255


def test_crop_returns_input_when_empty():
    blank = np.zeros((50, 60), dtype=np.uint8)
    assert crop_brain(blank).shape == (50, 60)


def test_model_shapes_for_baseline_and_pruned():
    x = to_input(torch.randint(0, 255, (2, 224, 224), dtype=torch.uint8))
    assert x.shape == (2, 3, 224, 224) and x.max() <= 1
    for filters in ([8, 16, 32, 64, 128, 256], [32, 64, 128, 256], [16, 256]):
        m = XCNN(filters)
        assert m(x).shape == (2, 4)
    assert XCNN([32, 64, 128, 256]).n_params() < XCNN().n_params()


def test_gradcam_relevance_and_heatmap():
    torch.manual_seed(0)
    m = XCNN([4, 8, 16])
    imgs = torch.randint(0, 255, (6, 64, 64), dtype=torch.uint8)
    rel = layer_relevance(m, imgs, torch.tensor([0, 1, 2, 3, 0, 1]), batch_size=4)
    assert rel.shape == (3,) and np.all(rel >= 0) and np.isfinite(rel).all()
    h = heatmap(m, imgs[0], block=-1)
    assert h.shape == (64, 64) and 0 <= h.min() and h.max() <= 1


def test_refinement_rule():
    rel = np.array([0.9, 0.1, 0.5, 0.05, 0.7, 0.3])
    keep = blocks_to_keep(rel, 25)
    assert 1 not in keep and 3 not in keep and len(keep) == 4
    assert refined_filters([8, 16, 32, 64, 128, 256], keep) == [8, 32, 128, 256]
    assert threshold(rel, 0) == rel.min()
    assert len(blocks_to_keep(rel, 99, min_keep=2)) >= 2


def test_metrics():
    s = scores(np.array([0, 1, 2, 3, 3]), np.array([0, 1, 2, 3, 0]))
    assert s["accuracy"] == 0.8 and s["n"] == 5 and abs(s["recall"] - 0.875) < 1e-9
    assert np.array(s["confusion"]).sum() == 5


def _meta():
    rows = []
    for i in range(80):
        ds = "msoud" if i < 60 else "sartaj"
        group = i % 50  # sartaj rows 60-79 share groups 10-29 with msoud rows
        rec = ["train", "validation", "test"][i % 3] if ds == "msoud" else "train"
        rows.append({"image_id": f"{ds}/{i}.jpg", "dataset": ds, "label": "glioma", "y": i % 4, "filename": f"{i}.jpg",
                     "path": f"/x/{ds}/{i}.jpg", "md5_pixels": f"h{i}", "group_phash4": group, "recommended_split": rec})
    return pd.DataFrame(rows)


def test_splits_disjoint_and_leakfree_sets_exclude_seen_groups():
    meta = _meta()
    for protocol in ("paper", "clean"):
        s = make_splits(meta, protocol, seed=0)
        assert set(s["train"]["path"]).isdisjoint(s["test"]["path"])
        seen = set(s["train"]["group_phash4"]) | set(s["val"]["group_phash4"])
        assert set(s["test_leakfree"]["group_phash4"]).isdisjoint(seen)
        assert set(s["d2_unseen"]["group_phash4"]).isdisjoint(seen)
        assert len(s["d2_all"]) == 20 and len(s["d2_unseen"]) <= 20


def test_fit_runs_and_restores_best_checkpoint():
    torch.manual_seed(0)
    x = torch.randint(0, 255, (16, 32, 32), dtype=torch.uint8)
    y = torch.arange(16) % 4
    m = XCNN([4, 8])
    info = fit(m, x, y, x, y, epochs=2, batch_size=8, log=lambda *_: None)
    assert 1 <= info["best_epoch"] <= 2 and len(info["history"]) == 2
    assert predict(m, x).shape == (16,)
