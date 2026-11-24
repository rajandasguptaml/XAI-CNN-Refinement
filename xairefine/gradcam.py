"""Grad-CAM (Eqs. 2-3) for every convolutional block, and layer-wise relevance m_l (Eq. 4).

For block l with activations A^k (output of its ReLU) and class score S_c:
    alpha_k = mean_{i,j} dS_c / dA^k(i, j)
    L_l     = ReLU( sum_k alpha_k A^k )
    m_l     = mean_{i,j} L_l(i, j), averaged over the images of a reference set.
The class c is the ground-truth class when labels are given, otherwise the predicted class.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import XCNN, to_input


class _Recorder:
    def __init__(self, model: XCNN):
        self.acts: list[torch.Tensor] = []
        self.handles = [b.act.register_forward_hook(self._hook) for b in model.blocks]

    def _hook(self, _m, _inp, out):
        out.retain_grad()
        self.acts.append(out)

    def close(self):
        for h in self.handles:
            h.remove()


def gradcam_maps(model: XCNN, x: torch.Tensor, target: torch.Tensor | None = None) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Return one (B, h_l, w_l) map per block and the logits. ``x`` is a float input batch."""
    model.eval()
    rec = _Recorder(model)
    try:
        x = x.detach().requires_grad_(False)
        logits = model(x)
        cls = target if target is not None else logits.argmax(1)
        score = logits.gather(1, cls.view(-1, 1)).sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        maps = []
        for a in rec.acts:
            alpha = a.grad.mean(dim=(2, 3), keepdim=True)
            maps.append(F.relu((alpha * a).sum(1)).detach())
        return maps, logits.detach()
    finally:
        rec.close()


def layer_relevance(model: XCNN, images_uint8: torch.Tensor, labels: torch.Tensor | None, batch_size: int = 40,
                    device: str = "cpu") -> np.ndarray:
    """m_l for every block, averaged over all images (Eq. 4)."""
    totals = np.zeros(len(model.blocks))
    n = 0
    for i in range(0, len(images_uint8), batch_size):
        xb = to_input(images_uint8[i : i + batch_size].to(device))
        yb = None if labels is None else labels[i : i + batch_size].to(device)
        maps, _ = gradcam_maps(model, xb, yb)
        totals += np.array([m.mean(dim=(1, 2)).sum().item() for m in maps])
        n += len(xb)
    return totals / max(n, 1)


def heatmap(model: XCNN, image_uint8: torch.Tensor, block: int = -1, target: int | None = None,
            device: str = "cpu") -> np.ndarray:
    """Grad-CAM of one image at one block, upsampled to the input size and scaled to [0, 1]."""
    x = to_input(image_uint8[None].to(device))
    t = None if target is None else torch.tensor([target], device=device)
    maps, _ = gradcam_maps(model, x, t)
    m = maps[block][:1, None]  # (1, 1, h, w)
    m = F.interpolate(m, size=image_uint8.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    m = m - m.min()
    return (m / m.max().clamp_min(1e-8)).cpu().numpy()
