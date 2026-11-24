"""Training and prediction. Settings of Table 3: Adam, learning rate 1e-3, cross-entropy, batch size 40, 40 epochs.

The whole uint8 image cache is moved to the device once, so an epoch is a few seconds on a small GPU. The checkpoint
with the best validation accuracy is kept.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn

from .model import XCNN, to_input


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: XCNN, images: torch.Tensor, batch_size: int = 200, device: str = "cpu") -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        out.append(model(to_input(images[i : i + batch_size].to(device))).argmax(1).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=int)


def fit(model: XCNN, x_tr: torch.Tensor, y_tr: torch.Tensor, x_va: torch.Tensor, y_va: torch.Tensor, *,
        epochs: int = 40, batch_size: int = 40, lr: float = 1e-3, seed: int = 0, device: str = "cpu",
        log=print) -> dict:
    set_seed(seed)
    model.to(device)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator(device="cpu").manual_seed(seed)
    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_tr), generator=g).to(device)
        total, correct, loss_sum = 0, 0, 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = to_input(x_tr[idx]), y_tr[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(idx)
        val_acc = float((predict(model, x_va, device=device) == y_va.numpy()).mean())
        history.append({"epoch": ep, "loss": loss_sum / total, "train_acc": correct / total, "val_acc": val_acc})
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": ep, "state": copy.deepcopy(model.state_dict())}
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            log(f"  epoch {ep:>2}/{epochs} loss {loss_sum / total:.4f} train {correct / total:.4f} val {val_acc:.4f} "
                f"({time.time() - t0:.0f}s)")
    model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "best_val_acc": best["val_acc"], "seconds": time.time() - t0,
            "history": history}
