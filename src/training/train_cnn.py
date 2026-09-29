"""Entrenamiento de la CNN de dígitos.

Uso: python -m src.training.train_cnn --train data/digits/train.npz --test data/digits/test.npz

- Partición de validación por puzzle (no por recorte), para no filtrar información.
- Aumento de datos en línea: rotación, escala, traslación, desenfoque, ruido y
  variaciones de grosor del trazo.
- Pérdida con pesos por clase (el 1 y el 2 abundan como primer dígito).
- Se guarda el modelo con mejor exactitud de validación; al final se evalúa en
  el conjunto de prueba (fuentes nunca vistas) y se guarda la matriz de confusión.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.vision.ocr_cnn import DEFAULT_WEIGHTS, DigitCNN


def load(path: str) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    d = np.load(path)
    x = torch.from_numpy(d["X"].astype(np.float32) / 255.0).unsqueeze(1)
    return x, torch.from_numpy(d["y"]), d["puzzle"]


def augment(x: torch.Tensor) -> torch.Tensor:
    """Aumento en GPU sobre un lote (n, 1, 28, 28)."""
    n = x.shape[0]
    dev = x.device
    angle = (torch.rand(n, device=dev) - 0.5) * 2 * np.deg2rad(8)
    scale = 1 + (torch.rand(n, device=dev) - 0.5) * 0.25
    shift = (torch.rand(n, 2, device=dev) - 0.5) * 0.2
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    # grosor del trazo: dilatación o erosión suaves con max-pool
    r = torch.rand(n, device=dev)
    thick = F.max_pool2d(x, 3, 1, 1)
    thin = -F.max_pool2d(-x, 3, 1, 1)
    x = torch.where((r < 0.15)[:, None, None, None], thick, x)
    x = torch.where((r > 0.9)[:, None, None, None], thin, x)
    # desenfoque y ruido
    blur = F.avg_pool2d(x, 3, 1, 1)
    x = torch.where((torch.rand(n, device=dev) < 0.3)[:, None, None, None], blur, x)
    x = x * (0.7 + 0.3 * torch.rand(n, 1, 1, 1, device=dev))
    x = x + torch.randn_like(x) * 0.05 * torch.rand(n, 1, 1, 1, device=dev)
    return x.clamp(0, 1)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: str, bs: int = 2048) -> tuple[float, np.ndarray]:
    model.eval()
    preds = []
    for k in range(0, len(x), bs):
        preds.append(model(x[k:k + bs].to(device)).argmax(1).cpu())
    pred = torch.cat(preds).numpy()
    truth = y.numpy()
    cm = np.zeros((10, 10), np.int64)
    np.add.at(cm, (truth, pred), 1)
    return float((pred == truth).mean()), cm


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena la CNN de dígitos.")
    ap.add_argument("--train", default="data/digits/train.npz")
    ap.add_argument("--test", default="data/digits/test.npz")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_WEIGHTS))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x, y, puzzle = load(args.train)
    ids = np.unique(puzzle)
    val_ids = rng.choice(ids, size=max(1, int(args.val_frac * len(ids))), replace=False)
    is_val = torch.from_numpy(np.isin(puzzle, val_ids))
    x_tr, y_tr, x_val, y_val = x[~is_val], y[~is_val], x[is_val], y[is_val]
    print(f"entrenamiento {len(y_tr)}  validación {len(y_val)}  dispositivo {device}")

    counts = torch.bincount(y_tr, minlength=10).float()
    weights = (counts.sum() / (10 * counts)).to(device)
    model = DigitCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(y_tr) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    x_tr_dev, y_tr_dev = x_tr.to(device), y_tr.to(device)

    best, history = -1.0, []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        perm = torch.randperm(len(y_tr), device=device)
        for k in range(0, len(perm), args.batch):
            idx = perm[k:k + args.batch]
            logits = model(augment(x_tr_dev[idx]))
            loss = F.cross_entropy(logits, y_tr_dev[idx], weight=weights, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        acc, _ = evaluate(model, x_val, y_val, device)
        history.append({"epoch": epoch, "loss": total / len(y_tr), "val_acc": acc})
        print(f"época {epoch:2d}  pérdida {total / len(y_tr):.4f}  val {acc:.4%}  ({time.time() - t0:.1f} s)")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), args.out)

    model.load_state_dict(torch.load(args.out, map_location=device, weights_only=True))
    report = {"val_acc": best, "history": history}
    if args.test and Path(args.test).exists():
        x_te, y_te, _ = load(args.test)
        acc, cm = evaluate(model, x_te, y_te, device)
        report.update({"test_acc": acc, "test_n": int(len(y_te)), "confusion": cm.tolist()})
        print(f"prueba (fuentes no vistas): {acc:.4%} sobre {len(y_te)} dígitos")
        off = [(int(i), int(j), int(cm[i, j])) for i in range(10) for j in range(10) if i != j and cm[i, j]]
        print("confusiones (real, predicho, n):", sorted(off, key=lambda t: -t[2])[:8])
    with open(Path(args.out).with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"modelo guardado en {args.out}")


if __name__ == "__main__":
    main()
