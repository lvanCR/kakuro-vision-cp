"""Paso 10 del pipeline: reconocimiento de dígitos con una CNN pequeña (PyTorch).

Arquitectura (similar a la del paper de referencia): tres bloques
conv 3x3 + BatchNorm + ReLU + max-pool (32/64/128 filtros), una capa densa de
128 con dropout y la salida softmax sobre los dígitos 0-9.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "digit_cnn.pt"


class DigitCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))    # 28 -> 14 -> 7 -> 3
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 3 * 3, 128), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class DigitReader:
    """Carga los pesos y devuelve probabilidades por dígito."""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DigitCNN().to(self.device)
        state = torch.load(weights, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def probabilities(self, canvases: list[np.ndarray]) -> np.ndarray:
        """Matriz (n, 10) de probabilidades para lienzos 28x28 con valores en [0, 1]."""
        if not canvases:
            return np.zeros((0, 10), np.float32)
        x = torch.from_numpy(np.stack(canvases).astype(np.float32)).unsqueeze(1).to(self.device)
        return torch.softmax(self.model(x), dim=1).cpu().numpy()
