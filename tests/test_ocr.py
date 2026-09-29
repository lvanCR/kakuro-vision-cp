import numpy as np
import pytest
import torch

from src.vision.ocr_cnn import DEFAULT_WEIGHTS, DigitCNN, DigitReader


def test_model_output_shape():
    out = DigitCNN()(torch.zeros(4, 1, 28, 28))
    assert out.shape == (4, 10)


@pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="modelo no entrenado")
def test_reader_on_test_digits():
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data" / "digits" / "test.npz"
    if not path.exists():
        pytest.skip("dataset de prueba no generado")
    d = np.load(path)
    reader = DigitReader(device="cpu")
    probs = reader.probabilities(list(d["X"][:500].astype(np.float32) / 255.0))
    assert probs.shape == (500, 10)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-4)
    assert (probs.argmax(axis=1) == d["y"][:500]).mean() > 0.98


@pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="modelo no entrenado")
def test_reader_empty():
    assert DigitReader(device="cpu").probabilities([]).shape == (0, 10)
