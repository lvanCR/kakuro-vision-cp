from pathlib import Path

import pytest

from src.vision.ocr_cnn import DEFAULT_WEIGHTS

pytestmark = pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="modelo no entrenado")
EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.app import app
    with TestClient(app) as c:          # ejecuta el lifespan (carga la CNN)
        yield c


def test_health_and_index(client):
    assert client.get("/api/health").json()["model_loaded"] is True
    r = client.get("/")
    assert r.status_code == 200 and "Kakuro" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_examples(client):
    names = client.get("/api/examples").json()
    assert len(names) >= 1
    assert client.get(f"/api/examples/{names[0]}").status_code == 200
    assert client.get("/api/examples/..%2Frequirements.txt").status_code == 404


def test_solve_example(client):
    path = EXAMPLES / "ejemplo_negro_foto.jpg"
    with open(path, "rb") as f:
        r = client.post("/api/solve", files={"file": (path.name, f, "image/jpeg")}, data={"model": "M2"})
    assert r.status_code == 200
    body = r.json()
    assert body["solved"] and body["unique"] is True
    assert (body["rows"], body["cols"]) == (6, 6)
    assert body["images"]["overlay"].startswith("data:image/jpeg;base64,")
    assert body["images"]["clean"].startswith("data:image/png;base64,")
    assert body["solution"]["solution"][1][1:3] == [7, 6]


def test_solve_rejects_bad_input(client):
    r = client.post("/api/solve", files={"file": ("x.jpg", b"no es una imagen", "image/jpeg")})
    assert r.status_code == 400
    with open(EXAMPLES / "ejemplo_negro_foto.jpg", "rb") as f:
        r = client.post("/api/solve", files={"file": ("x.jpg", f, "image/jpeg")}, data={"model": "M9"})
    assert r.status_code == 400
