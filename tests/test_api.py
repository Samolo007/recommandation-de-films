import joblib
import pytest
from fastapi.testclient import TestClient

import api

FAKE_CATALOG = {
    1: "Film A", 2: "Film B", 3: "Film C", 4: "Film D",
    5: "Film E", 6: "Film F", 7: "Film sans voisins",
}


def fake_neighbors(item_id):
    others = [i for i in range(1, 7) if i != item_id]
    return [
        {"item_id": i, "title": FAKE_CATALOG[i], "score": round(0.9 - 0.1 * rank, 2)}
        for rank, i in enumerate(others)
    ]


FAKE_INDEX = {
    "neighbors": {i: fake_neighbors(i) for i in range(1, 7)},
    "catalog": FAKE_CATALOG,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    path = tmp_path_factory.mktemp("index") / "knn_index.joblib"
    joblib.dump(FAKE_INDEX, path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(api, "MODEL", path)
        with TestClient(api.app) as test_client:
            yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid(client):
    response = client.post("/predict", json={"item_id": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["item_id"] == 1
    assert body["title"] == "Film A"
    assert len(body["recommendations"]) == 5
    assert 1 not in [r["item_id"] for r in body["recommendations"]]
    scores = [r["score"] for r in body["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    assert body["message"] is None


def test_predict_without_neighbors(client):
    response = client.post("/predict", json={"item_id": 7})
    assert response.status_code == 200
    assert response.json()["recommendations"] == []
    assert response.json()["message"] == api.NO_NEIGHBORS


def test_predict_unknown_id(client):
    response = client.post("/predict", json={"item_id": 99999})
    assert response.status_code == 404
    assert response.json() == {"detail": api.NOT_FOUND}


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"item_id": 0}, "item_id"),
        ({"item_id": -1}, "item_id"),
        ({"item_id": "50"}, "item_id"),
        ({"item_id": 50.5}, "item_id"),
        ({}, "item_id"),
        ({"item_id": 1, "x": 1}, "x"),
    ],
    ids=["zero", "negatif", "texte", "decimal", "absent", "champ-en-trop"],
)
def test_predict_rejects_invalid_input(client, payload, field):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]