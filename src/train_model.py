from pathlib import Path

import joblib
import pandas as pd

from knn import build_neighbors_svd
from split import load_ratings, make_split

ITEM_PATH = "data/ml-100k/u.item"
OUTPUT_PATH = Path("models/knn_index.joblib")


def load_catalog(path=ITEM_PATH):
    """{item_id: title} pour les 1 682 films."""
    items = pd.read_csv(
        path,
        sep="|",
        header=None,
        usecols=[0, 1],
        names=["item_id", "title"],
        encoding="latin-1",
    )
    return {int(i): str(t) for i, t in zip(items["item_id"], items["title"])}


def add_titles(neighbors, catalog):
    """Ajoute le titre à chaque voisin, dans l'ordre convenu : item_id, title, score."""
    return {
        item: [
            {"item_id": n["item_id"], "title": catalog[n["item_id"]], "score": n["score"]}
            for n in voisins
        ]
        for item, voisins in neighbors.items()
    }


def main():
    ratings = load_ratings()
    train, _ = make_split(ratings)  # le même train que pour l'évaluation

    catalog = load_catalog()
    neighbors = add_titles(build_neighbors_svd(train), catalog)
    artifact = {"neighbors": neighbors, "catalog": catalog}

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, OUTPUT_PATH)
    print("Fichier écrit :", OUTPUT_PATH, "|", OUTPUT_PATH.stat().st_size, "octets")

    # Vérification : on recharge le fichier et on contrôle son contenu
    loaded = joblib.load(OUTPUT_PATH)
    print("Rechargé identique :", loaded == artifact)
    print("Films dans le catalogue :", len(loaded["catalog"]))
    print("Films avec voisins :", len(loaded["neighbors"]))
    print("Films sans voisins :", len(loaded["catalog"]) - len(loaded["neighbors"]))

    voisins = loaded["neighbors"][1]
    print("Voisins du film 1 :", loaded["catalog"][1])
    for n in voisins:
        print("  ", n)
    ok_types = all(
        type(n["item_id"]) is int and type(n["title"]) is str and type(n["score"]) is float
        for v in loaded["neighbors"].values() for n in v
    )
    print("Types Python natifs :", ok_types)
    print("5 voisins partout :", all(len(v) == 5 for v in loaded["neighbors"].values()))


if __name__ == "__main__":
    main()