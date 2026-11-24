import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def build_neighbors(train, k=5):
    """Pour chaque film aimé au moins une fois dans train, ses k voisins les plus proches.

    Renvoie {item_id: [{"item_id": int, "score": float}, ...]} trié du plus
    proche au moins proche. score = similarité cosinus = 1 - distance.
    """
    liked = train[train["rating"] >= 4]
    item_ids = np.sort(liked["item_id"].unique())
    user_ids = np.sort(liked["user_id"].unique())

    rows = np.searchsorted(item_ids, liked["item_id"])
    cols = np.searchsorted(user_ids, liked["user_id"])
    matrix = csr_matrix(
        (np.ones(len(liked)), (rows, cols)),
        shape=(len(item_ids), len(user_ids)),
    )

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix)

    neighbors = {}
    for i, item in enumerate(item_ids):
        found = []
        for dist, j in zip(distances[i], indices[i]):
            other = int(item_ids[j])
            if other != item:
                score = min(1.0, max(0.0, float(1 - dist)))
                found.append({"item_id": other, "score": score})
        neighbors[int(item)] = found[:k]
    return neighbors


def recommend_knn(start_item, neighbors, k=5):
    """Les k voisins du film de départ (liste vide si le film n'a pas de voisins)."""
    return [n["item_id"] for n in neighbors.get(start_item, [])][:k]