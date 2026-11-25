from baseline import popularity_ranking, recommend_popular
from knn import build_neighbors, build_neighbors_svd, recommend_knn
from split import load_ratings, make_split


def hit_at_k(test, recommend, k=5):
    """Part des profils dont le film masqué est parmi les k recommandations."""
    hits = [
        row.hidden_item in recommend(row.start_item, k)
        for row in test.itertuples()
    ]
    return sum(hits) / len(hits)


if __name__ == "__main__":
    ratings = load_ratings()
    train, test = make_split(ratings)
    n = len(test)

    ranking = popularity_ranking(train)
    hit_pop = hit_at_k(test, lambda start, k: recommend_popular(start, ranking, k))

    neighbors = build_neighbors(train)
    hit_knn = hit_at_k(test, lambda start, k: recommend_knn(start, neighbors, k))

    neighbors_a = build_neighbors_svd(train)
    hit_a = hit_at_k(test, lambda start, k: recommend_knn(start, neighbors_a, k))

    print("Profils de test :", n)
    print("Films dans l'index : k-NN", len(neighbors), "| modèle A", len(neighbors_a))
    print("Profils dont le film de départ n'a pas de voisins : k-NN",
          int((~test["start_item"].isin(neighbors.keys())).sum()),
          "| modèle A",
          int((~test["start_item"].isin(neighbors_a.keys())).sum()))
    print()
    print(f"{'Modèle':<38}{'Trouvés':>9}{'Hit@5':>9}")
    for name, hit in (
        ("Top 5 populaire", hit_pop),
        ("Modèle de base (k-NN cosinus)", hit_knn),
        ("Modèle A (SVD 20 curseurs)", hit_a),
    ):
        print(f"{name:<38}{round(hit * n):>5} / {n}{hit:>9.4f}")