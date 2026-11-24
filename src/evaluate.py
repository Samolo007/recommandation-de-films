from baseline import popularity_ranking, recommend_popular
from split import load_ratings, make_split


from baseline import popularity_ranking, recommend_popular
from knn import build_neighbors, recommend_knn
from split import load_ratings, make_split


def hit_at_k(test, recommend, k=5):
    """Part des profils dont le film masqué est dans les k recommandations."""
    hits = [
        row.hidden_item in recommend(row.start_item, k)
        for row in test.itertuples()
    ]
    return sum(hits) / len(hits)


if __name__ == "__main__":
    ratings = load_ratings()
    train, test = make_split(ratings)

    ranking = popularity_ranking(train)
    hit_base = hit_at_k(test, lambda start, k: recommend_popular(start, ranking, k))

    neighbors = build_neighbors(train)
    hit_knn = hit_at_k(test, lambda start, k: recommend_knn(start, neighbors, k))

    print("Films dans l'index k-NN :", len(neighbors))
    print("Profils dont le film de départ n'a pas de voisins :",
          (~test["start_item"].isin(neighbors.keys())).sum())
    print("Hit@5 baseline populaire :", round(hit_base, 4))
    print("Hit@5 k-NN cosinus       :", round(hit_knn, 4))