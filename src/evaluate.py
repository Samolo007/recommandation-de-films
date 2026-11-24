from baseline import popularity_ranking, recommend_popular
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
    print("Top 5 populaire (item_id) :", ranking[:5])

    hit = hit_at_k(test, lambda start, k: recommend_popular(start, ranking, k))
    print("Hit@5 baseline populaire :", round(hit, 4))