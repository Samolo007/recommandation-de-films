def popularity_ranking(train):
    """Films classés du plus au moins aimé (nombre de notes >= 4 dans train)."""
    liked = train[train["rating"] >= 4]
    counts = liked.groupby("item_id").size().reset_index(name="n_liked")
    counts = counts.sort_values(["n_liked", "item_id"], ascending=[False, True])
    return counts["item_id"].tolist()


def recommend_popular(start_item, ranking, k=5):
    """Top k populaire, sans le film de départ."""
    return [item for item in ranking if item != start_item][:k]