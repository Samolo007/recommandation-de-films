import pandas as pd


def load_ratings(path="data/ml-100k/u.data"):
    return pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )


def make_split(ratings, min_liked=2):
    """Masque le dernier film aimé de chaque profil éligible.

    Renvoie (train, test) :
    - test : une ligne par profil éligible (user_id, hidden_item, start_item)
    - train : toutes les notes SAUF les interactions masquées
    """
    liked = ratings[ratings["rating"] >= 4].sort_values(
        ["user_id", "timestamp", "item_id"]
    )
    n_liked = liked.groupby("user_id")["item_id"].transform("size")
    eligible = liked[n_liked >= min_liked]

    # 0 = dernier film aimé, 1 = avant-dernier, etc.
    rank_from_end = eligible.groupby("user_id").cumcount(ascending=False)
    hidden = eligible[rank_from_end == 0]
    start = eligible[rank_from_end == 1]

    test = (
        hidden[["user_id", "item_id"]]
        .rename(columns={"item_id": "hidden_item"})
        .merge(
            start[["user_id", "item_id"]].rename(columns={"item_id": "start_item"}),
            on="user_id",
        )
    )
    train = ratings.drop(index=hidden.index)
    return train, test


if __name__ == "__main__":
    ratings = load_ratings()
    train, test = make_split(ratings)
    print("train :", train.shape, "| test :", test.shape)
    print(test.head())

    pairs = test.merge(
        train, left_on=["user_id", "hidden_item"], right_on=["user_id", "item_id"]
    )
    print("films masqués encore présents dans train :", len(pairs))
    print("départ == masqué :", (test["start_item"] == test["hidden_item"]).sum())
    liked_train = train[train["rating"] >= 4]
    print("films avec au moins un 'aimé' dans train :", liked_train["item_id"].nunique())
    print("reproductible :", make_split(ratings)[1].equals(test))