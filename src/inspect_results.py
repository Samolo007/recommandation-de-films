from knn import build_neighbors, recommend_knn
from split import load_ratings, make_split
from train_model import load_catalog


def main():
    ratings = load_ratings()
    train, test = make_split(ratings)
    catalog = load_catalog()
    neighbors = build_neighbors(train)

    test = test.copy()
    test["recs"] = [recommend_knn(s, neighbors, 5) for s in test["start_item"]]
    test["hit"] = [h in r for h, r in zip(test["hidden_item"], test["recs"])]
    print("Profils :", len(test), "| trouvés :", int(test["hit"].sum()))

    def show(row):
        print("Départ  :", catalog[row.start_item])
        print("Proposés:", [catalog[i] for i in row.recs])
        print("Masqué  :", catalog[row.hidden_item], "->", "TROUVÉ" if row.hit else "raté")
        print()

    print("=== 3 profils trouvés ===")
    for row in test[test["hit"]].head(3).itertuples():
        show(row)
    print("=== 3 profils ratés ===")
    for row in test[~test["hit"]].head(3).itertuples():
        show(row)

    # Popularité (notes >= 4 dans train) du film masqué
    n_liked = train[train["rating"] >= 4].groupby("item_id").size()
    top100 = set(n_liked.sort_values(ascending=False, kind="mergesort").index[:100])
    test["populaire"] = test["hidden_item"].isin(top100)
    print("=== Hit@5 selon la popularité du film masqué ===")
    for flag, name in [(True, "dans le top 100 des films aimés"), (False, "hors du top 100")]:
        sub = test[test["populaire"] == flag]
        print(f"{name} : {len(sub)} profils, {int(sub['hit'].sum())} trouvés "
              f"({sub['hit'].mean():.3f})")

    # Films sans voisins
    sans = sorted(set(catalog) - set(neighbors))
    print("=== Films sans voisins (exemples) ===")
    for i in sans[:5]:
        print(i, catalog[i])


if __name__ == "__main__":
    main()