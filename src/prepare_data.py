import pandas as pd

ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"],
)

print(ratings.shape)
print(ratings.dtypes)
print(ratings.head())
items = pd.read_csv(
    "data/ml-100k/u.item",
    sep="|",
    header=None,
    usecols=[0, 1],
    names=["item_id", "title"],
    encoding="latin-1",
)

print(items.shape)
print(items.dtypes)
print(items.head())
print("--- valeurs manquantes (u.data) ---")
print(ratings.isna().sum())

print("doublons exacts :", ratings.duplicated().sum())
print("doublons (user_id, item_id) :", ratings.duplicated(subset=["user_id", "item_id"]).sum())

print("--- répartition des notes ---")
print(ratings["rating"].value_counts().sort_index())
print("notes hors 1-5 :", (~ratings["rating"].between(1, 5)).sum())

print("item_id absents du catalogue :", (~ratings["item_id"].isin(items["item_id"])).sum())
print("utilisateurs :", ratings["user_id"].nunique(), "| films notés :", ratings["item_id"].nunique())

print("--- u.item ---")
print(items.isna().sum())
print("titres en double :", items["title"].duplicated().sum())