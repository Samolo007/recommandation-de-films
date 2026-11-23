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