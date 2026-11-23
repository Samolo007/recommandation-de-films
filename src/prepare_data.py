import pandas as pd

ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"],
)

print(ratings.shape)
print(ratings.dtypes)
print(ratings.head())