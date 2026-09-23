import pandas as pd

df = pd.read_csv("data/amazon_reviews.csv")

print("Dataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())