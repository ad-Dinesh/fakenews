import pandas as pd

# Load datasets
fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

# Display first 5 rows
print("Fake News Dataset")
print(fake_df.head())

print("\nReal News Dataset")
print(true_df.head())
