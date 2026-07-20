import pandas as pd

# Load datasets
fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

# Display basic information
print("=" * 50)
print("FAKE NEWS DATASET")
print("=" * 50)
print(fake_df.head())

print("\nDataset Shape:", fake_df.shape)
print("\nColumns:")
print(fake_df.columns)

print("\nMissing Values:")
print(fake_df.isnull().sum())

print("\n" + "=" * 50)
print("REAL NEWS DATASET")
print("=" * 50)
print(true_df.head())

print("\nDataset Shape:", true_df.shape)
print("\nColumns:")
print(true_df.columns)

print("\nMissing Values:")
print(true_df.isnull().sum())