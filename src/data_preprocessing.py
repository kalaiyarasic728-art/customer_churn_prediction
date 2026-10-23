import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("data/Bank_Churn_Classification_Dataset.csv")

print("Original Dataset Shape:", df.shape)

# Drop unnecessary columns
df = df.drop(["Unnamed: 0", "CustomerID"], axis=1)

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Encode Categorical Columns
le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["Contract"] = le.fit_transform(df["Contract"])
df["PaymentMethod"] = le.fit_transform(df["PaymentMethod"])

# Save Preprocessed Dataset
df.to_csv("data/preprocessed_churn_data.csv", index=False)

print("\nData Preprocessing Completed Successfully!")
print("Preprocessed Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())