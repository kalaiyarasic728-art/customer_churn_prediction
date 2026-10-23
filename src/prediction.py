import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("data/Bank_Churn_Classification_Dataset.csv")

# Drop unnecessary columns
df = df.drop(["Unnamed: 0", "CustomerID"], axis=1)

# Encode categorical columns
le_gender = LabelEncoder()
le_contract = LabelEncoder()
le_payment = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Contract"] = le_contract.fit_transform(df["Contract"])
df["PaymentMethod"] = le_payment.fit_transform(df["PaymentMethod"])

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Select a sample customer
sample_customer = X.iloc[[0]]

# Predict
prediction = model.predict(sample_customer)

print("Prediction Result:")
if prediction[0] == 1:
    print("Customer is likely to Churn.")
else:
    print("Customer is likely to Stay.")
    import pandas as pd

# Example predictions
results = pd.DataFrame({
    "Prediction": prediction
})

results.to_csv("outputs/predictions.csv", index=False)

print("Predictions saved successfully!")