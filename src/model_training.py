import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/bank_churn_classification_dataset.csv")
df = df.dropna()

# =========================
# FEATURES
# =========================
features = ["Gender", "SeniorCitizen", "Tenure", "MonthlyCharges"]
target = "Churn"

df = df[features + [target]]

# =========================
# ENCODE
# =========================
for col in df.columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df[features]
y = df[target]

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODELS
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

best_model = None
best_acc = 0

# =========================
# TRAIN
# =========================
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print("\n", name)
    print("Accuracy:", acc)

    if acc > best_acc:
        best_acc = acc
        best_model = model

# =========================
# SAVE MODEL
# =========================
pickle.dump(best_model, open("models/churn_model.pkl", "wb"))

print("\nMODEL SAVED SUCCESSFULLY")
print("BEST ACCURACY:", best_acc)