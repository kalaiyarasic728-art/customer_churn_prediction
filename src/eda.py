import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/charts", exist_ok=True)

df = pd.read_csv("data/bank_churn_classification_dataset.csv")

df["Churn"].value_counts().plot(kind="bar")

plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")

plt.savefig("outputs/charts/churn_distribution.png")

plt.close()

print("Chart saved successfully!")