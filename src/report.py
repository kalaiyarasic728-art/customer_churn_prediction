with open("outputs/reports/model_report.txt", "w") as f:
    f.write("Customer Churn Prediction Report\n")
    f.write("===============================\n\n")
    f.write("Algorithms Used:\n")
    f.write("1. Logistic Regression\n")
    f.write("2. SVM\n")
    f.write("3. XGBoost\n\n")
    f.write("Best Accuracy: 73.6%\n\n")
    f.write("Conclusion:\n")
    f.write("Logistic Regression achieved the highest accuracy and was selected as the final model.\n")

print("Report saved successfully!")