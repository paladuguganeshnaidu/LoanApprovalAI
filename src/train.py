"""Training entry point."""
import joblib
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:/Users/ganes/OneDrive/Desktop/LoanApprovalAI/generated_data/Loan_approval_data_2025.csv")
"""print(df.shape)
print(df.info())           -- Checking Dataset
print(df.isnull().sum())
"""

X = df.drop(
    columns=[
        "customer_id",
        "loan_status"
    ]
)
y = df["loan_status"]
categorical_features = X.select_dtypes(include=["object","string"]).columns
encoders = {}

for column in categorical_features:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column])
    encoders[column] = encoder

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
model=RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)
os.makedirs("saved_models", exist_ok=True)
joblib.dump(model, "saved_models/random_forest.pkl")
joblib.dump(encoders, "saved_models/encoders.pkl")

import matplotlib.pyplot as plt

importance = importance.sort_values(by="Importance", ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(importance["Feature"], importance["Importance"])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()