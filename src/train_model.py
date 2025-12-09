import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc
import joblib
import os

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Split features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model with class imbalance handling
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate model
preds = model.predict(X_test)
probas = model.predict_proba(X_test)[:, 1]

prec, recall, _ = precision_recall_curve(y_test, probas)
pr_auc = auc(recall, prec)

print("\nMODEL EVALUATION")
print(classification_report(y_test, preds))
print("PR-AUC =", pr_auc)

# Save model
os.makedirs("src/artifacts", exist_ok=True)
joblib.dump(model, "src/artifacts/fraud_model.pkl")

print("\nModel saved to src/artifacts/fraud_model.pkl")
