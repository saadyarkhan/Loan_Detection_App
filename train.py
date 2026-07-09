"""
Trains a loan-approval classifier and saves it to model.pkl.

Fixes vs. the original notebook:
- Categorical columns are label-encoded (models can't handle raw strings).
- train_test_split arguments are in the correct order (X_train, X_test, y_train, y_test).
- fillna() results are actually kept (original notebook called fillna() without saving it).

Usage:
    python train.py
Requires a CSV named train_data.csv in the same folder, with the same columns
as the original Kaggle "Loan Prediction" dataset (Loan_ID, Gender, Married,
Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome,
LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status).
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("train_data.csv")

# --- Fill missing values ---
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mean())
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mean())

# --- Drop columns not used as features ---
df = df.drop(["Loan_ID", "Dependents"], axis=1)

# --- Encode categorical columns ---
cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Test accuracy:", model.score(X_test, y_test))

with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "encoders": encoders, "columns": list(X.columns)}, f)

print("Saved model.pkl")
