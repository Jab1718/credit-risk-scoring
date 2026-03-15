# Chạy thẳng, không cần Flask
# python test_model.py
import pandas as pd
import numpy as np
import pickle

MODEL_PATH = r"models\lightGBM_model.pkl"
CSV_PATH = r"C:\Users\ADMIN\Downloads\ML_Credit_Scoring (1)\ML_Credit_Scoring\data\processed_data\test.csv"
FEATURES = [
    "EXT_SOURCE_MEAN","EXT_SOURCE_3","EXT_SOURCE_2","EXT_SOURCE_1",
    "YEARS_EMPLOYED","ANNUITY LENGTH","CREDIT_TO_GOODS_RATIO",
    "EMPLOYMENT_STABILITY","AGE","LOG_AMT_ANNUITY","LOG_AMT_GOODS_PRICE",
    "CONVERTED_DAYS_ID_PUBLISH","CONVERTED_DAYS_LAST_PHONE_CHANGE",
    "LOG_AMT_CREDIT","ANNUITY_BY_INCOME","CONVERTED_DAYS_REGISTRATION",
    "CREDIT_BY_INCOME","GOODS_PRICE_BY_INCOME",
    "REGION_POPULATION_RELATIVE","INCOME_PER_PERSON",
]

print("1. Load model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print(f"   OK: {type(model).__name__}, n_features={model.named_steps['preprocessor'].n_features_in_}")

print("2. Load CSV...")
df = pd.read_csv(CSV_PATH)
print(f"   OK: {df.shape}")

print("3. Check features...")
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    print(f"   MISSING: {missing}")
    exit(1)
print("   OK: all features present")

print("4. Predict...")
X = df[FEATURES].copy()
probs = model.predict_proba(X)[:, 1]
print(f"   OK: {len(probs)} predictions")

print("\n=== KẾT QUẢ ===")
for i, p in enumerate(probs):
    dec  = "Reject" if p >= 0.5 else "Approve"
    tier = "Low" if p < 0.3 else ("Medium" if p < 0.6 else "High")
    print(f"Row {i+1}: {dec:7s} | prob={p:.1%} | risk={tier}")

print(f"\nApprove: {sum(1 for p in probs if p < 0.5)} | Reject: {sum(1 for p in probs if p >= 0.5)}")