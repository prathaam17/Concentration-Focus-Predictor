import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def clean_str(val):
    if not isinstance(val, str):
        return val
    # Normalize various forms of dashes/mojibake artifacts to clean standard dash '-'
    return re.sub(r'[\x96\u2013\u2014â€“â€”]+', '-', val).strip()

def main():
    csv_path = "focus.csv"
    if not os.path.exists(csv_path):
        csv_path = r"c:\Users\prath\Downloads\ML Project\focus.csv"
        
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, encoding='cp1252')
    df.columns = df.columns.str.strip()

    # Clean string encoding artifacts (e.g. 21â€“26 -> 21-26)
    df = df.map(clean_str)

    # Drop non-feature columns if present
    df = df.drop(columns=['Timestamp', 'Email Address'], errors='ignore')

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    target_col = '14.Rate your overall concentration ability'
    feature_cols = [c for c in df.columns if c != target_col]

    # Fit LabelEncoders and preserve category choices for UI
    encoders = {}
    feature_options = {}

    df_encoded = df.copy()

    for col in feature_cols:
        le = LabelEncoder()
        unique_vals = list(df[col].astype(str).unique())
        try:
            unique_vals.sort()
        except Exception:
            pass
        feature_options[col] = unique_vals
        
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df_encoded[feature_cols]
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {acc:.4f}")

    # Save artifacts
    artifacts = {
        'model': model,
        'encoders': encoders,
        'feature_cols': feature_cols,
        'feature_options': feature_options,
        'target_col': target_col
    }

    model_file = r"c:\Users\prath\Downloads\ML Project\model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(artifacts, f)

    print(f"Artifacts successfully saved to {model_file}")

if __name__ == "__main__":
    main()
