import joblib
import pandas as pd
import os

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    return joblib.load(path)

def save_predictions(preds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    preds.to_csv(path, index=False)
    print(f"Predictions saved to {path}")