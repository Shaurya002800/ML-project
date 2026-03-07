import pandas as pd
import numpy as np
import joblib
import os
import sys

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

"""
Module to train and predict crop diseases.

Important change: training was previously performed at import time which caused
heavy computation (and blocked Streamlit). Training is now performed only when
you call `train_and_save_model()` explicitly. The `predict_disease` function
performs lazy loading of saved model artifacts.
"""

MODEL_PATH = os.path.join(project_root, 'models', 'disease_model.pkl')
SCALER_PATH = os.path.join(project_root, 'models', 'scaler.pkl')
ENCODER_PATH = os.path.join(project_root, 'models', 'encoder.pkl')
DISEASE_ENCODER_PATH = os.path.join(project_root, 'models', 'disease_encoder.pkl')

# module-level cached artifacts (loaded lazily)
_model = None
_scaler = None
_crop_enc = None
_disease_enc = None


def train_and_save_model():
    """Train the disease model from raw CSV and save artifacts.

    Call this once during development or from a maintenance script. This will
    read data/data/raw/disease_data.csv and produce the model and encoders.
    """
    csv_path = os.path.join(project_root, 'data', 'raw', 'disease_data.csv')
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.dropna()
    print(f"Dataset shape: {df.shape}")

    crop_enc = LabelEncoder()
    df['crop_encoded'] = crop_enc.fit_transform(df['crop_type'])
    print(f"Crops in model: {list(crop_enc.classes_)}")
    joblib.dump(crop_enc, ENCODER_PATH)

    disease_enc = LabelEncoder()
    y = disease_enc.fit_transform(df['disease_label'])
    print(f"Disease classes: {list(disease_enc.classes_)}")
    joblib.dump(disease_enc, DISEASE_ENCODER_PATH)

    feature_cols = ['VOC1', 'VOC2', 'VOC3', 'humidity', 'temperature', 'crop_encoded']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, SCALER_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6,
        learning_rate=0.1, subsample=0.8,
        eval_metric='mlogloss', random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred, target_names=disease_enc.classes_))

    joblib.dump(model, MODEL_PATH)
    print("✅ Model saved")


def _lazy_load_artifacts():
    """Load model artifacts into module-level variables if not already loaded."""
    global _model, _scaler, _crop_enc, _disease_enc
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model artifacts not found. Run `train_and_save_model()` first to create them."
            )
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _crop_enc = joblib.load(ENCODER_PATH)
        _disease_enc = joblib.load(DISEASE_ENCODER_PATH)


def predict_disease(voc1, voc2, voc3, humidity, temperature, crop_name):
    """Predict disease and return (disease_name, confidence_percent).

    This function lazily loads saved artifacts. It does not retrain the model.
    """
    _lazy_load_artifacts()
    crop_enc = _crop_enc
    scaler = _scaler
    model = _model
    disease_enc = _disease_enc

    known_crops = list(crop_enc.classes_)
    if crop_name not in known_crops:
        # handle unseen crop names gracefully by returning a clear message
        raise ValueError(
            f"Crop '{crop_name}' not in trained crops: {known_crops}. "
            f"Please select one of: {known_crops}"
        )

    crop_num = crop_enc.transform([crop_name])[0]
    features = scaler.transform([[voc1, voc2, voc3, humidity, temperature, crop_num]])
    pred_num = model.predict(features)[0]
    disease = disease_enc.inverse_transform([pred_num])[0]
    confidence = round(model.predict_proba(features).max() * 100, 1)
    return disease, confidence


if __name__ == "__main__":
    # Convenience: train and save when running this script directly.
    train_and_save_model()