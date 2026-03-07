import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load data ──────────────────────────────────────
df = pd.read_csv('data/raw/disease_data.csv')
df.columns = df.columns.str.strip()
df = df.dropna()
print(f"Dataset shape: {df.shape}")

# ── 2. Encode crop type (text → number) ──────────────
crop_enc = LabelEncoder()
df['crop_encoded'] = crop_enc.fit_transform(df['crop_type'])
joblib.dump(crop_enc, 'models/encoder.pkl')

# ── 3. Encode disease label (THIS WAS MISSING) ────────
disease_enc = LabelEncoder()
y = disease_enc.fit_transform(df['disease_label'])  # Healthy=0, Powdery_Mildew=1, Rust=2
joblib.dump(disease_enc, 'models/disease_encoder.pkl')
print(f"Disease classes: {list(disease_enc.classes_)}")  # shows the mapping

# ── 4. Scale numeric features ─────────────────────────
feature_cols = ['VOC1', 'VOC2', 'VOC3', 'humidity', 'temperature', 'crop_encoded']
scaler = StandardScaler()
X = scaler.fit_transform(df[feature_cols])
joblib.dump(scaler, 'models/scaler.pkl')

# ── 5. Train / test split ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 6. Train XGBoost ──────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train, y_train)

# ── 7. Evaluate ───────────────────────────────────────
y_pred = model.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=disease_enc.classes_))

# ── 8. Save model ─────────────────────────────────────
joblib.dump(model, 'models/disease_model.pkl')
print("✅ Model saved to models/disease_model.pkl")


# ── 9. Prediction function ────────────────────────────
def predict_disease(voc1, voc2, voc3, humidity, temperature, crop_name):
    model        = joblib.load('models/disease_model.pkl')
    scaler       = joblib.load('models/scaler.pkl')
    crop_enc     = joblib.load('models/encoder.pkl')
    disease_enc  = joblib.load('models/disease_encoder.pkl')

    crop_num  = crop_enc.transform([crop_name])[0]
    features  = scaler.transform([[voc1, voc2, voc3, humidity, temperature, crop_num]])

    pred_num   = model.predict(features)[0]
    disease    = disease_enc.inverse_transform([pred_num])[0]  # number → text name
    confidence = round(model.predict_proba(features).max() * 100, 1)

    return disease, confidence


# ── Quick test ────────────────────────────────────────
disease, conf = predict_disease(0.45, 0.55, 0.40, 75, 25, 'Wheat')
print(f"\nTest prediction: {disease} ({conf}% confidence)")