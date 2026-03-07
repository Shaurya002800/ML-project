import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── 1. Load data ──────────────────────────────────────
df = pd.read_csv('data/raw/disease_data.csv')
df.columns = df.columns.str.strip()   # remove accidental spaces
df = df.dropna()                        # remove rows with missing values
print(f"Dataset shape: {df.shape}")

# ── 2. Encode crop type (text → number) ──────────────
crop_enc = LabelEncoder()
df['crop_encoded'] = crop_enc.fit_transform(df['crop_type'])
joblib.dump(crop_enc, 'models/encoder.pkl')

# ── 3. Scale numeric features ─────────────────────────
feature_cols = ['VOC1', 'VOC2', 'VOC3', 'humidity', 'temperature', 'crop_encoded']
scaler = StandardScaler()
X = scaler.fit_transform(df[feature_cols])
y = df['disease_label']
joblib.dump(scaler, 'models/scaler.pkl')

# ── 4. Train / test split ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# ── Train the model ────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=200,        # number of trees
    max_depth=6,              # tree depth
    learning_rate=0.1,        # how fast it learns
    subsample=0.8,            # prevents overfitting
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))

# ── Save model ─────────────────────────────────────────
joblib.dump(model, 'models/disease_model.pkl')
print("Model saved to models/disease_model.pkl")