import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# =====================
# PATHS ROBUSTOS
# =====================

BASE_DIR = Path(__file__).resolve().parent.parent

EMB_PATH = BASE_DIR / "data/embeddings/realestate_embeddings.parquet"
MODEL_PATH = BASE_DIR / "models/quality_head.joblib"

# =====================
# LOAD DATA
# =====================

print("\n📂 Cargando embeddings...")

df = pd.read_parquet(EMB_PATH)
df = df[df["final_quality"].notna()].copy()

X = np.vstack(df["embedding"].values)
y = df["final_quality"].values

print(f"Shape embeddings: {X.shape}")

# =====================
# NORMALIZE EMBEDDINGS (SAFE)
# =====================

norms = np.linalg.norm(X, axis=1, keepdims=True)
norms[norms == 0] = 1e-8
X = X / norms

# =====================
# LABEL ENCODER
# =====================

le = LabelEncoder()
y_enc = le.fit_transform(y)

# =====================
# TRAIN TEST SPLIT
# =====================

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc
)

# =====================
# MODEL
# =====================

print("\n🧠 Entrenando quality head...")

model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    multi_class="auto",
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =====================
# EVALUATION
# =====================

print("\n📊 Evaluación:")

y_pred = model.predict(X_val)

print(classification_report(
    y_val,
    y_pred,
    target_names=le.classes_
))

# =====================
# SAVE MODEL
# =====================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "model": model,
    "label_encoder": le
}, MODEL_PATH)

print("\n✅ Modelo guardado en:")
print(MODEL_PATH)
