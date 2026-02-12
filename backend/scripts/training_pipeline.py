import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import random

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# =====================================
# CONFIG
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

EMB_PATH = BASE_DIR / "data/embeddings/realestate_embeddings.parquet"

RANKER_PATH = BASE_DIR / "models/pairwise_ranker.joblib"
QUALITY_HEAD_PATH = BASE_DIR / "models/quality_head.joblib"

PAIR_SAMPLES = 5000

RANK_MAP = {
    "bad": 0,
    "medium": 1,
    "good": 2
}

random.seed(42)
np.random.seed(42)

# =====================================
# LOAD EMBEDDINGS
# =====================================
def load_embeddings():

    print("\n📂 Cargando embeddings...")
    df = pd.read_parquet(EMB_PATH)
    df = df[df["final_quality"].notna()].copy()

    print(f"Embeddings disponibles: {len(df)}")

    return df

# =====================================
# TRAIN PAIRWISE RANKER
# =====================================
def train_pairwise_ranker(df):

    print("\n🧠 Entrenando pairwise ranker...")

    X = np.vstack(df["embedding"].values)
    y = df["final_quality"].values

    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    pairs_X = []
    pairs_y = []

    indices = list(range(len(df)))

    attempts = 0
    max_attempts = PAIR_SAMPLES * 10

    while len(pairs_X) < PAIR_SAMPLES and attempts < max_attempts:

        i, j = random.sample(indices, 2)

        # ⚠️ Evita crash si una clase desaparece
        if y[i] not in RANK_MAP or y[j] not in RANK_MAP:
            attempts += 1
            continue

        rank_i = RANK_MAP[y[i]]
        rank_j = RANK_MAP[y[j]]

        if rank_i == rank_j:
            attempts += 1
            continue

        emb_i = X[i]
        emb_j = X[j]

        pairs_X.append(emb_i - emb_j)
        pairs_y.append(1 if rank_i > rank_j else 0)

        pairs_X.append(emb_j - emb_i)
        pairs_y.append(1 if rank_j > rank_i else 0)

        attempts += 1

    pairs_X = np.array(pairs_X)
    pairs_y = np.array(pairs_y)

    model = LogisticRegression(
        max_iter=2000,
        fit_intercept=False,
        n_jobs=-1
    )

    model.fit(pairs_X, pairs_y)

    RANKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, RANKER_PATH)

    print(f"✅ Ranker guardado en {RANKER_PATH}")

# =====================================
# TRAIN QUALITY HEAD
# =====================================
def train_quality_head(df):

    print("\n🧠 Entrenando quality head...")

    X = np.vstack(df["embedding"].values)
    y = df["final_quality"].values

    # 🔥 Normalización robusta (alineado runtime)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    X = X / norms

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ⚠️ Si solo queda 1 clase, no entrenamos
    if len(np.unique(y_enc)) < 2:
        print("⚠️ Solo hay una clase presente. Quality head no se entrena.")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc
    )

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    print("\n📊 Evaluación:")

    y_pred = model.predict(X_val)

    # 🔥 SOLUCIÓN DEFINITIVA AL ERROR
    present_labels = np.unique(np.concatenate([y_val, y_pred]))

    print(classification_report(
        y_val,
        y_pred,
        labels=present_labels,
        target_names=le.inverse_transform(present_labels)
    ))

    QUALITY_HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": model,
        "label_encoder": le
    }, QUALITY_HEAD_PATH)

    print(f"✅ Quality head guardado en {QUALITY_HEAD_PATH}")

# =====================================
# TRAINING PIPELINE (SIMPLE)
# =====================================
def training_pipeline(train_ranker=True, train_quality=True):

    print("\n========== TRAINING PIPELINE ==========")

    df = load_embeddings()

    if train_ranker:
        train_pairwise_ranker(df)

    if train_quality:
        train_quality_head(df)

    print("\n🏁 TRAINING COMPLETADO")

# =====================================
# ENTRYPOINT
# =====================================

if __name__ == "__main__":
    training_pipeline()
