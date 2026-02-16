import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import random

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score

# =====================================
# CONFIG
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

RANKER_PATH = BASE_DIR / "models/pairwise_ranker.joblib"
QUALITY_HEAD_PATH = BASE_DIR / "models/quality_head.joblib"

PAIR_SAMPLES = 5000

AUTO_CONF_THRESHOLD = 0.90

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

    print("\n📂 Cargando embeddings acumulativos...")

    emb_dir = BASE_DIR / "data/embeddings"
    files = list(emb_dir.glob("*_embeddings.parquet"))

    dfs = []

    for f in files:
        print(f"  + {f.name}")
        dfs.append(pd.read_parquet(f))

    if not dfs:
        raise ValueError("❌ No hay embeddings disponibles.")

    df = pd.concat(dfs, ignore_index=True)

    # convertir embeddings a numpy
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(x, dtype=np.float32)
    )

    # pseudo-labels auto si existen
    if "auto_quality" in df.columns and "auto_confidence" in df.columns:

        mask = (
            df["final_quality"].isna()
            & (df["auto_confidence"] >= AUTO_CONF_THRESHOLD)
        )

        df.loc[mask, "final_quality"] = df.loc[mask, "auto_quality"]
        df.loc[mask, "label_source"] = "auto"

    # dedupe seguro
    df = df.drop_duplicates(
        subset=["image_path", "label_source"],
        keep="last"
    )

    # solo entrenables
    df = df[df["final_quality"].notna()].copy()
    df = df.sort_values("image_path").reset_index(drop=True)

    print(f"\nEmbeddings totales cargados: {len(df)}")

    print("\nDistribución etiquetas:")
    print(df["final_quality"].value_counts())

    if "label_source" in df.columns:
        print("\nFuente etiquetas:")
        print(df["label_source"].value_counts())

    dataset_hash = hash(tuple(df["image_path"]))
    print(f"\n🔎 DATASET HASH: {dataset_hash}")

    return df

# =====================================
# BUILD SAMPLE WEIGHTS
# =====================================
def build_sample_weights(df):

    if "label_source" not in df.columns:
        return np.ones(len(df), dtype=np.float32)

    weights = np.ones(len(df), dtype=np.float32)

    # humanos pesan más
    weights[df["label_source"] != "human"] = 0.5

    return weights

# =====================================
# NORMALIZE
# =====================================
def normalize_embeddings(X):

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return X / norms

# =====================================
# TRAIN PAIRWISE RANKER
# =====================================
def train_pairwise_ranker(df):

    print("\n🧠 Entrenando pairwise ranker...")

    X = np.vstack(df["embedding"].values)
    y = df["final_quality"].values

    # 🔥 protección real
    if len(np.unique(y)) < 2:
        print("⚠️ Solo hay una clase presente. Ranker no se entrena.")
        return

    X = normalize_embeddings(X)

    pairs_X = []
    pairs_y = []

    indices = list(range(len(df)))

    attempts = 0
    max_attempts = PAIR_SAMPLES * 10

    while len(pairs_X) < PAIR_SAMPLES and attempts < max_attempts:

        i, j = random.sample(indices, 2)

        if y[i] not in RANK_MAP or y[j] not in RANK_MAP:
            attempts += 1
            continue

        rank_i = RANK_MAP[y[i]]
        rank_j = RANK_MAP[y[j]]

        if rank_i == rank_j:
            attempts += 1
            continue

        diff = X[i] - X[j]

        if np.any(np.isnan(diff)):
            attempts += 1
            continue

        pairs_X.append(diff)
        pairs_y.append(1 if rank_i > rank_j else 0)

        attempts += 1

    if len(pairs_X) == 0:
        print("⚠️ No hay pares comparables.")
        return

    pairs_X = np.array(pairs_X)
    pairs_y = np.array(pairs_y)

    model = LogisticRegression(
        max_iter=2000,
        solver="saga",
        fit_intercept=False,
        n_jobs=-1,
        random_state=42
    )

    model.fit(pairs_X, pairs_y)

    RANKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, RANKER_PATH)

    print(f"✅ Ranker guardado en {RANKER_PATH}")

# =====================================
# TRAIN QUALITY HEAD
# =====================================
def train_quality_head(df):

    print("\n🧠 Entrenando quality head (Stratified K-Fold)...")

    X = np.vstack(df["embedding"].values)
    y = df["final_quality"].values

    X = normalize_embeddings(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if len(np.unique(y_enc)) < 2:
        print("⚠️ Solo hay una clase presente. Quality head no se entrena.")
        return

    sample_weights = build_sample_weights(df)

    # 🔥 folds dinámicos si dataset pequeño
    n_splits = min(5, np.bincount(y_enc).min())
    if n_splits < 2:
        n_splits = 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_scores = []
    fold = 1

    for train_idx, val_idx in skf.split(X, y_enc):

        print(f"\n🔁 Fold {fold}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_enc[train_idx], y_enc[val_idx]

        w_train = sample_weights[train_idx]

        model = LogisticRegression(
            max_iter=2000,
            solver="saga",
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

        model.fit(X_train, y_train, sample_weight=w_train)

        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        acc_scores.append(acc)

        present_labels = np.unique(np.concatenate([y_val, y_pred]))

        print(classification_report(
            y_val,
            y_pred,
            labels=present_labels,
            target_names=le.inverse_transform(present_labels)
        ))

        fold += 1

    print("\n📊 Accuracy media K-Fold:", np.mean(acc_scores))

    print("\n🏁 Entrenando modelo final con todo el dataset...")

    final_model = LogisticRegression(
        max_iter=2000,
        solver="saga",
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    final_model.fit(X, y_enc, sample_weight=sample_weights)

    QUALITY_HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": final_model,
        "label_encoder": le
    }, QUALITY_HEAD_PATH)

    print(f"✅ Quality head guardado en {QUALITY_HEAD_PATH}")

# =====================================
# TRAINING PIPELINE
# =====================================
def training_pipeline(train_ranker=True, train_quality=True):

    print("\n========== TRAINING PIPELINE ==========")

    df = load_embeddings()

    # 🔥 YA NO SE PARA NUNCA POR TAMAÑO
    if len(df) < 10:
        print("⚠️ Dataset pequeño, pero seguimos entrenando.")

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
