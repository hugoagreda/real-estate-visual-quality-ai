import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import joblib
import random

# =====================
# SEED (REPRODUCIBLE)
# =====================

random.seed(42)
np.random.seed(42)

# =====================
# PATHS ROBUSTOS
# =====================

BASE_DIR = Path(__file__).resolve().parent.parent

EMB_PATH = BASE_DIR / "data/embeddings/realestate_embeddings.parquet"
MODEL_PATH = BASE_DIR / "models/pairwise_ranker.joblib"

PAIR_SAMPLES = 5000

RANK_MAP = {
    "bad": 0,
    "medium": 1,
    "good": 2
}

# =====================
# LOAD DATA
# =====================

print("\n📂 Cargando embeddings...")

df = pd.read_parquet(EMB_PATH)
df = df[df["final_quality"].notna()].copy()

print(f"Embeddings cargados: {len(df)}")

X = np.vstack(df["embedding"].values)
y = df["final_quality"].values

# =====================
# NORMALIZACIÓN (ALINEADO CON RUNTIME)
# =====================

X = X / np.linalg.norm(X, axis=1, keepdims=True)

# =====================
# GENERAR PARES BALANCEADOS
# =====================

print("\n🧠 Generando pares pairwise...")

pairs_X = []
pairs_y = []

indices = list(range(len(df)))

attempts = 0
max_attempts = PAIR_SAMPLES * 10

while len(pairs_X) < PAIR_SAMPLES and attempts < max_attempts:

    i, j = random.sample(indices, 2)

    rank_i = RANK_MAP[y[i]]
    rank_j = RANK_MAP[y[j]]

    if rank_i == rank_j:
        attempts += 1
        continue

    emb_i = X[i]
    emb_j = X[j]

    # 🔥 Añadir ambos sentidos (más estable)
    pairs_X.append(emb_i - emb_j)
    pairs_y.append(1 if rank_i > rank_j else 0)

    pairs_X.append(emb_j - emb_i)
    pairs_y.append(1 if rank_j > rank_i else 0)

    attempts += 1

pairs_X = np.array(pairs_X)
pairs_y = np.array(pairs_y)

print(f"Pares generados válidos: {len(pairs_X)}")

if len(np.unique(pairs_y)) < 2:
    raise ValueError("❌ Solo se generó una clase en pairs_y.")

# =====================
# TRAIN RANKER (ALINEADO CON FAST RANKING)
# =====================

print("\n🚀 Entrenando pairwise ranker...")

model = LogisticRegression(
    max_iter=2000,
    fit_intercept=False,   # 🔥 CRÍTICO para coherencia con fast_ranking
    n_jobs=-1
)

model.fit(pairs_X, pairs_y)

# =====================
# SAVE MODEL
# =====================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\n✅ Pairwise ranker guardado en:")
print(MODEL_PATH)
