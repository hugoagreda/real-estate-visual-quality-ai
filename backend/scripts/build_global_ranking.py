import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# =====================
# CONFIG
# =====================

EMB_PATH = Path("../data/embeddings/realestate_embeddings.parquet")
MODEL_PATH = Path("../models/pairwise_ranker.joblib")
OUTPUT_PATH = Path("../data/embeddings/realestate_global_ranking.parquet")

pd.set_option('display.max_colwidth', None)

# =====================
# LOAD DATA
# =====================

print("\n📂 Cargando embeddings...")

df = pd.read_parquet(EMB_PATH)

loaded = joblib.load(MODEL_PATH)
model = loaded["model"] if isinstance(loaded, dict) else loaded

X = np.vstack(df["embedding"].values)

print(f"Embeddings: {X.shape}")

# =====================
# NORMALIZE (MISMO QUE RUNTIME)
# =====================

X = X / np.linalg.norm(X, axis=1, keepdims=True)

# =====================
# GLOBAL RANKING SCORE
# =====================

print("\n🧠 Calculando ranking global...")

scores = []

N = len(X)

for i in range(N):

    emb_i = X[i]

    # diferencias vectorizadas contra todos
    diffs = emb_i - X

    preds = model.predict(diffs)

    # ignorar self-compare
    wins = preds.sum() - preds[i]

    scores.append(int(wins))

df["global_rank_score"] = scores

# =====================
# SORT
# =====================

df_sorted = df.sort_values(by="global_rank_score", ascending=False)

print("\n🔥 Top 10 ranking global (paths completos):\n")

for _, row in df_sorted.head(10).iterrows():
    print(f"{row['global_rank_score']}  |  {row['image_path']}")

# =====================
# SAVE
# =====================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_sorted.to_parquet(OUTPUT_PATH, index=False)

print("\n✅ Ranking global guardado en:")
print(OUTPUT_PATH)
