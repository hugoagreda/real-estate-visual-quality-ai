import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# =====================
# CONFIG
# =====================

EMB_PATH = Path("../data/embeddings/realestate_embeddings.parquet")
MODEL_PATH = Path("../models/pairwise_ranker.joblib")
OUTPUT_PATH = Path("../data/embeddings/realestate_fast_ranking.parquet")

pd.set_option('display.max_colwidth', None)

# =====================
# LOAD
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
# FAST RANKING
# =====================

print("\n⚡ Calculando fast ranking score...")

scores = model.decision_function(X)

df["fast_rank_score"] = scores

# =====================
# SORT
# =====================

df_sorted = df.sort_values(by="fast_rank_score", ascending=False)

print("\n🔥 Top 10 FAST ranking:\n")

for _, row in df_sorted.head(10).iterrows():
    print(f"{row['fast_rank_score']:.4f}  |  {row['image_path']}")

# =====================
# SAVE
# =====================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_sorted.to_parquet(OUTPUT_PATH, index=False)

print("\n✅ Fast ranking guardado en:")
print(OUTPUT_PATH)
