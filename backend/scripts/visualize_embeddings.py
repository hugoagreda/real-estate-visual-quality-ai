import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import umap

# =====================
# CONFIG
# =====================

EMB_PATH = Path("../data/embeddings/realestate_embeddings.parquet")
OUTPUT_PLOT = Path("../data/embeddings/umap_projection.png")

# =====================
# LOAD
# =====================

print("\n📂 Cargando embeddings...")

df = pd.read_parquet(EMB_PATH)

print(f"Embeddings cargados: {len(df)}")

X = np.vstack(df["embedding"].values)
y = df["final_quality"].values

print(f"Shape embeddings: {X.shape}")

# =====================
# NORMALIZE (MISMO ESPACIO QUE RUNTIME)
# =====================

X = X / np.linalg.norm(X, axis=1, keepdims=True)

# =====================
# UMAP
# =====================

print("\n🧠 Calculando proyección UMAP...")

n_neighbors = min(15, max(5, len(X)//5))

reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

X_2d = reducer.fit_transform(X)

# =====================
# PLOT
# =====================

print("\n🎨 Dibujando scatter...")

colors = {
    "good": "green",
    "medium": "orange",
    "bad": "red"
}

plt.figure(figsize=(10, 8))

for label in np.unique(y):

    idx = y == label

    plt.scatter(
        X_2d[idx, 0],
        X_2d[idx, 1],
        label=label,
        color=colors.get(label, "gray"),
        alpha=0.7,
        s=30
    )

plt.title("Real Estate Visual Quality — Embedding Space")
plt.legend()
plt.grid(True)

# guardar imagen
OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PLOT, dpi=150)

print(f"\n✅ UMAP guardado en: {OUTPUT_PLOT}")

plt.show()
