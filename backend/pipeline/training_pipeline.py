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
AUTO_CONF_THRESHOLD = 0.08   # 🔥 ajustado

RANK_MAP = {
    "bad": 0,
    "medium": 1,
    "good": 2
}

random.seed(42)
np.random.seed(42)

# =====================================
# TRAINING HISTORY LOGGER
# =====================================
def save_training_history(df, mean_accuracy):

    history_path = BASE_DIR / "data/debug/round_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    if history_path.exists():
        old = pd.read_csv(history_path)
        round_id = len(old) + 1
    else:
        old = pd.DataFrame()
        round_id = 1

    new_row = pd.DataFrame([{
        "round": round_id,
        "accuracy": float(mean_accuracy),
        "total_samples": len(df),
        "human_labels": int((df["label_source"] == "human").sum()),
        "auto_labels": int((df["label_source"] == "auto").sum())
    }])

    df_final = pd.concat([old, new_row], ignore_index=True)
    df_final.to_csv(history_path, index=False)

    print(f"\n📈 Training history actualizado → {history_path}")

# =====================================
# LOAD EMBEDDINGS
# =====================================
def load_embeddings():

    print("\n📂 Cargando embeddings acumulativos...")

    emb_dir = BASE_DIR / "data/embeddings"
    files = list(emb_dir.glob("*_embeddings.parquet"))

    if not files:
        raise ValueError("❌ No hay embeddings disponibles.")

    dfs = []

    for f in files:
        print(f"  + {f.name}")
        dfs.append(pd.read_parquet(f))

    df = pd.concat(dfs, ignore_index=True)

    # 🔥 embeddings a numpy
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(x, dtype=np.float32)
    )

    # =====================================
    # MERGE DATASET CSV (SOURCE OF TRUTH)
    # =====================================

    csv_path = BASE_DIR / "data/datasets/interior_final_candidates.csv"

    if csv_path.exists():

        df_csv = pd.read_csv(csv_path, dtype=str)

        for col in ["final_quality","auto_quality","auto_confidence","label_source"]:
            if col not in df_csv.columns:
                df_csv[col] = np.nan if col=="auto_confidence" else ""

        # 🔥 normalizar paths (CLAVE)
        def normalize_path(p):
            if pd.isna(p):
                return ""
            return Path(p).name.lower()

        df["image_key"] = df["image_path"].apply(normalize_path)
        df_csv["image_key"] = df_csv["image_path"].apply(normalize_path)

        # 🔥 renombrar columnas CSV para evitar conflictos
        df_csv = df_csv.rename(columns={
            "final_quality":"final_quality_csv",
            "auto_quality":"auto_quality_csv",
            "auto_confidence":"auto_confidence_csv",
            "label_source":"label_source_csv"
        })

        df = df.merge(
            df_csv[[
                "image_key",
                "final_quality_csv",
                "auto_quality_csv",
                "auto_confidence_csv",
                "label_source_csv"
            ]],
            on="image_key",
            how="left"
        )

        # 🔥 consolidar columnas
        df["final_quality"] = df["final_quality_csv"]
        df["auto_quality"] = df["auto_quality_csv"]
        df["auto_confidence"] = df["auto_confidence_csv"]
        df["label_source"] = df["label_source_csv"]

    # =====================================
    # GARANTIZAR COLUMNAS
    # =====================================
    for col in ["final_quality","auto_quality","auto_confidence","label_source"]:
        if col not in df.columns:
            df[col] = np.nan if col=="auto_confidence" else ""

    # =====================================
    # auto_confidence seguro
    # =====================================
    df["auto_confidence"] = pd.to_numeric(
        df["auto_confidence"],
        errors="coerce"
    )

    # =====================================
    # pseudo labels auto
    # =====================================
    mask = (
        df["final_quality"].isna()
        & (df["auto_confidence"] >= AUTO_CONF_THRESHOLD)
    )

    df.loc[mask,"final_quality"] = df.loc[mask,"auto_quality"]
    df.loc[mask,"label_source"] = "auto"

    # =====================================
    # SOLO DATOS ENTRENABLES
    # =====================================
    df = df[df["final_quality"].notna()]
    df = df[df["final_quality"]!=""]

    print(f"\nEmbeddings totales cargados: {len(df)}")

    print("\nDistribución etiquetas:")
    print(df["final_quality"].value_counts())

    if "label_source" in df.columns:
        print("\nFuente etiquetas:")
        print(df["label_source"].value_counts())

    return df

# =====================================
# UTILS
# =====================================
def build_sample_weights(df):

    weights = np.ones(len(df), dtype=np.float32)
    weights[df["label_source"]!="human"] = 0.5
    return weights

def normalize_embeddings(X):

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms==0] = 1e-8
    return X / norms

# =====================================
# TRAIN PAIRWISE RANKER
# =====================================
def train_pairwise_ranker(df):

    print("\n🧠 Entrenando pairwise ranker...")

    X = normalize_embeddings(np.vstack(df["embedding"].values))
    y = df["final_quality"].values

    pairs_X = []
    pairs_y = []

    for i in range(len(df)):
        for j in range(i+1, len(df)):

            if y[i]==y[j]:
                continue

            diff = X[i]-X[j]

            if np.any(np.isnan(diff)):
                continue

            ri = RANK_MAP[y[i]]
            rj = RANK_MAP[y[j]]

            pairs_X.append(diff)
            pairs_y.append(1 if ri>rj else 0)

    if not pairs_X:
        print("⚠️ No hay pares comparables.")
        return

    pairs_X = np.array(pairs_X)
    pairs_y = np.array(pairs_y)

    if len(pairs_X)>PAIR_SAMPLES:
        idx = np.random.choice(len(pairs_X),PAIR_SAMPLES,replace=False)
        pairs_X = pairs_X[idx]
        pairs_y = pairs_y[idx]

    model = LogisticRegression(
        max_iter=2000,
        solver="saga",
        fit_intercept=False,
        n_jobs=-1,
        random_state=42
    )

    model.fit(pairs_X,pairs_y)

    joblib.dump(model,RANKER_PATH)
    print(f"✅ Ranker guardado en {RANKER_PATH}")

# =====================================
# TRAIN QUALITY HEAD
# =====================================
def train_quality_head(df):

    print("\n🧠 Entrenando quality head...")

    X = normalize_embeddings(np.vstack(df["embedding"].values))
    y = df["final_quality"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    weights = build_sample_weights(df)

    skf = StratifiedKFold(n_splits=min(5,np.bincount(y_enc).min()),shuffle=True,random_state=42)

    acc_scores = []

    for fold,(tr,va) in enumerate(skf.split(X,y_enc),1):

        print(f"\n🔁 Fold {fold}")

        model = LogisticRegression(
            max_iter=2000,
            solver="saga",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

        model.fit(X[tr],y_enc[tr],sample_weight=weights[tr])

        pred = model.predict(X[va])

        acc = accuracy_score(y_enc[va],pred)
        acc_scores.append(acc)

        print(classification_report(y_enc[va],pred,target_names=le.classes_))

    mean_acc = float(np.mean(acc_scores))
    print("\n📊 Accuracy media K-Fold:",mean_acc)

    save_training_history(df,mean_acc)

    final_model = LogisticRegression(
        max_iter=2000,
        solver="saga",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    final_model.fit(X,y_enc,sample_weight=weights)

    joblib.dump({
        "model":final_model,
        "label_encoder":le
    },QUALITY_HEAD_PATH)

    print(f"✅ Quality head guardado en {QUALITY_HEAD_PATH}")

# =====================================
# PIPELINE
# =====================================

def training_pipeline():

    print("\n========== TRAINING PIPELINE ==========")

    df = load_embeddings()

    train_pairwise_ranker(df)
    train_quality_head(df)

    print("\n🏁 TRAINING COMPLETADO")

if __name__ == "__main__":
    training_pipeline()
