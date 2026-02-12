import time
from pathlib import Path
import pandas as pd

from .runtime_score import score_image


# =====================================
# CONFIG
# =====================================

WATCH_FOLDER = Path("backend/data/images/kaggle_raw")
HISTORY_PATH = Path("backend/runtime/runtime_history.parquet")

SCAN_INTERVAL = 5   # segundos entre scans


# =====================================
# LOAD HISTORY
# =====================================

def load_history():

    if HISTORY_PATH.exists():
        df = pd.read_parquet(HISTORY_PATH)
        processed = set(df["image_path"].values)
        history = df.to_dict("records")
        print(f"♻️ Historial cargado: {len(history)}")
    else:
        processed = set()
        history = []

    return processed, history


# =====================================
# SAVE HISTORY
# =====================================

def save_history(history):

    df = pd.DataFrame(history)

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(HISTORY_PATH, index=False)


# =====================================
# MONITOR LOOP
# =====================================

def run_monitor():

    print("\n🧠 Runtime Monitor iniciado")
    print(f"📂 Carpeta observada: {WATCH_FOLDER}")

    processed, history = load_history()

    while True:

        new_files = list(WATCH_FOLDER.glob("*.webp"))

        for img_path in new_files:

            norm_path = img_path.resolve().as_posix()

            if norm_path in processed:
                continue

            print(f"\n📸 Nueva imagen detectada: {img_path.name}")

            start = time.time()

            try:
                result = score_image(norm_path)
            except Exception as e:
                print(f"⚠️ Error scoring: {e}")
                continue

            elapsed = round(time.time() - start, 3)

            result["latency"] = elapsed
            result["timestamp"] = time.time()

            history.append(result)
            processed.add(norm_path)

            print(f"🔥 Score: {result['score']:.3f} | ⏱ {elapsed}s")

        # guardar cada ciclo
        save_history(history)

        # =====================
        # STATS LIVE
        # =====================

        if history:
            df = pd.DataFrame(history)

            print("\n📊 Stats runtime:")

            print(
                "images:",
                len(df),
                "| avg_score:",
                round(df["score"].mean(), 3),
                "| avg_latency:",
                round(df["latency"].mean(), 3),
            )

        time.sleep(SCAN_INTERVAL)


# =====================================
# ENTRYPOINT
# =====================================

if __name__ == "__main__":
    run_monitor()
