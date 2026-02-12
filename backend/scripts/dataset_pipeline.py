import sys
from pathlib import Path

# Add parent directory to path so imports work when running script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import cv2
import requests
import time
import torch

from runtime.runtime_models import encode_image

IMG = 50
YOLO_MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# UTILS
# =====================================
def run_step(name, func, *args, **kwargs):

    print(f"\n▶️ Iniciando {name}")
    start = time.time()

    try:
        result = func(*args, **kwargs)
    except Exception as e:
        print(f"💥 ERROR en {name}: {e}")
        raise

    elapsed = round(time.time() - start, 2)
    print(f"⏱ {name} completado en {elapsed}s")

    return result

# =====================================
# 1️⃣ DOWNLOAD
# =====================================
def download_kaggle_images(max_new_downloads=IMG, timeout=10, sleep_time=0.05):

    BASE_DIR = Path(__file__).resolve().parent.parent

    CSV_PATH = BASE_DIR / "data/datasets/kaggle_prefiltered.csv"
    OUTPUT_DIR = BASE_DIR / "data/images/kaggle_raw"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    HEADERS = {"User-Agent": "Mozilla/5.0"}

    print("\n🚀 Iniciando descarga incremental Kaggle")

    df = pd.read_csv(CSV_PATH, sep=",", engine="python")

    existing_files = {p.name for p in OUTPUT_DIR.glob("*.webp")}

    downloaded = 0
    skipped = 0

    # =====================================
    # PROGRESS BAR FUNCTION
    # =====================================

    def print_progress(current, total):

        bar_len = 30
        filled_len = int(bar_len * current / total)

        bar = "█" * filled_len + "░" * (bar_len - filled_len)

        print(
            f"\r⬇️ [{bar}] {current}/{total} | skipped:{skipped}",
            end="",
            flush=True
        )

    # =====================================
    # DOWNLOAD LOOP
    # =====================================

    for idx, row in df.iterrows():

        if downloaded >= max_new_downloads:
            break

        url = row["image"]
        filename = OUTPUT_DIR / f"img_{idx}.webp"

        if filename.name in existing_files:
            continue

        try:
            r = requests.get(url, timeout=timeout, headers=HEADERS)

            if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
                with open(filename, "wb") as f:
                    f.write(r.content)

                downloaded += 1
                existing_files.add(filename.name)

                # 🔥 ACTUALIZA PROGRESO
                print_progress(downloaded, max_new_downloads)

            else:
                skipped += 1

        except Exception:
            skipped += 1

        time.sleep(sleep_time)

    print("\n\n✅ FIN DESCARGA INCREMENTAL")
    print(f"Nuevas descargadas: {downloaded}")

# =====================================
# 2️⃣ FILTER FEATURES
# =====================================
def has_sky(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    mask = ((h > 90) & (h < 130))
    return np.sum(mask) / mask.size

def edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size

def line_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180, threshold=80,
        minLineLength=40, maxLineGap=5,
    )

    if lines is None:
        return 0

    return len(lines) / (img.shape[0] * img.shape[1])

def texture_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.std()

def brightness_uniformity(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.var()

def center_texture(img):
    h, w, _ = img.shape
    crop = img[h//4:3*h//4, w//4:3*w//4]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return gray.std()

def color_entropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    p = hist / np.sum(hist)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def edge_direction_ratio(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    vx = np.mean(np.abs(sobelx))
    vy = np.mean(np.abs(sobely))

    if vx + vy == 0:
        return 0

    return vy / (vx + vy)

def color_diversity(img, k=6):
    small = cv2.resize(img, (64, 64))
    data = small.reshape((-1, 3)).astype(np.float32)

    _, labels, _ = cv2.kmeans(
        data,
        k,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        3,
        cv2.KMEANS_RANDOM_CENTERS
    )

    return len(np.unique(labels)) / k

def saturation_variance(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1].std()

def compute_indoor_score(f):

    score = 0
    score += (1 - min(f["sky"]*3, 1)) * 0.12
    score += min(f["edges"]*10, 1) * 0.12
    score += min(f["lines"]*90000, 1) * 0.12
    score += min(f["texture"]/80, 1) * 0.12
    score += min(f["center_tex"]/60, 1) * 0.12
    score += min(f["entropy"]/6, 1) * 0.10
    score += min(f["brightness"]/6000, 1) * 0.10
    score += (1 - f["color_div"]) * 0.10
    score += (1 - min(f["sat_var"]/80, 1)) * 0.10

    return round(score, 4)

def filter_interiors(max_images=IMG):

    BASE_DIR = Path(__file__).resolve().parent.parent
    INPUT_DIR = BASE_DIR / "data/images/kaggle_raw"
    OUTPUT_CSV = BASE_DIR / "data/datasets/interior_filter.csv"

    rows = []

    # =====================================
    # LISTA DE IMÁGENES
    # =====================================

    images = list(INPUT_DIR.glob("*.webp"))

    if max_images is not None:
        images = images[:max_images]

    total = len(images)

    print(f"\n🧠 Analizando {total} imágenes...")

    # =====================================
    # PROGRESS BAR
    # =====================================

    def print_progress(current, total):

        bar_len = 30
        filled_len = int(bar_len * current / total)

        bar = "█" * filled_len + "░" * (bar_len - filled_len)

        print(
            f"\r🔍 [{bar}] {current}/{total}",
            end="",
            flush=True
        )

    # =====================================
    # LOOP PRINCIPAL
    # =====================================

    for i, img_path in enumerate(images, start=1):

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        f = {
            "sky": has_sky(img),
            "edges": edge_density(img),
            "lines": line_density(img),
            "texture": texture_variance(img),
            "brightness": brightness_uniformity(img),
            "center_tex": center_texture(img),
            "entropy": color_entropy(img),
            "edir": edge_direction_ratio(img),
            "color_div": color_diversity(img),
            "sat_var": saturation_variance(img),
        }

        indoor_score = compute_indoor_score(f)

        rows.append({
            "image_path": img_path.resolve().as_posix(),
            "indoor_score": indoor_score,
            **f
        })

        # 🔥 ACTUALIZA PROGRESO
        print_progress(i, total)

    print("\n")

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print("🔥 FILTER PRO COMPLETADO")

# =====================================
# 3️⃣ YOLO SEMANTIC
# =====================================
def get_yolo_model():
    global YOLO_MODEL

    if YOLO_MODEL is None:
        from ultralytics import YOLO
        print("🚀 Cargando YOLO GLOBAL...")
        YOLO_MODEL = YOLO(str(Path(__file__).parent / "yolov8n.pt"))

    return YOLO_MODEL



    BASE_DIR = Path(__file__).resolve().parent.parent
    CSV_PATH = BASE_DIR / "data/datasets/interior_filter.csv"
    OUTPUT_CSV = BASE_DIR / "data/datasets/interior_semantic.csv"

    df = pd.read_csv(CSV_PATH)
    MODEL = get_yolo_model()

    results = []

    for _, row in df.iterrows():

        img_path = row["image_path"]

        preds = MODEL(img_path, verbose=False)[0]

        detected_names = (
            [MODEL.names[int(c)] for c in preds.boxes.cls]
            if preds.boxes is not None else []
        )

        has_interior_object = any(
            obj in ["bed","couch","chair","dining table","tv","potted plant"]
            for obj in detected_names
        )

        results.append({
            "image_path": img_path,
            "indoor_score": row["indoor_score"],
            "has_semantic_interior": int(has_interior_object),
            "detected_objects": ",".join(detected_names)
        })

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    print("\n✅ YOLO semantic filter DONE")

def yolo_semantic_filter():

    BASE_DIR = Path(__file__).resolve().parent.parent
    CSV_PATH = BASE_DIR / "data/datasets/interior_filter.csv"
    OUTPUT_CSV = BASE_DIR / "data/datasets/interior_semantic.csv"

    df = pd.read_csv(CSV_PATH)
    MODEL = get_yolo_model()

    results = []

    total = len(df)

    print(f"\n🧠 Ejecutando YOLO semantic filter ({total} imágenes)")

    # =====================================
    # PROGRESS BAR
    # =====================================

    def print_progress(current, total):

        bar_len = 30
        filled_len = int(bar_len * current / total)

        bar = "█" * filled_len + "░" * (bar_len - filled_len)

        print(
            f"\r🤖 [{bar}] {current}/{total}",
            end="",
            flush=True
        )

    # =====================================
    # LOOP PRINCIPAL
    # =====================================

    for i, (_, row) in enumerate(df.iterrows(), start=1):

        img_path = row["image_path"]

        try:
            preds = MODEL(img_path, verbose=False)[0]
        except Exception as e:
            print(f"\n⚠️ Error YOLO con {img_path}: {e}")
            continue

        detected_names = (
            [MODEL.names[int(c)] for c in preds.boxes.cls]
            if preds.boxes is not None else []
        )

        has_interior_object = any(
            obj in ["bed", "couch", "chair", "dining table", "tv", "potted plant"]
            for obj in detected_names
        )

        results.append({
            "image_path": img_path,
            "indoor_score": row["indoor_score"],
            "has_semantic_interior": int(has_interior_object),
            "detected_objects": ",".join(detected_names)
        })

        # 🔥 ACTUALIZA PROGRESO
        print_progress(i, total)

    print("\n")

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    print("✅ YOLO semantic filter DONE")

# =====================================
# 4️⃣ FINAL DATASET
# =====================================
def create_final_dataset():

    BASE_DIR = Path(__file__).resolve().parent.parent
    INPUT_CSV = BASE_DIR / "data/datasets/interior_semantic.csv"
    OUTPUT_CSV = BASE_DIR / "data/datasets/interior_final_candidates.csv"

    df = pd.read_csv(INPUT_CSV)

    # 🔥 DATASET LIMPIO
    df["quality_bucket"] = "unknown"
    df["quality_bucket_human"] = ""
    df["final_quality"] = "unknown"

    df = df.astype({
        "quality_bucket": "string",
        "quality_bucket_human": "string",
        "final_quality": "string"
    })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("✅ FINAL DATASET creado (modo CLEAN)")

def human_label_step():

    print("\n🧠 HUMAN LABEL STEP (ACTIVE LEARNING CLEAN MODE)")

    while True:
        try:
            limit = int(input("👉 ¿Cuántas imágenes quieres etiquetar? "))
            break
        except:
            print("Introduce un número válido.")

    BASE_DIR = Path(__file__).resolve().parent.parent
    CSV_PATH = BASE_DIR / "data/datasets/interior_final_candidates.csv"

    df = pd.read_csv(CSV_PATH, dtype=str)
    df["quality_bucket_human"] = df["quality_bucket_human"].fillna("")

    pending = df[df["quality_bucket_human"] == ""].copy()

    print(f"Imágenes disponibles para etiquetar: {len(pending)}")

    count = 0

    for idx, row in pending.iterrows():

        if count >= limit:
            break

        img_path = Path(row["image_path"])

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        cv2.imshow("Human Label", img)

        print("\n1 = bad | 2 = medium | 3 = good | ESC = salir")

        key = cv2.waitKey(0)

        if key == 27:
            print("⛔ Salida manual.")
            break
        elif key == ord("1"):
            df.loc[idx, "quality_bucket_human"] = "bad"
        elif key == ord("2"):
            df.loc[idx, "quality_bucket_human"] = "medium"
        elif key == ord("3"):
            df.loc[idx, "quality_bucket_human"] = "good"

        count += 1

    cv2.destroyAllWindows()

    # 🔥 SOLO etiquetas humanas pasan a final_quality
    def merge_labels(row):
        human = str(row.get("quality_bucket_human", "")).strip()
        return human if human not in ["", "nan"] else "unknown"

    df["final_quality"] = df.apply(merge_labels, axis=1)

    df.to_csv(CSV_PATH, index=False)

    print(f"\n✅ Etiquetadas manualmente: {count}")

def prune_outdoor_images():

    print("\n🧹 Eliminando imágenes OUTDOOR según YOLO...")

    BASE_DIR = Path(__file__).resolve().parent.parent

    SEMANTIC_CSV = BASE_DIR / "data/datasets/interior_semantic.csv"
    IMAGE_DIR = BASE_DIR / "data/images/kaggle_raw"

    df = pd.read_csv(SEMANTIC_CSV)

    keep_paths = set(
        Path(p).resolve().as_posix()
        for p in df[df["has_semantic_interior"] == 1]["image_path"]
    )

    deleted = 0
    kept = 0

    for img in IMAGE_DIR.glob("*.webp"):

        norm = img.resolve().as_posix()

        if norm not in keep_paths:
            try:
                img.unlink()
                deleted += 1
            except Exception as e:
                print(f"⚠️ No se pudo borrar {img}: {e}")
        else:
            kept += 1

    print(f"✅ Imágenes interiores mantenidas: {kept}")
    print(f"❌ Imágenes outdoor eliminadas: {deleted}")

# =====================================
# 5️⃣ CLIP EMBEDDINGS
# =====================================
def extract_embeddings():

    from PIL import Image

    BASE_DIR = Path(__file__).resolve().parent.parent
    CSV_PATH = BASE_DIR / "data/datasets/interior_final_candidates.csv"
    OUTPUT_PATH = BASE_DIR / "data/embeddings/realestate_embeddings.parquet"

    df = pd.read_csv(CSV_PATH)

    # 🔥 SOLO DATA HUMANA
    df = df[df["final_quality"] != "unknown"].copy()

    rows = []

    for _, row in df.iterrows():

        img_path = Path(row["image_path"])

        if not img_path.exists():
            continue

        with Image.open(img_path) as im:
            emb = encode_image(im)

        rows.append({
            "image_path": img_path.as_posix(),
            "final_quality": row["final_quality"],
            "embedding": emb.tolist()
        })

    pd.DataFrame(rows).to_parquet(OUTPUT_PATH, index=False)

    print(f"✅ Embeddings creados SOLO con etiquetas humanas: {len(rows)}")

# =====================================
# BOOTSTRAP
# =====================================
def bootstrap_dataset(img_batch=IMG):

    print("\n========== BOOTSTRAP DATASET ==========")
    total_start = time.time()

    run_step("[1/7] DOWNLOAD", download_kaggle_images, max_new_downloads=img_batch)
    run_step("[2/7] FILTER", filter_interiors, max_images=None)
    run_step("[3/7] YOLO SEMANTIC", yolo_semantic_filter)
    run_step("[4/7] CREATE FINAL DATASET", create_final_dataset)
    run_step("[5/7] HUMAN LABEL", human_label_step)
    run_step("[6/7] PRUNE BAD IMAGES", prune_outdoor_images)
    run_step("[7/7] EXTRACT EMBEDDINGS", extract_embeddings)

    print("\n🏁 BOOTSTRAP TOTAL:", round(time.time() - total_start, 2), "s")

if __name__ == "__main__":
    bootstrap_dataset()
