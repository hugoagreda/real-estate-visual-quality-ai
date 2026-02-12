import pandas as pd
import numpy as np
import cv2
import requests
from pathlib import Path
import time
import torch

IMG = 100
YOLO_MODEL = None
CLIP_MODEL = None
CLIP_PREPROCESS = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# ===== 1. DOWNLOAD ===== #
def download_kaggle_images(
    max_new_downloads=IMG,
    timeout=10,
    sleep_time=0.05,
):

    # =========================
    # PATHS ROBUSTOS
    # =========================

    BASE_DIR = Path(__file__).resolve().parent.parent

    CSV_PATH = BASE_DIR / "data/datasets/kaggle_prefiltered.csv"
    OUTPUT_DIR = BASE_DIR / "data/images/kaggle_raw"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    HEADERS = {"User-Agent": "Mozilla/5.0"}

    print("\n🚀 Iniciando descarga incremental Kaggle")

    # =========================
    # CARGAR DATASET
    # =========================

    df = pd.read_csv(CSV_PATH, sep=",", engine="python")

    # =========================
    # DETECTAR EXISTENTES
    # =========================

    existing_files = {p.name for p in OUTPUT_DIR.glob("*.webp")}

    downloaded = 0
    skipped = 0
    already_present = 0

    print(f"📦 Imágenes ya existentes detectadas: {len(existing_files)}")

    # =========================
    # PIPELINE
    # =========================

    for idx, row in df.iterrows():

        if downloaded >= max_new_downloads:
            break

        url = row["image"]
        filename = OUTPUT_DIR / f"img_{idx}.webp"

        if filename.name in existing_files:
            already_present += 1
            continue

        try:
            r = requests.get(url, timeout=timeout, headers=HEADERS)

            if r.status_code == 200 and "image" in r.headers.get(
                "Content-Type", ""
            ):
                with open(filename, "wb") as f:
                    f.write(r.content)

                downloaded += 1
                existing_files.add(filename.name)
            else:
                skipped += 1

        except Exception:
            skipped += 1

        if downloaded % 100 == 0 and downloaded != 0:
            print(
                f"⬇️ nuevas descargadas: {downloaded} | "
                f"ya existentes: {already_present} | "
                f"saltadas: {skipped}"
            )

        time.sleep(sleep_time)

    # =========================
    # RESULTADO FINAL
    # =========================

    print("\n✅ FIN DESCARGA INCREMENTAL")
    print(f"Nuevas descargadas: {downloaded}")
    print(f"Ya existentes: {already_present}")
    print(f"Saltadas: {skipped}")

# ===== 2. FILTER ===== #
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
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=40,
        maxLineGap=5,
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

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, labels, centers = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_RANDOM_CENTERS
    )

    unique_labels = len(np.unique(labels))
    return unique_labels / k

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

    if OUTPUT_CSV.exists():
        df_results = pd.read_csv(OUTPUT_CSV)

        processed = set(
            Path(p).resolve().as_posix()
            for p in df_results["image_path"].values
        )
    else:
        df_results = pd.DataFrame()
        processed = set()

    count = 0
    rows = []

    for img_path in INPUT_DIR.glob("*.webp"):

        if max_images is not None and count >= max_images:
            break

        norm_path = img_path.resolve().as_posix()

        if norm_path in processed:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        f = {}

        f["sky"] = has_sky(img)
        f["edges"] = edge_density(img)
        f["lines"] = line_density(img)
        f["texture"] = texture_variance(img)
        f["brightness"] = brightness_uniformity(img)
        f["center_tex"] = center_texture(img)
        f["entropy"] = color_entropy(img)
        f["edir"] = edge_direction_ratio(img)
        f["color_div"] = color_diversity(img)
        f["sat_var"] = saturation_variance(img)

        indoor_score = compute_indoor_score(f)

        rows.append({
            "image_path": norm_path,
            "indoor_score": indoor_score,
            "visual_class": "candidate",
            **f
        })

        count += 1

        if count % 50 == 0:
            print(f"Procesadas nuevas: {count}")

    if rows:
        df_new = pd.DataFrame(rows)
        df_results = pd.concat([df_results, df_new], ignore_index=True)
        df_results.to_csv(OUTPUT_CSV, index=False)

    print("\n🔥 FILTER PRO V3 COMPLETADO")
    print(f"Nuevas procesadas: {count}")
    print(f"Total acumulado: {len(df_results)}")

# ===== 3. SEMANTIC FILTER (YOLO) ===== #
def get_yolo_model():
    global YOLO_MODEL

    if YOLO_MODEL is None:
        from ultralytics import YOLO
        print("🚀 Cargando YOLO GLOBAL...")
        YOLO_MODEL = YOLO("yolov8n.pt")

    return YOLO_MODEL

def yolo_semantic_filter():

    print("\n🧠 Ejecutando YOLO semantic filter")

    BASE_DIR = Path(__file__).resolve().parent.parent

    CSV_PATH = BASE_DIR / "data/datasets/interior_filter.csv"
    OUTPUT_CSV = BASE_DIR / "data/datasets/interior_semantic.csv"

    INTERIOR_CLASSES = [
        "bed",
        "couch",
        "chair",
        "dining table",
        "tv",
        "potted plant",
    ]

    df = pd.read_csv(CSV_PATH)

    # =========================
    # INCREMENTAL LOAD (NORMALIZADO)
    # =========================

    if OUTPUT_CSV.exists():
        df_done = pd.read_csv(OUTPUT_CSV)

        processed = set(
            Path(p).resolve().as_posix()
            for p in df_done["image_path"].values
        )

        results = df_done.to_dict("records")
    else:
        processed = set()
        results = []

    print(f"📊 Ya procesadas YOLO: {len(processed)}")

    MODEL = get_yolo_model()

    new_count = 0

    for _, row in df.iterrows():

        img_path = Path(row["image_path"]).resolve().as_posix()

        if img_path in processed:
            continue

        try:
            preds = MODEL(img_path, verbose=False)[0]
        except Exception as e:
            print(f"⚠️ Error YOLO con {img_path}: {e}")
            continue

        detected_names = (
            [MODEL.names[int(c)] for c in preds.boxes.cls]
            if preds.boxes is not None else []
        )

        has_interior_object = any(
            obj in INTERIOR_CLASSES for obj in detected_names
        )

        results.append({
            "image_path": img_path,
            "indoor_score": row["indoor_score"],
            "has_semantic_interior": int(has_interior_object),
            "detected_objects": ",".join(detected_names)
        })

        new_count += 1

        if new_count % 50 == 0:
            print(f"YOLO nuevas procesadas: {new_count}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print("\n✅ YOLO semantic filter DONE")
    print(f"Nuevas procesadas: {new_count}")
    print(f"Total acumulado: {len(df_out)}")

# ===== 4. FINAL DATASET ===== #
def create_final_dataset():

    print("\n🧩 Creando FINAL DATASET (fusion visual + YOLO)")

    BASE_DIR = Path(__file__).resolve().parent.parent

    INPUT_CSV = BASE_DIR / "data/datasets/interior_semantic.csv"
    OUTPUT_CSV = BASE_DIR / "data/datasets/interior_final_candidates.csv"

    VISUAL_THRESHOLD = 0.70

    df = pd.read_csv(INPUT_CSV)

    print(f"📊 Total imágenes iniciales: {len(df)}")

    # =====================
    # FILTRO FINAL
    # =====================

    df_final = df[
        (df["indoor_score"] < VISUAL_THRESHOLD) &
        (df["has_semantic_interior"] == 1)
    ].copy()

    print(f"🏠 Interiores confirmados: {len(df_final)}")

    # =====================
    # QUALITY BUCKET AUTO
    # =====================

    def assign_quality(row):

        detected = str(row["detected_objects"])

        if detected.strip() == "":
            obj_count = 0
        else:
            obj_count = len(detected.split(","))

        if row["indoor_score"] < 0.45 and obj_count >= 2:
            return "good"

        if row["indoor_score"] < 0.60:
            return "medium"

        return "bad"

    df_final["quality_bucket"] = df_final.apply(assign_quality, axis=1)

    # columnas necesarias para pipeline completo
    df_final["quality_bucket_human"] = ""

    # =====================
    # FUSIÓN FINAL
    # =====================

    def merge_labels(row):

        human = str(row.get("quality_bucket_human", "")).strip()

        if human != "" and human != "nan":
            return human

        return row["quality_bucket"]

    df_final["final_quality"] = df_final.apply(merge_labels, axis=1)

    # =====================
    # SAVE
    # =====================

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False)

    print("\n✅ FINAL DATASET creado")
    print(f"📁 Guardado en: {OUTPUT_CSV}")

    print("\n📊 Distribución final_quality:")
    print(df_final["final_quality"].value_counts())

# ===== 5. EMBEDDINGS ===== #
def get_clip_model():
    global CLIP_MODEL, CLIP_PREPROCESS

    if CLIP_MODEL is None:
        import open_clip

        print("🚀 Cargando OpenCLIP GLOBAL...")

        CLIP_MODEL, _, CLIP_PREPROCESS = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
        )

        CLIP_MODEL = CLIP_MODEL.to(DEVICE)
        CLIP_MODEL.eval()

    return CLIP_MODEL, CLIP_PREPROCESS

def extract_embeddings():
    from PIL import Image

    print("\n🧠 Extrayendo embeddings OpenCLIP (INCREMENTAL)")

    BASE_DIR = Path(__file__).resolve().parent.parent
    CSV_PATH = BASE_DIR / "data/datasets/interior_final_candidates.csv"
    OUTPUT_PATH = BASE_DIR / "data/embeddings/realestate_embeddings.parquet"

    # =====================
    # LOAD DATASET
    # =====================

    df = pd.read_csv(CSV_PATH)
    df = df[df["final_quality"].notna()].copy()

    print(f"📊 Total imágenes dataset: {len(df)}")

    # =====================
    # LOAD EXISTING EMBEDDINGS (INCREMENTAL REAL)
    # =====================

    if OUTPUT_PATH.exists():
        df_old = pd.read_parquet(OUTPUT_PATH)

        processed = set(
            Path(p).resolve().as_posix()
            for p in df_old["image_path"].values
        )

        embeddings = df_old.to_dict("records")

        print(f"♻️ Embeddings existentes: {len(processed)}")
    else:
        processed = set()
        embeddings = []

    # =====================
    # LOAD MODEL GLOBAL (LAZY)
    # =====================

    model, preprocess = get_clip_model()

    new_count = 0

    # =====================
    # PIPELINE
    # =====================

    for idx, row in df.iterrows():

        img_path = Path(row["image_path"]).resolve()
        norm_path = img_path.as_posix()

        if norm_path in processed:
            continue

        if not img_path.exists():
            print(f"❌ No existe: {img_path}")
            continue

        try:
            image = preprocess(
                Image.open(img_path).convert("RGB")
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb = model.encode_image(image)
                emb = emb.cpu().numpy()[0].tolist()

            embeddings.append({
                "image_path": norm_path,
                "final_quality": row["final_quality"],
                "embedding": emb
            })

            new_count += 1

            if new_count % 50 == 0:
                print(f"📦 Nuevas embeddings: {new_count}")

        except Exception as e:
            print(f"⚠️ Error con {img_path}: {e}")

    # =====================
    # SAVE
    # =====================

    print("\n💾 Guardando embeddings...")

    df_emb = pd.DataFrame(embeddings)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_emb.to_parquet(OUTPUT_PATH, index=False)

    print("\n✅ Embeddings actualizados")
    print(f"Nuevas embeddings: {new_count}")
    print(f"Total embeddings acumulados: {len(df_emb)}")

# ===== BOOTSTRAP ===== #
def bootstrap_dataset(img_batch=IMG):

    print("\n========== BOOTSTRAP DATASET ==========")
    total_start = time.time()

    run_step("[1/5] DOWNLOAD", download_kaggle_images, max_new_downloads=img_batch)
    run_step("[2/5] FILTER", filter_interiors, max_images=None)
    run_step("[3/5] YOLO SEMANTIC", yolo_semantic_filter)
    run_step("[4/5] CREATE FINAL DATASET", create_final_dataset)
    run_step("[5/5] EXTRACT EMBEDDINGS", extract_embeddings)

    total_end = time.time()
    print("\n🏁 BOOTSTRAP TOTAL:", round(total_end - total_start, 2), "s")

if __name__ == "__main__":
    bootstrap_dataset()