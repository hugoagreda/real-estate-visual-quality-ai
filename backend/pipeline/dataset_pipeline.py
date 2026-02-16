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
import joblib

from runtime.runtime_models import encode_image
from pipeline.training_pipeline import training_pipeline

IMG = 100 # 500 para auto
YOLO_MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# UTILS
# =====================================
def run_step(name, func, *args, **kwargs):

    print(f"\n{'='*50}")
    print(f"▶️ {name}")
    print(f"{'='*50}")

    start = time.time()

    try:
        result = func(*args, **kwargs)
    except Exception as e:
        print(f"\n💥 ERROR en {name}: {e}")
        raise

    elapsed = round(time.time() - start, 2)

    print(f"\n⏱ {name} completado en {elapsed}s")

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

    print("\n🚀 Iniciando descarga Kaggle (sin tracker)")

    df = pd.read_csv(CSV_PATH, sep=",", engine="python")

    existing_files = {p.name for p in OUTPUT_DIR.glob("*.webp")}

    downloaded = 0
    skipped = 0

    def print_progress(current, total):

        bar_len = 30
        filled_len = int(bar_len * current / total)
        bar = "█" * filled_len + "░" * (bar_len - filled_len)

        print(
            f"\r⬇️ [{bar}] {current}/{total} | skipped:{skipped}",
            end="",
            flush=True
        )

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

                print_progress(downloaded, max_new_downloads)

            else:
                skipped += 1

        except Exception:
            skipped += 1

        time.sleep(sleep_time)

    print("\n📊 DOWNLOAD STATS")
    print(f"  ✔ Nuevas: {downloaded}")
    print(f"  ❌ Fallidas: {skipped}")  

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
    edges = cv2.Canny(gray, 50, 120)
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
    
    return len(lines) / 1000

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

def compute_indoor_score(f):
    score = 0
    score += (1 - min(f["sky"]*2, 1)) * 0.08
    score += min(f["edges"]*6, 1) * 0.14
    score += min(f["lines"], 1) * 0.12
    score += min(f["texture"]/80, 1) * 0.14
    score += min(f["center_tex"]/60, 1) * 0.14
    score += min(f["entropy"]/6, 1) * 0.10
    score += min(f["brightness"]/10000, 1) * 0.08
    score += f["color_div"] * 0.05
    score += (1 - min(f["sat_var"]/80, 1)) * 0.05

    return round(score, 4)

def filter_interiors(max_images=IMG):

    BASE_DIR = Path(__file__).resolve().parent.parent
    INPUT_DIR = BASE_DIR / "data/images/kaggle_raw"
    FINAL_CSV = BASE_DIR / "data/datasets/interior_final_candidates.csv"

    rows = []

    # =====================================
    # 🔥 CARGAR TODAS LAS IMÁGENES FÍSICAS
    # =====================================
    images = list(INPUT_DIR.glob("*.webp"))

    # =====================================
    # 🔥 FIX CRÍTICO — FILTRADO INCREMENTAL REAL
    # =====================================
    if FINAL_CSV.exists():

        df_existing = pd.read_csv(FINAL_CSV, dtype=str)

        if "image_path" in df_existing.columns:
            existing_paths = set(df_existing["image_path"].astype(str))
        else:
            existing_paths = set()

        images = [
            p for p in images
            if p.resolve().as_posix() not in existing_paths
        ]

    # =====================================
    # LIMIT OPCIONAL
    # =====================================
    if max_images is not None:
        images = images[:max_images]

    total = len(images)

    if total == 0:
        print("\n⚠️ No hay imágenes nuevas para feature filter.")
        return None

    print(f"\n🧠 Analizando {total} imágenes NUEVAS (feature filter)...")

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

        try:
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
                "is_new": True,   # 🔥 IMPORTANTÍSIMO PARA YOLO
                **f
            })

        except Exception as e:
            print(f"\n⚠️ Error procesando {img_path}: {e}")

        print_progress(i, total)

    print("\n")

    if len(rows) == 0:
        print("⚠️ No se generaron features.")
        return None

    df_filter = pd.DataFrame(rows)

    # =====================================
    # STATS
    # =====================================
    print("\n📊 FILTER STATS")
    print("  indoor_score mean:", round(df_filter["indoor_score"].mean(),3))
    print("  indoor_score std:", round(df_filter["indoor_score"].std(),3))
    print("  edges mean:", round(df_filter["edges"].mean(),3))

    return df_filter

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

def yolo_semantic_filter(df_filter):

    BASE_DIR = Path(__file__).resolve().parent.parent
    OUTPUT_CSV = BASE_DIR / "data/datasets/interior_semantic.csv"

    if df_filter is None or len(df_filter) == 0:
        print("⚠️ DataFrame vacío recibido en YOLO semantic filter.")
        return None

    df = df_filter.copy()

    # =====================================
    # SOLO NUEVAS
    # =====================================
    if "is_new" in df.columns:
        df = df[df["is_new"] == True].copy()

    total = len(df)

    if total == 0:
        print("⚠️ No hay imágenes nuevas para YOLO semantic filter.")
        return None

    MODEL = get_yolo_model()

    CORE_OBJECTS = {
        "bed","couch","chair","dining table",
        "tv","refrigerator","sink","oven","microwave"
    }

    SECONDARY_OBJECTS = {
        "potted plant","person","umbrella",
        "vase","clock","book","laptop"
    }

    results = []

    print(f"\n🧠 Ejecutando YOLO semantic filter GOD ({total} imágenes)")

    def print_progress(current, total):
        bar_len = 30
        filled_len = int(bar_len * current / total)
        bar = "█"*filled_len + "░"*(bar_len-filled_len)
        print(f"\r🤖 [{bar}] {current}/{total}", end="", flush=True)

    # =====================================
    # LOOP PRINCIPAL
    # =====================================
    for i, (_, row) in enumerate(df.iterrows(), start=1):

        img_path = row["image_path"]
        indoor_score = float(row.get("indoor_score", 0))

        try:
            preds = MODEL(img_path, verbose=False)[0]
        except Exception as e:
            print(f"\n⚠️ Error YOLO con {img_path}: {e}")
            continue

        detected_names = (
            [MODEL.names[int(c)] for c in preds.boxes.cls]
            if preds.boxes is not None else []
        )

        unique_objects = set(detected_names)

        core_objects_present = unique_objects.intersection(CORE_OBJECTS)
        secondary_objects_present = unique_objects.intersection(SECONDARY_OBJECTS)

        core_count = len(core_objects_present)
        secondary_count = len(secondary_objects_present)

        variety_score = min(core_count / 2, 1.0)
        core_presence = 1.0 if core_count >= 1 else 0.0
        secondary_noise = secondary_count / (core_count + secondary_count + 1)

        room_score = (
            0.5 * variety_score +
            0.3 * indoor_score +
            0.2 * core_presence
        ) * (1 - secondary_noise)

        room_score = round(float(room_score), 4)

        results.append({
            "image_path": img_path,
            "indoor_score": indoor_score,
            "room_score": room_score,
            "core_count": core_count,
            "secondary_count": secondary_count,
            "detected_objects": ",".join(detected_names)
        })

        print_progress(i, total)

    print("\n")

    df_semantic = pd.DataFrame(results)

    if len(df_semantic) == 0:
        print("⚠️ YOLO semantic filter no generó resultados.")
        return None

    # =====================================
    # 🔥 THRESHOLD ADAPTATIVO INTELIGENTE
    # =====================================

    BASE_THRESHOLD = 0.55
    MIN_THRESHOLD = 0.45

    room_scores = df_semantic["room_score"].values

    # Si el lote es pequeño o los scores vienen bajos → ajustar
    if len(room_scores) < 40:

        dynamic_thr = np.percentile(room_scores, 85)
        dynamic_thr = max(dynamic_thr, MIN_THRESHOLD)
        dynamic_thr = min(dynamic_thr, BASE_THRESHOLD)

        print(f"🧠 Threshold dinámico activado: {round(dynamic_thr,3)}")

    else:
        dynamic_thr = BASE_THRESHOLD

    df_semantic["has_semantic_interior"] = (
        df_semantic["room_score"] >= dynamic_thr
    ).astype(int)

    # =====================================
    # GUARDADO
    # =====================================
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_semantic.to_csv(OUTPUT_CSV, index=False)

    interiors = (df_semantic["has_semantic_interior"] == 1).sum()
    total = len(df_semantic)

    print("\n📊 YOLO STATS")
    print(f"  interiores detectados: {interiors}/{total}")
    print(f"  ratio interior: {round(interiors/total,3)}")
    print(f"  room_score mean: {round(df_semantic['room_score'].mean(),3)}")
    print(f"  room_score max : {round(df_semantic['room_score'].max(),3)}")
    print(f"  room_score min : {round(df_semantic['room_score'].min(),3)}")

    return df_semantic

# =====================================
# 4️⃣ FINAL DATASET
# =====================================
def prune_bad_images(df_semantic):

    print("\n🧹 Eliminando imágenes OUTDOOR según YOLO...")

    import pandas as pd
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    IMAGE_DIR = BASE_DIR / "data/images/kaggle_raw"

    if df_semantic is None or len(df_semantic) == 0:
        print("⚠️ No hay datos semánticos para prune.")
        return df_semantic

    # =====================================
    # 🔥 PATHS QUE SE CONSERVAN
    # =====================================
    keep_paths = set(
        Path(p).resolve().as_posix()
        for p in df_semantic[
            df_semantic["has_semantic_interior"] == 1
        ]["image_path"]
    )

    deleted = 0
    kept = 0

    # =====================================
    # 🔥 SOLO BORRAR ARCHIVOS FÍSICOS
    # =====================================
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

    print("\n📊 PRUNE STATS")
    print(f"  interiores mantenidos: {kept}")
    print(f"  outdoor eliminadas: {deleted}")

    # =====================================
    # 🔥 IMPORTANTÍSIMO
    # =====================================
    # NO tocar interior_final_candidates.csv
    # NO modificar datasets
    # SOLO filesystem

    return df_semantic

def create_final_dataset(df_semantic=None):

    import pandas as pd
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    INPUT_CSV = BASE_DIR / "data/datasets/interior_semantic.csv"
    OUTPUT_CSV = BASE_DIR / "data/datasets/interior_final_candidates.csv"

    # =====================================
    # SOURCE DATA
    # =====================================
    if df_semantic is None:

        if not INPUT_CSV.exists():
            print("⚠️ No existe interior_semantic.csv")
            return

        df_new = pd.read_csv(INPUT_CSV)

    else:
        df_new = df_semantic.copy()

    # =====================================
    # SOLO INTERIORES SEGÚN YOLO
    # =====================================
    df_new = df_new[df_new["has_semantic_interior"] == 1].copy()

    if len(df_new) == 0:
        print("⚠️ No hay nuevas imágenes interiores.")
        return

    # =====================================
    # COMPATIBILIDAD room_score / semantic_score
    # =====================================
    if "room_score" not in df_new.columns:

        if "semantic_score" in df_new.columns:
            df_new["room_score"] = df_new["semantic_score"].astype(float)
        else:
            print("⚠️ No existe room_score ni semantic_score.")
            return

    # =====================================
    # 🔥 AUTO QUALITY CORREGIDO (YA NO ELIMINA BAD)
    # =====================================
    def assign_quality(score):
        try:
            s = float(score)
        except:
            return None

        if s >= 0.75:
            return "good"
        elif s >= 0.60:
            return "medium"
        else:
            return "bad"

    df_new["quality_bucket"] = df_new["room_score"].apply(assign_quality)

    before = len(df_new)
    df_new = df_new[df_new["quality_bucket"].notna()].copy()
    after = len(df_new)

    print(f"🧹 Filtrado por room_score: {before} → {after}")

    if len(df_new) == 0:
        print("⚠️ Ninguna imagen supera el filtro.")
        return

    # =====================================
    # CARGAR DATASET EXISTENTE SI EXISTE
    # =====================================
    if OUTPUT_CSV.exists():

        df_old = pd.read_csv(OUTPUT_CSV, dtype=str)

        df_old["quality_bucket_human"] = df_old.get(
            "quality_bucket_human", ""
        ).fillna("")

        df_old["final_quality"] = df_old.get(
            "final_quality", ""
        ).fillna("")

        df_old = df_old.set_index("image_path")
        df_new = df_new.set_index("image_path")

        # 🔥 SOLO añadir imágenes realmente nuevas
        new_only = df_new[~df_new.index.isin(df_old.index)].copy()

        new_only["quality_bucket_human"] = ""
        new_only["final_quality"] = new_only["quality_bucket"]
        new_only["is_new"] = True

        df_final = pd.concat([df_old, new_only])
        df_final = df_final.reset_index()

    else:

        df_new["quality_bucket_human"] = ""
        df_new["final_quality"] = df_new["quality_bucket"]
        df_new["is_new"] = True

        df_final = df_new.reset_index()

    # =====================================
    # SI TIENE LABEL HUMANO → NO ES NEW
    # =====================================
    if "quality_bucket_human" in df_final.columns:

        df_final.loc[
            df_final["quality_bucket_human"].fillna("") != "",
            "is_new"
        ] = False

    # =====================================
    # TYPES SEGUROS
    # =====================================
    df_final["quality_bucket"] = df_final["quality_bucket"].astype("string")
    df_final["quality_bucket_human"] = df_final["quality_bucket_human"].astype("string")
    df_final["final_quality"] = df_final["final_quality"].astype("string")
    df_final["is_new"] = df_final["is_new"].astype(bool)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False)

    # =====================================
    # LOGS
    # =====================================
    print("\n📊 DATASET STATE")
    print("  total dataset:", len(df_final))
    print("  nuevas imágenes:", int(df_final["is_new"].sum()))

    human_count = (df_final["quality_bucket_human"] != "").sum()
    print("  etiquetas humanas:", int(human_count))

    print("\nDistribución final_quality:")
    print(df_final["final_quality"].value_counts())

    return df_final

def human_label_step():

    print("\n🧠 HUMAN LABEL STEP (ACTIVE LEARNING REAL)")

    import cv2
    import pandas as pd
    from pathlib import Path

    # =====================================
    # INPUT LIMIT
    # =====================================
    while True:
        try:
            limit = int(input("👉 ¿Cuántas imágenes quieres etiquetar? "))
            break
        except:
            print("Introduce un número válido.")

    BASE_DIR = Path(__file__).resolve().parent.parent
    CSV_PATH = BASE_DIR / "data/datasets/interior_final_candidates.csv"

    if not CSV_PATH.exists():
        print("⚠️ No existe interior_final_candidates.csv")
        return

    # =====================================
    # LOAD DATA
    # =====================================
    df = pd.read_csv(CSV_PATH, dtype=str)

    # 🔥 FIX CRÍTICO: convertir is_new REALMENTE a bool
    if "is_new" in df.columns:
        df["is_new"] = df["is_new"].astype(str).str.lower() == "true"
    else:
        df["is_new"] = True

    if "quality_bucket_human" not in df.columns:
        df["quality_bucket_human"] = ""

    df["quality_bucket_human"] = df["quality_bucket_human"].fillna("")

    # =====================================
    # 🔥 SOLO NUEVAS Y SIN LABEL HUMANO
    # =====================================
    pending = df[
        (df["is_new"] == True) &
        (df["quality_bucket_human"] == "")
    ].copy()

    total_pending = len(pending)

    print("\n📊 HUMAN LABEL STATS")
    print(f"  pendientes reales: {total_pending}")

    already_human = (df["quality_bucket_human"] != "").sum()
    print(f"  ya etiquetadas humano: {already_human}")

    total_new = df["is_new"].sum()
    print(f"  imágenes marcadas new: {int(total_new)}")

    if total_pending == 0:
        print("⚠️ No hay imágenes pendientes de etiquetar.")
        return

    # =====================================
    # LOOP VISUAL
    # =====================================
    count = 0

    for idx, row in pending.iterrows():

        if count >= limit:
            break

        img_path = Path(row["image_path"])

        if not img_path.exists():
            print(f"⚠️ No existe imagen: {img_path}")
            continue

        img = cv2.imread(str(img_path))

        if img is None:
            print(f"⚠️ Error leyendo imagen: {img_path}")
            continue

        cv2.imshow("Human Label Active", img)
        cv2.waitKey(1)

        print("\n1 = bad | 2 = medium | 3 = good | ESC = salir")

        key = cv2.waitKey(0)

        if key == 27:
            print("⛔ Salida manual.")
            break

        elif key == ord("1"):
            df.loc[idx, "quality_bucket_human"] = "bad"
            df.loc[idx, "final_quality"] = "bad"

        elif key == ord("2"):
            df.loc[idx, "quality_bucket_human"] = "medium"
            df.loc[idx, "final_quality"] = "medium"

        elif key == ord("3"):
            df.loc[idx, "quality_bucket_human"] = "good"
            df.loc[idx, "final_quality"] = "good"

        else:
            continue

        # 🔥 IMPORTANTÍSIMO → evitar que vuelva a salir
        df.loc[idx, "is_new"] = False

        count += 1

    cv2.destroyAllWindows()

    # =====================================
    # SAVE
    # =====================================
    df.to_csv(CSV_PATH, index=False)

    print(f"\n✅ Etiquetadas manualmente (Active Learning REAL): {count}")

# =====================================
# 5️⃣ CLIP EMBEDDINGS
# =====================================
def extract_embeddings(auto_mode=False):

    from PIL import Image

    BASE_DIR = Path(__file__).resolve().parent.parent
    CSV_PATH = BASE_DIR / "data/datasets/interior_final_candidates.csv"

    # =====================================
    # OUTPUT SEGÚN MODO
    # =====================================
    if auto_mode:
        OUTPUT_PATH = BASE_DIR / "data/embeddings/auto_round_embeddings.parquet"
        label_source = "auto"
        print("\n🤖 Modo AUTO embeddings")
    else:
        OUTPUT_PATH = BASE_DIR / "data/embeddings/human_embeddings.parquet"
        label_source = "human"
        print("\n🧑 Modo HUMAN embeddings")

    if not CSV_PATH.exists():
        print("❌ No existe CSV dataset.")
        return

    df = pd.read_csv(CSV_PATH, dtype=str)

    # FIX dtype
    if "is_new" in df.columns:
        df["is_new"] = df["is_new"].astype(str).str.lower() == "true"
    else:
        df["is_new"] = True

    # =====================================
    # 🔥 LÓGICA CORRECTA
    # AUTO = SIEMPRE INCREMENTAL
    # HUMAN = full solo primera vez
    # =====================================
    if auto_mode:
        df_new = df[df["is_new"] == True].copy()
    else:
        if not OUTPUT_PATH.exists():
            print("🆕 Primera ejecución detectada → generando embeddings completos")
            df_new = df.copy()
        else:
            df_new = df[df["is_new"] == True].copy()

    df_new = df_new[df_new["final_quality"].notna()].copy()

    total = len(df_new)

    if total == 0:
        print("⚠️ No hay imágenes nuevas para embeddings.")
        return

    rows = []

    print(f"\n🧠 Extrayendo embeddings ({total} imágenes)")

    for i, (_, row) in enumerate(df_new.iterrows(), 1):

        img_path = Path(row["image_path"])

        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as im:
                emb = encode_image(im)

            rows.append({
                "image_path": img_path.as_posix(),
                "final_quality": row["final_quality"],
                "embedding": emb.tolist(),
                "label_source": label_source
            })

        except Exception as e:
            print(f"⚠️ Error con {img_path}: {e}")

        if i % 50 == 0 or i == total:
            print(f"🧠 [{i}/{total}] embeddings")

    if len(rows) == 0:
        print("⚠️ No se generaron embeddings.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # =====================================
    # APPEND SEGURO
    # =====================================
    df_new_emb = pd.DataFrame(rows)

    if OUTPUT_PATH.exists():
        df_old = pd.read_parquet(OUTPUT_PATH)

        df_final = pd.concat([df_old, df_new_emb], ignore_index=True)

        # 🔥 IMPORTANTE: dedupe por imagen + source
        df_final = df_final.drop_duplicates(
            subset=["image_path", "label_source"],
            keep="last"
        )
    else:
        df_final = df_new_emb

    df_final.to_parquet(OUTPUT_PATH, index=False)

    print(f"✅ Embeddings añadidos: {len(rows)}")
    print(f"📂 Guardados en: {OUTPUT_PATH}")

    processed_paths = set(df_new["image_path"])

    # SOLO HUMAN CIERRA CICLO
    if not auto_mode:
        df.loc[df["image_path"].isin(processed_paths), "is_new"] = False

    df["is_new"] = df["is_new"].astype(bool)
    df.to_csv(CSV_PATH, index=False)

    print("\n📊 EMBEDDINGS STATS")
    print("  nuevas embeddings:", total)

# =====================================
# 6️⃣ SECOND ROUND
# =====================================
def auto_label_step():

    print("\n🤖 AUTO LABEL STEP (MODEL ASSISTED)")

    BASE_DIR = Path(__file__).resolve().parent.parent

    CSV_PATH = BASE_DIR / "data/datasets/interior_final_candidates.csv"
    EMB_DIR = BASE_DIR / "data/embeddings"
    MODEL_PATH = BASE_DIR / "models/quality_head.joblib"

    if not MODEL_PATH.exists():
        print("⚠️ No existe quality_head. Auto label cancelado.")
        return

    files = [
        EMB_DIR / "human_embeddings.parquet",
        EMB_DIR / "auto_round_embeddings.parquet",
    ]

    dfs = [pd.read_parquet(f) for f in files if f.exists()]

    if not dfs:
        print("⚠️ No existen embeddings. Auto label cancelado.")
        return

    df_emb = pd.concat(dfs, ignore_index=True)

    df = pd.read_csv(CSV_PATH, dtype=str)

    df["quality_bucket_human"] = df["quality_bucket_human"].fillna("")
    df["is_new"] = df["is_new"].astype(str).str.lower().isin(["true","1"])

    pending = df[
        (df["quality_bucket_human"] == "") &
        (df["is_new"] == True)
    ].copy()

    if len(pending) == 0:
        print("✅ No hay imágenes nuevas que auto-etiquetar.")
        return

    data = joblib.load(MODEL_PATH)
    model = data["model"]
    le = data["label_encoder"]

    merged = pending.merge(
        df_emb[["image_path", "embedding"]],
        on="image_path",
        how="left"
    )

    merged = merged[merged["embedding"].notna()]

    if len(merged) == 0:
        print("⚠️ No hay embeddings disponibles para auto label.")
        return

    X = np.vstack(merged["embedding"].values)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    X = X / norms

    proba = model.predict_proba(X)
    preds = np.argmax(proba, axis=1)
    labels = le.inverse_transform(preds)

    updated = 0

    for i, (img_path, label) in enumerate(zip(merged["image_path"], labels)):

        mask = df["image_path"] == img_path

        df.loc[mask, "final_quality"] = label
        df.loc[mask, "auto_quality"] = label
        df.loc[mask, "auto_confidence"] = float(np.max(proba[i]))
        df.loc[mask, "label_source"] = "auto"

        # 🔥 NO tocar is_new aquí
        updated += 1

    df.to_csv(CSV_PATH, index=False)

    print("\n📊 AUTO LABEL STATS")
    print("  candidatas:", len(pending))
    print("  con embedding:", len(merged))

# =====================================
# BOOTSTRAP
# =====================================
def bootstrap_dataset(auto_mode=False, img_batch=IMG):

    if auto_mode:
        print("\n========== BOOTSTRAP DATASET (AUTO MODE) ==========")
    else:
        print("\n========== BOOTSTRAP DATASET (HUMAN MODE) ==========")

    total_start = time.time()

    run_step("[1/8] DOWNLOAD", download_kaggle_images, max_new_downloads=img_batch)
    df_filter = run_step("[2/8] FILTER", filter_interiors, max_images=None)
    df_semantic = run_step("[3/8] YOLO SEMANTIC", yolo_semantic_filter, df_filter)
    run_step("[4/8] PRUNE BAD IMAGES", prune_bad_images, df_semantic)
    run_step("[5/8] CREATE FINAL DATASET", create_final_dataset, df_semantic)

    if auto_mode:
        run_step("[6/8] EXTRACT EMBEDDINGS", extract_embeddings, True)
        run_step("[7/8] AUTO LABEL", auto_label_step)

    else:

        run_step("[6/8] HUMAN LABEL", human_label_step)
        run_step("[7/8] EXTRACT EMBEDDINGS", extract_embeddings, False)

    run_step("[8/8] TRAINING PIPELINE", training_pipeline)

    print("\n🏁 BOOTSTRAP TOTAL:", round(time.time() - total_start, 2), "s")

if __name__ == "__main__":

    import sys

    # 🔥 Detecta si ejecutas con --auto
    auto_mode = "--auto" in sys.argv

    bootstrap_dataset(auto_mode=auto_mode)
