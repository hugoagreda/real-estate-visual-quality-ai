from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
from pathlib import Path
import time

# =====================================
# IMPORT RUNTIME
# =====================================

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.runtime_score import scan_image, score_image
from runtime.runtime_models import encode_image  # solo para warmup

# =====================================
# CONFIG
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
TMP_DIR = BASE_DIR / "data/tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ImageScoreAI API")

# =====================================
# CORS (IMPORTANTE PARA MOVIL)
# =====================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# WARMUP MODELOS
# =====================================
@app.on_event("startup")
def warmup_models():
    print("\n🔥 Warming up models...")

    try:
        # fuerza carga CLIP
        dummy = encode_image
        print("✅ CLIP encoder ready")
    except Exception as e:
        print(f"⚠️ Warmup CLIP fallo: {e}")

# =====================================
# UTILS
# =====================================
def save_temp_file(upload_file: UploadFile) -> Path:

    ext = Path(upload_file.filename).suffix.lower()
    filename = f"{uuid.uuid4().hex}{ext}"

    file_path = TMP_DIR / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return file_path

# =====================================
# ROOT
# =====================================
@app.get("/")
def root():
    return {"status": "ImageScoreAI API running"}

# =====================================
# 🔥 FULL SCAN (YOLO + METRICS + RANKER)
# =====================================
@app.post("/scan")
async def scan(file: UploadFile = File(...)):

    start = time.time()

    try:

        temp_path = save_temp_file(file)

        result = scan_image(str(temp_path))

        elapsed = round(time.time() - start, 3)

        return {
            "status": "ok",
            "mode": "full_scan",
            "time": elapsed,
            "data": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except:
            pass

# =====================================
# ⚡ FAST SCORE (SOLO CLIP + RANKER)
# =====================================

@app.post("/score-lite")
async def score_lite(file: UploadFile = File(...)):

    start = time.time()

    try:

        temp_path = save_temp_file(file)

        result = score_image(str(temp_path))

        elapsed = round(time.time() - start, 3)

        return {
            "status": "ok",
            "mode": "lite",
            "time": elapsed,
            "data": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except:
            pass
