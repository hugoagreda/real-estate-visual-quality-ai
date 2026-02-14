from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid

# 🔥 IMPORTAMOS TU RUNTIME REAL
from runtime.runtime_score import scan_image

app = FastAPI(title="ImageScoreAI API")

# =====================================
# CORS (para app móvil / pruebas)
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# TEMP DIR
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "temp_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# =====================================
# HEALTH CHECK
# =====================================

@app.get("/")
def root():
    return {"status": "ok", "service": "ImageScoreAI"}

# =====================================
# SCAN IMAGE ENDPOINT
# =====================================

@app.post("/scan-image")
async def scan_image_endpoint(file: UploadFile = File(...)):

    # ---------------------------------
    # GUARDAR TEMPORALMENTE
    # ---------------------------------

    temp_name = f"{uuid.uuid4()}.webp"
    temp_path = UPLOAD_DIR / temp_name

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ---------------------------------
    # EJECUTAR SCAN COMPLETO
    # ---------------------------------

    try:
        result = scan_image(str(temp_path))

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

    finally:
        # 🔥 BORRAMOS SIEMPRE
        if temp_path.exists():
            temp_path.unlink()

    # ---------------------------------
    # RESPUESTA LIMPIA
    # ---------------------------------

    return {
        "success": True,
        "score": result["score"],
        "metrics": result["metrics"],
        "detected_objects": result["detected_objects"]
    }
