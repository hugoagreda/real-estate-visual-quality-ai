from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

from backend.api.services import score_image_service

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/score")
async def score_image(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Archivo no es imagen")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo imagen: {e}")

    try:
        result = score_image_service(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring: {e}")

    return result
