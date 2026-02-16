from fastapi import APIRouter, UploadFile, File, HTTPException

from .services import score_uploaded_image

router = APIRouter()


# =====================================
# SCORE ENDPOINT
# =====================================

@router.post("/score")
async def score_image(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    try:
        contents = await file.read()
        result = score_uploaded_image(contents)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
