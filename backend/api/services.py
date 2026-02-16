from PIL import Image
from io import BytesIO

from backend.runtime.runtime_score import score_image_pil


# =====================================
# SCORE SERVICE
# =====================================

def score_uploaded_image(file_bytes: bytes):

    try:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Invalid image file: {e}")

    result = score_image_pil(image)

    return result
