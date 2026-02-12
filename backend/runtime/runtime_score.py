from pathlib import Path
from PIL import Image

from .runtime_models import encode_image, score_embedding


# =====================================
# SCORE IMAGE (HIGH LEVEL API)
# =====================================

def score_image(image_path: str):

    """
    Recibe ruta de imagen
    Devuelve score visual global
    """

    img_path = Path(image_path).resolve()

    if not img_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {img_path}")

    # =====================
    # LOAD IMAGE
    # =====================

    image = Image.open(img_path).convert("RGB")

    # =====================
    # EMBEDDING
    # =====================

    embedding = encode_image(image)

    # =====================
    # SCORE
    # =====================

    score = score_embedding(embedding)

    return {
        "image_path": str(img_path),
        "score": score,
    }


# =====================================
# SCORE BATCH (para desktop testing)
# =====================================

def score_folder(folder_path: str):

    folder = Path(folder_path).resolve()

    if not folder.exists():
        raise FileNotFoundError(folder)

    results = []

    for img_path in folder.glob("*.webp"):

        try:
            r = score_image(str(img_path))
            results.append(r)

        except Exception as e:
            print(f"⚠️ Error con {img_path}: {e}")

    # ordenar por score
    results.sort(key=lambda x: x["score"], reverse=True)

    return results
