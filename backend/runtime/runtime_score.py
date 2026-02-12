from pathlib import Path
from PIL import Image
from .runtime_models import encode_image, score_embedding

VALID_EXT = {".webp", ".jpg", ".jpeg", ".png"}

# =====================================
# SCORE IMAGE (HIGH LEVEL API)
# =====================================

def score_image(image_path: str, return_embedding: bool = False):
    """
    Recibe ruta de imagen
    Devuelve score visual global

    Params:
        image_path (str)
        return_embedding (bool): opcional para debugging o clustering

    Returns:
        dict
    """

    img_path = Path(image_path).resolve()

    if not img_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {img_path}")

    # =====================
    # LOAD IMAGE (SAFE)
    # =====================

    try:
        with Image.open(img_path) as img:
            image = img.convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Error loading image {img_path}: {e}")

    # =====================
    # EMBEDDING
    # =====================

    embedding = encode_image(image)

    # =====================
    # SCORE
    # =====================

    score = score_embedding(embedding)

    result = {
        "image_path": str(img_path),
        "score": float(score),
    }

    if return_embedding:
        result["embedding"] = embedding

    return result

# =====================================
# SCORE BATCH (DESKTOP / TESTING)
# =====================================

def score_folder(folder_path: str):
    """
    Scoring batch de una carpeta completa.

    Devuelve lista ordenada por score desc.
    """

    folder = Path(folder_path).resolve()

    if not folder.exists():
        raise FileNotFoundError(folder)

    results = []

    for img_path in folder.iterdir():

        if img_path.suffix.lower() not in VALID_EXT:
            continue

        try:
            r = score_image(str(img_path))
            results.append(r)

        except Exception as e:
            print(f"⚠️ Error con {img_path}: {e}")

    # ordenar por score
    results.sort(key=lambda x: x["score"], reverse=True)

    return results
