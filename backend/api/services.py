"""
API Services Layer
--------------------------------------------------
Separación entre routes (HTTP) y runtime (IA).

Aquí vive la lógica real del producto.
"""

from PIL import Image
from backend.runtime.runtime_score import score_image_pil


# =====================================================
# SCORE SERVICE
# =====================================================

def score_image_service(image: Image.Image) -> dict:
    """
    Servicio central de scoring.

    - Ejecuta modelo
    - Ejecuta visual critic
    - Devuelve respuesta lista para API
    """

    result = score_image_pil(
        image,
        return_embedding=False,
        with_review=True
    )

    # 🔥 Aquí luego podrás añadir:
    # - guardar feedback
    # - métricas usage
    # - logs
    # - AB testing

    return result
