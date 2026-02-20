"""
Visual Metrics Module
--------------------------------------------------
Explainable visual signals for Real Estate Visual Quality AI.

IMPORTANTE:
- Estas métricas NO deciden el score final.
- Solo aportan señales interpretables.
"""

import cv2
import numpy as np


# =====================================================
# LIGHTING
# =====================================================

def compute_lighting_score(image: np.ndarray) -> float:

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    dark_pixels_ratio = np.sum(gray < 30) / gray.size
    bright_pixels_ratio = np.sum(gray > 220) / gray.size

    brightness_score = 100 - abs(mean_intensity - 127) * 100 / 127
    brightness_score = np.clip(brightness_score, 0, 100)

    uniformity_score = 100 - std_intensity
    uniformity_score = np.clip(uniformity_score, 0, 100)

    extreme_penalty = (dark_pixels_ratio + bright_pixels_ratio) * 100

    lighting_score = (
        0.5 * brightness_score +
        0.3 * uniformity_score -
        0.2 * extreme_penalty
    )

    return float(np.clip(lighting_score, 0, 100))


# =====================================================
# SHARPNESS
# =====================================================

def compute_sharpness_score(image: np.ndarray) -> float:

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_variance = laplacian.var()

    min_var = 50
    max_var = 1000

    sharpness_score = (
        (laplacian_variance - min_var) /
        (max_var - min_var)
    ) * 100

    return float(np.clip(sharpness_score, 0, 100))


# =====================================================
# COMPOSITION
# =====================================================

def _score_angle_deviation(deviations: np.ndarray, max_deviation: float) -> float:
    if deviations.size == 0:
        return 50.0

    mean_deviation = float(np.mean(deviations))
    score = 100 - (mean_deviation / max_deviation) * 100
    return float(np.clip(score, 0, 100))


def compute_composition_score(image: np.ndarray) -> float:

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    height, width = edges.shape
    min_line_length = max(20, int(min(height, width) * 0.1))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=10
    )

    vertical_deviation = []
    horizontal_deviation = []

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue

            angle = np.degrees(np.arctan2(dy, dx))
            angle = angle % 180

            if 75 <= angle <= 105:
                vertical_deviation.append(abs(90 - angle))
            elif angle <= 15 or angle >= 165:
                horizontal_deviation.append(min(angle, 180 - angle))

    vertical_deviation = np.array(vertical_deviation, dtype=float)
    horizontal_deviation = np.array(horizontal_deviation, dtype=float)

    vertical_score = _score_angle_deviation(vertical_deviation, 15.0)
    horizontal_score = _score_angle_deviation(horizontal_deviation, 10.0)

    edge_points = np.column_stack(np.where(edges > 0))
    if edge_points.size == 0:
        balance_score = 50.0
    else:
        ys, xs = edge_points[:, 0], edge_points[:, 1]
        center_x = float(np.mean(xs))
        center_y = float(np.mean(ys))

        norm_dx = abs(center_x - (width / 2)) / (width / 2)
        norm_dy = abs(center_y - (height / 2)) / (height / 2)
        imbalance = (norm_dx + norm_dy) / 2

        balance_score = float(np.clip(100 - imbalance * 100, 0, 100))

    composition_score = (
        0.4 * vertical_score +
        0.3 * horizontal_score +
        0.3 * balance_score
    )

    return float(np.clip(composition_score, 0, 100))


# =====================================================
# COLOR
# =====================================================

def compute_color_score(image: np.ndarray) -> float:

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, a_channel, b_channel = cv2.split(lab)

    a_mean = np.mean(a_channel)
    b_mean = np.mean(b_channel)

    color_cast_deviation = abs(a_mean - 128) + abs(b_mean - 128)

    max_cast_deviation = 40.0
    balance_score = 100 - (color_cast_deviation / max_cast_deviation) * 100
    balance_score = np.clip(balance_score, 0, 100)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, saturation, _ = cv2.split(hsv)

    mean_saturation = np.mean(saturation)
    ideal_saturation = 80

    saturation_score = 100 - abs(mean_saturation - ideal_saturation) * 100 / ideal_saturation
    saturation_score = np.clip(saturation_score, 0, 100)

    color_score = 0.6 * balance_score + 0.4 * saturation_score

    return float(np.clip(color_score, 0, 100))


# =====================================================
# CLUTTER
# =====================================================

def compute_clutter_score(image: np.ndarray) -> float:

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size

    edge_density = edge_pixels / total_pixels

    max_edge_density = 0.15
    clutter_score = 100 - (edge_density / max_edge_density) * 100

    return float(np.clip(clutter_score, 0, 100))


# =====================================================
# GLOBAL METRICS ENTRYPOINT
# =====================================================

def compute_all_metrics(image: np.ndarray) -> dict:

    return {
        "lighting": compute_lighting_score(image),
        "sharpness": compute_sharpness_score(image),
        "composition": compute_composition_score(image),
        "color": compute_color_score(image),
        "clutter": compute_clutter_score(image),
    }
    
# =====================================================
# VISUAL REVIEW (EXPLAINABLE CRITIC)
# =====================================================

def visual_review(image, score: float, margin: float):

    """
    Genera explicación visual basada en:
    - score del modelo
    - métricas interpretables (lighting, sharpness, etc)
    """

    review = []
    caption = "Interior scene"

    # -------------------------------------
    # Convertir a numpy si viene PIL
    # -------------------------------------
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # PIL -> RGB
    if image.shape[-1] == 3:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image

    # -------------------------------------
    # METRICS
    # -------------------------------------
    metrics = compute_all_metrics(img_bgr)

    lighting = metrics["lighting"]
    sharpness = metrics["sharpness"]
    composition = metrics["composition"]
    color = metrics["color"]
    clutter = metrics["clutter"]

    # -------------------------------------
    # SCORE-BASED FEEDBACK (modelo real)
    # -------------------------------------
    if score > 0.85:
        review.append("✔ Imagen muy sólida visualmente")
    elif score > 0.65:
        review.append("👍 Buena base visual con margen de mejora")
    else:
        review.append("⚠️ Calidad visual baja según el modelo")

    if margin < 0.15:
        review.append("🤔 El modelo no está completamente seguro del ranking")

    # -------------------------------------
    # LIGHTING
    # -------------------------------------
    if lighting < 40:
        review.append("💡 Iluminación pobre o mal balanceada")
    elif lighting > 75:
        review.append("🌞 Iluminación natural bien equilibrada")

    # -------------------------------------
    # SHARPNESS
    # -------------------------------------
    if sharpness < 35:
        review.append("📉 Posible falta de nitidez o ligera borrosidad")
    elif sharpness > 80:
        review.append("🔎 Imagen muy nítida")

    # -------------------------------------
    # COMPOSITION
    # -------------------------------------
    if composition < 40:
        review.append("📐 Líneas inclinadas o encuadre mejorable")
    elif composition > 80:
        review.append("📏 Buena alineación arquitectónica")

    # -------------------------------------
    # COLOR
    # -------------------------------------
    if color < 40:
        review.append("🎨 Posible dominante de color o saturación poco natural")
    elif color > 75:
        review.append("🎨 Balance de color agradable")

    # -------------------------------------
    # CLUTTER
    # -------------------------------------
    if clutter < 35:
        review.append("🧱 Escena visualmente cargada o con exceso de objetos")
    elif clutter > 80:
        review.append("🧘 Espacio limpio y ordenado visualmente")

    # -------------------------------------
    # CAPTION SIMPLE (para UI)
    # -------------------------------------
    if clutter > 70 and lighting > 70:
        caption = "Clean and bright interior"
    elif clutter < 40:
        caption = "Busy interior scene"
    else:
        caption = "Standard interior"

    return {
        "caption": caption,
        "review": review,
        "metrics": metrics,   # 👈 esto es oro para el frontend luego
    }
