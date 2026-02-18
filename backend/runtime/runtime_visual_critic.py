from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# =====================================
# LOAD MODEL ONCE (GLOBAL)
# =====================================

print("🧠 Loading Visual Critic (BLIP)...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("✅ Visual Critic ready")


# =====================================
# GENERATE CAPTION
# =====================================
def generate_caption(image: Image.Image):

    inputs = processor(images=image, return_tensors="pt").to(device)

    out = model.generate(
        **inputs,
        max_length=50
    )

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.lower()


# =====================================
# SIMPLE VISUAL REVIEW
# =====================================

def visual_review(image: Image.Image, score=None, margin=None):
    
    caption = generate_caption(image)

    feedback = []

    # =====================================
    # REGLAS VISUALES BASE
    # =====================================

    if "clutter" in caption or "messy" in caption:
        feedback.append("⚠️ Demasiados objetos visibles en la escena")

    if "dark" in caption or "dim" in caption:
        feedback.append("⚠️ Iluminación mejorable")

    if "modern" in caption or "clean" in caption:
        feedback.append("✔ Estética limpia y moderna")

    if "living room" in caption or "bedroom" in caption:
        feedback.append("✔ Espacio interior bien identificado")

    # =====================================
    # 🔥 REGLAS CONTEXTUALES CON EL MODELO
    # =====================================

    if score is not None:

        if score > 0.75:
            feedback.append("🧠 El modelo considera esta imagen visualmente fuerte")

        elif score < 0.4:
            feedback.append("⚠️ El modelo detecta baja calidad estética general")

    if margin is not None:

        if margin < 0.15:
            feedback.append("⚠️ Baja confianza del modelo → posible composición ambigua")

        elif margin > 0.4:
            feedback.append("✔ Alta claridad visual según el modelo")

    # fallback
    if len(feedback) == 0:
        feedback.append("✔ Escena visualmente correcta")

    return {
        "caption": caption,
        "review": feedback
    }
