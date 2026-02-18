# 🧠 Real Estate Visual Quality AI — ImageScoreAI

Sistema de Inteligencia Artificial para evaluar calidad visual de imágenes inmobiliarias mediante aprendizaje incremental humano + auto-labeling.

El proyecto combina:

- CLIP embeddings
- Ranking pairwise
- Clasificación visual explainable
- Active learning humano
- Auto-labeling progresivo
- API REST
- App móvil (Expo / React Native)

---

# 📌 Objetivo

Evaluar automáticamente imágenes de interiores y:

- ✅ Calcular score visual
- ✅ Analizar composición, iluminación, color, clutter y nitidez
- ✅ Generar review explicable
- ✅ Aprender continuamente con feedback humano

---

# ⚙️ Dependencias

Instalar entorno:

```bash
pip install fastapi uvicorn
pip install pandas numpy
pip install scikit-learn joblib
pip install pillow opencv-python
pip install open_clip_torch
pip install pyarrow
pip install ultralytics
