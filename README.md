# 🧠 ImageScoreAI Backend

Backend para scoring visual de imágenes interiores usando embeddings CLIP + modelo de ranking, con salida explicable (caption + review) y exposición vía API/Gradio.

---

## ⚙️ Requisitos

- Python 3.10+ (recomendado 3.10/3.11)
- (Opcional) GPU CUDA para acelerar inferencia

Como no hay `requirements.txt` en el estado actual del repo, instala dependencias con:

```bash
pip install fastapi uvicorn python-multipart
pip install numpy pandas scikit-learn joblib pyarrow
pip install pillow opencv-python requests
pip install torch transformers
pip install open_clip_torch ultralytics
pip install gradio
pip install matplotlib seaborn
```

## 🏋️ Entrenamiento del modelo

### 🧑 Modo humano (recomendado para calidad)

```bash
python backend/pipeline/dataset_pipeline.py
```

Flujo que ejecuta:

- Descarga incremental de imágenes Kaggle prefiltradas
- Filtro visual + filtro semántico YOLO
- Creación/actualización de `interior_final_candidates.csv`
- Etiquetado humano interactivo (teclas: `1=bad`, `2=medium`, `3=good`, `ESC` para salir)
- Extracción de embeddings CLIP
- Entrenamiento (`training_pipeline`)

### 🤖 Modo auto (sin etiquetado manual)

```bash
python backend/pipeline/dataset_pipeline.py --auto
```

Este modo usa auto-labeling cuando existe:

- `backend/models/quality_head.joblib`

## 📊 Visualización del dataset (en las 2 opciones)

Este modo **no modifica el entrenamiento**, solo añade análisis visual al finalizar el pipeline.

Genera automáticamente:

- Histograma de `room_score`
- Scatter `indoor_score` vs `room_score`
- Distribución `quality_bucket`
- Histograma de `auto_confidence`

Ideal para validar que el auto-learning no introduce sesgos.

## 📈 ¿Qué muestran las gráficas?

1. **Distribución `room_score`**  
	Permite verificar si el filtro semántico está siendo demasiado estricto o permisivo.

2. **Indoor vs Room Score**  
	Visualiza cómo el filtro YOLO afecta al scoring interior.

3. **Quality Buckets**  
	Ayuda a detectar desbalance entre clases:
	- `bad`
	- `medium`
	- `good`

4. **Auto Confidence**  
	Muestra cómo se distribuye la confianza del modelo durante el auto-labeling.

## 🔎 Verificación post-entrenamiento

Confirma que se generaron/actualizaron:

- `backend/models/pairwise_ranker.joblib`
- `backend/models/quality_head.joblib`
- `backend/data/embeddings/human_embeddings.parquet` (modo humano)
- `backend/data/embeddings/auto_round_embeddings.parquet` (modo auto)

## ⚠️ Notas importantes

- Ejecuta estos comandos desde la raíz del repo.
- El modo humano abre ventanas con OpenCV (`cv2.imshow`), por lo que requiere entorno con UI.
- La primera ejecución puede ser lenta por carga de modelos y descarga de datos.
- Las gráficas son solo visualización y no afectan al entrenamiento.

## ▶️ Levantar la app

### 1️⃣ Levantar app visual (Gradio)

En otra terminal:

```bash
setx HF_TOKEN "TU_TOKEN_HUGGINGFACE"
python -m backend.api.visual_app
```

Tienes 2 opciones de acceso en `backend/api/visual_app.py`:

- Opción A (local): `share=False` → solo http://127.0.0.1:7860
- Opción B (enlace temporal): `share=True` → URL pública temporal `*.gradio.live`

Nota: en este proyecto actualmente está configurado con `share=True`.

## ✅ ¿Conviene tener API y app visual a la vez?

Sí, en la mayoría de casos conviene tener ambas porque cumplen funciones distintas:

- **API (FastAPI):** integración con app móvil/web, automatizaciones y consumo programático.
- **App visual (Gradio):** pruebas rápidas, demos y validación manual de resultados.

## 🧠 Notas finales

- El sistema usa CLIP embeddings + ranking model como núcleo.
- El Visual Critic genera explicación textual sin afectar al scoring.
- La primera carga puede tardar por descarga de modelos (BLIP, CLIP, YOLO).
- Después de la primera ejecución, los modelos se cargan desde cache local.
- El backend está preparado para consumo desde app móvil, web o cliente desktop.