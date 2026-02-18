import gradio as gr
from PIL import Image

# 🔥 IMPORT REAL
from backend.runtime.runtime_score import score_image_pil


# =====================================
# PREDICT FUNCTION
# =====================================

def predict(files):

    if files is None:
        return []

    results = []

    for f in files:

        try:
            img = Image.open(f.name).convert("RGB")

            # 🔥 llamada directa a tu runtime
            out = score_image_pil(img)

            text = f"Score: {round(out['score'],3)} | Margin: {round(out['margin'],3)}"

            results.append((f.name, text))

        except Exception as e:
            results.append((f.name, f"ERROR: {str(e)}"))

    return results


# =====================================
# UI
# =====================================

with gr.Blocks(title="ImageScoreAI Demo") as demo:

    gr.Markdown("# 🧠 ImageScoreAI Visual Demo")
    gr.Markdown("Sube imágenes interiores y el modelo calcula score visual.")

    uploader = gr.File(
        file_count="multiple",
        file_types=["image"],
        label="Subir imágenes"
    )

    run_btn = gr.Button("Analizar")

    gallery = gr.Gallery(label="Resultados")

    run_btn.click(
        fn=predict,
        inputs=uploader,
        outputs=gallery
    )


if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        share=True   # 🔥 link público para tus compañeros
    )
