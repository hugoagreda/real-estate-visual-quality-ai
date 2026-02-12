from pathlib import Path
import torch

# ============================
# GLOBAL STATE
# ============================

CLIP_MODEL = None
CLIP_PREPROCESS = None
PAIRWISE_RANKER = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================
# PATHS
# ============================

BASE_DIR = Path(__file__).resolve().parent.parent

PAIRWISE_PATH = BASE_DIR / "models/pairwise_ranker.joblib"


# ============================
# LOAD CLIP (LAZY GLOBAL)
# ============================

def get_clip():

    global CLIP_MODEL, CLIP_PREPROCESS

    if CLIP_MODEL is None:
        import open_clip

        print("🚀 [RUNTIME] Loading OpenCLIP model...")

        CLIP_MODEL, _, CLIP_PREPROCESS = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
        )

        CLIP_MODEL = CLIP_MODEL.to(DEVICE)
        CLIP_MODEL.eval()

        print("✅ CLIP ready")

    return CLIP_MODEL, CLIP_PREPROCESS


# ============================
# LOAD PAIRWISE RANKER
# ============================

def get_ranker():

    global PAIRWISE_RANKER

    if PAIRWISE_RANKER is None:
        from joblib import load

        print("🚀 [RUNTIME] Loading Pairwise Ranker...")

        if not PAIRWISE_PATH.exists():
            raise RuntimeError(
                f"No se encontró pairwise_ranker en: {PAIRWISE_PATH}"
            )

        PAIRWISE_RANKER = load(PAIRWISE_PATH)

        print("✅ Ranker ready")

    return PAIRWISE_RANKER


# ============================
# ENCODE IMAGE
# ============================

def encode_image(image_pil):

    model, preprocess = get_clip()

    image = preprocess(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb.cpu().numpy()[0]

    return emb


# ============================
# SCORE IMAGE
# ============================

def score_embedding(embedding):

    ranker = get_ranker()

    # fast ranking → dot product con vector w
    score = ranker.decision_function([embedding])[0]

    return float(score)
