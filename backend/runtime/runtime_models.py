from pathlib import Path
import torch

# =====================================================
# DEVICE GLOBAL
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

BASE_DIR = Path(__file__).resolve().parent.parent
PAIRWISE_PATH = BASE_DIR / "models/pairwise_ranker.joblib"


# =====================================================
# MODEL REGISTRY
# =====================================================

class ModelRegistry:

    def __init__(self):
        self.clip_model = None
        self.clip_preprocess = None
        self.pairwise_ranker = None

    # -------------------------------------------------
    # CLIP
    # -------------------------------------------------

    def get_clip(self):

        if self.clip_model is None:

            import open_clip

            print("🚀 [RUNTIME] Loading OpenCLIP model...")

            self.clip_model, _, self.clip_preprocess = (
                open_clip.create_model_and_transforms(
                    "ViT-B-32",
                    pretrained="openai",
                )
            )

            self.clip_model = self.clip_model.to(DEVICE)
            self.clip_model.eval()

            print("✅ CLIP ready")

        return self.clip_model, self.clip_preprocess

    # -------------------------------------------------
    # PAIRWISE RANKER
    # -------------------------------------------------

    def get_ranker(self):

        if self.pairwise_ranker is None:

            from joblib import load

            print("🚀 [RUNTIME] Loading Pairwise Ranker...")

            if not PAIRWISE_PATH.exists():
                raise RuntimeError(
                    f"No se encontró pairwise_ranker en: {PAIRWISE_PATH}"
                )

            self.pairwise_ranker = load(PAIRWISE_PATH)

            print("✅ Ranker ready")

        return self.pairwise_ranker


# =====================================================
# GLOBAL REGISTRY INSTANCE
# =====================================================

REGISTRY = ModelRegistry()


# =====================================================
# ENCODE IMAGE
# =====================================================

def encode_image(image_pil):

    model, preprocess = REGISTRY.get_clip()

    image_pil = image_pil.convert("RGB")

    image = preprocess(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu().numpy().astype("float32")[0]

    return emb


# =====================================================
# SCORE EMBEDDING
# =====================================================

def score_embedding(embedding):

    ranker = REGISTRY.get_ranker()

    raw_score = ranker.decision_function([embedding])[0]
    raw_score = float(raw_score)

    return {
        "score": raw_score,
        "margin": abs(raw_score),
    }
