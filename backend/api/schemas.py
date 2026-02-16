from pydantic import BaseModel


# =====================================
# REQUEST
# =====================================

class ScoreRequest(BaseModel):
    image_base64: str


# =====================================
# RESPONSE
# =====================================

class ScoreResponse(BaseModel):
    quality_label: str
    score: float
    margin: float
