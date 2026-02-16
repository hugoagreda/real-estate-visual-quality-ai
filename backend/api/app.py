from fastapi import FastAPI
from .routes import router

app = FastAPI(
    title="ImageScoreAI API",
    version="0.1",
)

# register routes
app.include_router(router)


@app.get("/")
def root():
    return {"status": "ok", "service": "ImageScoreAI"}
