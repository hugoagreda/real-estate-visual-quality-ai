from fastapi import FastAPI

from backend.api.routes import router


app = FastAPI(title="ImageScoreAI API")
app.include_router(router)
