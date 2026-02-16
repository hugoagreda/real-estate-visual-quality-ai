python -m backend.pipeline.dataset_pipeline (HUMAN)

python -m backend.pipeline.training_pipeline --auto (AUTO)

uvicorn backend.api.app:app --reload

http://127.0.0.1:8000/docs

CONFIGURAR NGROK
ngrok http 8000 (DEPENDIENDO DEL PUERTO)