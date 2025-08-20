from fastapi import FastAPI
from .routes import router

app = FastAPI(
    title="Gistify API",
    description="API for summarizing text using the Gistify model.",
    version="1.0.0",
)

app.include_router(router)