import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.routes import router
from app.core.model_manager import model_manager
from app.core.config import settings
from app.core.db import init_db
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the database
    logger.info("Starting up: Initializing database...")
    init_db()
    
    # Startup: Load the model
    logger.info("Starting up: Loading model...")
    adapter_path = settings.ADAPTER_PATH if os.path.exists(settings.ADAPTER_PATH) else None
    success = model_manager.load_model(adapter_path)
    if success:
        logger.info("Model loaded successfully on startup.")
    else:
        logger.warning("Failed to load model on startup. It will be loaded on the first request.")
    
    yield
    # Shutdown: Clean up resources if needed
    logger.info("Shutting down: Releasing resources...")
    if model_manager.model:
        del model_manager.model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

app = FastAPI(
    title="Medical Chatbot API",
    description="Fine-tuned Qwen2-0.5B-Instruct for medical consultation.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/v1")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Medical Chatbot API",
        "docs": "/docs",
        "status": "online"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
