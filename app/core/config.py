import os
import torch
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Model configuration
    MODEL_ID: str = "Qwen/Qwen2-0.5B-Instruct"
    ADAPTER_PATH: str = "./models/qwen2-medical-adapter"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # QLoRa configuration
    LOAD_IN_4BIT: bool = True
    BNB_4BIT_QUANT_TYPE: str = "nf4"
    BNB_4BIT_USE_DOUBLE_QUANT: bool = True
    BNB_4BIT_COMPUTE_DTYPE: str = "bfloat16" if torch.cuda.is_available() else "float32"

    # Training configuration
    DATASET_NAME: str = "ruslanmv/ai-medical-chatbot"
    OUTPUT_DIR: str = "./models/qwen2-medical-chatbot-v1"
    NUM_TRAIN_EPOCHS: int = 1
    PER_DEVICE_TRAIN_BATCH_SIZE: int = 4
    GRADIENT_ACCUMULATION_STEPS: int = 4
    LEARNING_RATE: float = 2e-4
    LOGGING_STEPS: int = 10
    SAVE_STEPS: int = 50
    EVAL_STEPS: int = 50
    
    # System Prompt
    SYSTEM_PROMPT: str = (
        "You are a helpful AI medical assistant. "
        "Please provide informative and safe medical advice. "
        "Do not provide a diagnosis. Advise the user to consult a professional."
    )

    class Config:
        case_sensitive = True

settings = Settings()
