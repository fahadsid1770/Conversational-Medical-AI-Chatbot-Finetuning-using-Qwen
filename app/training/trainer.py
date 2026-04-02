import os
import torch
import logging
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl.trainer.sft_trainer import SFTTrainer
from app.core.config import settings
from typing import cast, Dict, Any

logger = logging.getLogger(__name__)

# Global state to track training status
training_status = {
    "status": "idle",  # idle, training, success, failed
    "last_loss": None,
    "current_step": 0,
    "total_steps": 0,
    "error": None
}

def format_chat_template(row):
    messages = [
        {"role": "system", "content": settings.SYSTEM_PROMPT},
        {"role": "user", "content": row["Patient"]},
        {"role": "assistant", "content": row["Doctor"]},
    ]
    return {"messages": messages}

def run_training(params: Dict[str, Any] = None):
    """Function to run QLoRa training. Should be called as a background task."""
    global training_status
    training_status["status"] = "training"
    training_status["error"] = None
    
    try:
        # Configuration
        dataset_name = params.get("dataset_name", settings.DATASET_NAME)
        num_epochs = params.get("num_epochs", settings.NUM_TRAIN_EPOCHS)
        
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split="train")
        if isinstance(dataset, DatasetDict):
            dataset = dataset["train"]
        dataset = cast(Dataset, dataset)
        dataset = dataset.shuffle(seed=42)

        # Preprocessing
        processed_dataset = dataset.map(
            format_chat_template,
            remove_columns=dataset.features
        )
        
        # Split
        training_dataset = processed_dataset.select(range(min(900, len(processed_dataset))))
        eval_dataset = processed_dataset.select(range(min(900, len(processed_dataset)), min(1000, len(processed_dataset))))

        # BitsAndBytes for training
        compute_dtype = getattr(torch, settings.BNB_4BIT_COMPUTE_DTYPE)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=settings.LOAD_IN_4BIT,
            bnb_4bit_quant_type=settings.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_use_double_quant=settings.BNB_4BIT_USE_DOUBLE_QUANT,
            bnb_4bit_compute_dtype=compute_dtype
        )

        logger.info(f"Loading base model for training: {settings.MODEL_ID}")
        model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token

        # PEFT Configuration
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=settings.OUTPUT_DIR,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=settings.PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=settings.GRADIENT_ACCUMULATION_STEPS,
            optim="paged_adamw_8bit",
            learning_rate=settings.LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=settings.LOGGING_STEPS,
            eval_strategy="steps",
            eval_steps=settings.EVAL_STEPS,
            save_strategy="steps",
            save_steps=settings.SAVE_STEPS,
            save_total_limit=3,
            load_best_model_at_end=True,
            bf16=(compute_dtype == torch.bfloat16),
            report_to="none", # avoid tensorboard for simplicity in non-interactive
        )

        # Initialize the SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
        )

        logger.info("Starting model training")
        trainer.train()

        # Save the final adapter weights
        os.makedirs(settings.ADAPTER_PATH, exist_ok=True)
        trainer.save_model(settings.ADAPTER_PATH)
        logger.info(f"Training complete. Adapter saved to {settings.ADAPTER_PATH}")
        
        training_status["status"] = "success"
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        training_status["status"] = "failed"
        training_status["error"] = str(e)
