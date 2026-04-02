import os
import torch
import gc
import logging
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import KTOTrainer, KTOConfig
from app.core.config import settings
from app.core.db import get_pending_feedback, mark_feedback_processed
from app.core.model_manager import model_manager
from app.training.trainer import training_status

logger = logging.getLogger(__name__)

def run_continuous_training():
    """Run KTO training on pending feedback from the SQLite database."""
    global training_status
    if training_status["status"] == "training":
        logger.warning("Training is already in progress. Aborting.")
        return

    training_status["status"] = "training"
    training_status["error"] = None
    
    try:
        # 1. Fetch pending feedback
        pending_data = get_pending_feedback()
        if not pending_data:
            logger.info("No pending feedback data found for training.")
            training_status["status"] = "idle"
            return

        # Threshold for demo
        threshold = 5 
        if len(pending_data) < threshold:
            logger.info(f"Only {len(pending_data)} pending items. Waiting for more data (threshold: {threshold}).")
            training_status["status"] = "idle"
            return

        logger.info(f"Starting continuous training on {len(pending_data)} feedback items.")

        # 2. Format for KTO (requires 'prompt', 'completion', 'label')
        # We transform the raw prompt from DB into a chat template
        prompts = []
        for item in pending_data:
            messages = [
                {"role": "system", "content": settings.SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]}
            ]
            prompts.append(messages)

        kto_data = {
            "prompt": prompts,
            "completion": [{"role": "assistant", "content": item["model_response"]} for item in pending_data],
            "label": [item["is_liked"] for item in pending_data]
        }
        dataset = Dataset.from_dict(kto_data)

        # Clear memory before starting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Load Model and Tokenizer
        compute_dtype = getattr(torch, settings.BNB_4BIT_COMPUTE_DTYPE)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=settings.LOAD_IN_4BIT,
            bnb_4bit_quant_type=settings.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_use_double_quant=settings.BNB_4BIT_USE_DOUBLE_QUANT,
            bnb_4bit_compute_dtype=compute_dtype
        )

        tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load Base Model
        model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)

        # 4. Initialize PEFT
        adapter_path = settings.ADAPTER_PATH if os.path.exists(settings.ADAPTER_PATH) else None
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )

        if adapter_path:
            logger.info(f"Loading existing adapter for continued training from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        else:
            model = get_peft_model(model, peft_config)

        # 5. KTO Training Arguments
        kto_args = KTOConfig(
            output_dir=os.path.join(settings.OUTPUT_DIR, "kto_continuous"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            logging_steps=1,
            num_train_epochs=1,
            optim="paged_adamw_8bit",
            bf16=(compute_dtype == torch.bfloat16),
            report_to="none",
            remove_unused_columns=False,
            beta=0.1,
            max_steps=max(1, len(pending_data) // 2)
        )

        # Initialize KTOTrainer
        trainer = KTOTrainer(
            model=model,
            args=kto_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        logger.info("Executing KTOTrainer.train()...")
        trainer.train()

        # 6. Save Model
        os.makedirs(settings.ADAPTER_PATH, exist_ok=True)
        trainer.save_model(settings.ADAPTER_PATH)
        logger.info(f"Continuous training complete. Updated adapter saved to {settings.ADAPTER_PATH}")

        # 7. Cleanup Training Weights from Memory
        del model
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 8. Mark Data Processed
        processed_ids = [item["id"] for item in pending_data]
        mark_feedback_processed(processed_ids)
        logger.info(f"Marked {len(processed_ids)} feedback records as processed.")

        # 9. Reload API Model
        logger.info("Reloading model for API serving...")
        success = model_manager.load_model(settings.ADAPTER_PATH)
        if success:
            logger.info("Model reloaded successfully with new adapter.")
            training_status["status"] = "success"
        else:
            logger.error("Failed to reload model after continuous training.")
            training_status["status"] = "failed"
            training_status["error"] = "Model reload failed"

    except Exception as e:
        logger.error(f"Continuous training failed: {e}")
        training_status["status"] = "failed"
        training_status["error"] = str(e)
        # Final cleanup attempt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
