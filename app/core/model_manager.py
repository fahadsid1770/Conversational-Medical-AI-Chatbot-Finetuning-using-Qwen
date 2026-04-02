import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
        return cls._instance

    def load_model(self, adapter_path: str = None):
        """Load the base model and optionally an adapter."""
        try:
            # Configure BitsAndBytes for QLoRa
            compute_dtype = getattr(torch, settings.BNB_4BIT_COMPUTE_DTYPE)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=settings.LOAD_IN_4BIT,
                bnb_4bit_quant_type=settings.BNB_4BIT_QUANT_TYPE,
                bnb_4bit_use_double_quant=settings.BNB_4BIT_USE_DOUBLE_QUANT,
                bnb_4bit_compute_dtype=compute_dtype
            )

            logger.info(f"Loading base model: {settings.MODEL_ID}")
            base_model = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto"
            )

            # Load adapter if path is provided
            if adapter_path and os.path.exists(adapter_path):
                logger.info(f"Loading adapter from: {adapter_path}")
                # For 4-bit models, we usually don't merge and unload as it requires de-quantization
                self.model = PeftModel.from_pretrained(base_model, adapter_path)
            else:
                self.model = base_model

            logger.info("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def generate_response(self, user_query: str, history: list = None, system_prompt: str = None):
        """Generate a response using the model."""
        if not self.model or not self.tokenizer:
            success = self.load_model(settings.ADAPTER_PATH if os.path.exists(settings.ADAPTER_PATH) else None)
            if not success:
                return "Error: Model not loaded."

        system_content = system_prompt or settings.SYSTEM_PROMPT
        
        messages = [{"role": "system", "content": system_content}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_query})

        # Apply Chat template
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize input
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        # Generate (Aligned with notebook parameters)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        # Decode
        response_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response_text

model_manager = ModelManager()
