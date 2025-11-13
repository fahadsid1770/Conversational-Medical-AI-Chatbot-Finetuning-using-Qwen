import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig
from trl.trainer.sft_trainer import SFTTrainer
from datasets import load_dataset



def main():
    # Model and Tokenizer Loading
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    adapter_path = "./qwen2-0.5b-healthcare-agent"

    # 4-bit Quantization (QLoRA) Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    #LoRA Config As per [21, 56]
    peft_config = LoraConfig(
        r=16,  # LoRA rank (higher is more expressive, more params)
        lora_alpha=32,  # A scaling factor
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Target all linear layers in attention and MLP blocks
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    #Load Model and Tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically uses GPU
        trust_remote_code=True # Qwen2 models require this
    )
    
    #Qwen2 tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # A Causal LM requires a pad token, and eos_token is a common choice
    tokenizer.pad_token = tokenizer.eos_token

    # Load Dataset The dataset is expected to be in ChatML format
    dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train")

    # Training Arguments
    # Hyperparameters are optimized for 8-12GB VRAM
    training_args = TrainingArguments(
        output_dir=adapter_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        num_train_epochs=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        bf16=True,  # Use bf16 if on Ampere+ GPU (e.g., RTX 30/40xx)
        fp16=False, # Set fp16=True if on older GPU
        logging_steps=10,
        save_strategy="epoch"
    )

    # Initialize SFTTrainer
    # SFTTrainer will automatically apply the tokenizer's chat template to the "messages" column of our dataset.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # Start Training
    print("Starting QLoRA Fine-Tuning")
    trainer.train()
    
    # Save Final Adapter
    trainer.save_model(adapter_path)
    print(f"Training complete. Adapter saved to {adapter_path}")

if __name__ == "__main__":
    main()