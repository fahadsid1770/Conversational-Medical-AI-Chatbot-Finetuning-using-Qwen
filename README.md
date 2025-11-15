# Medical Chatbot Fine-tuning with Qwen2-0.5B

A medical chatbot fine-tuned from Qwen/Qwen2-0.5B-Instruct using QLoRA (Quantized Low-Rank Adaptation) technique. This project demonstrates how to efficiently fine-tune a language model for medical assistance while maintaining computational efficiency through 4-bit quantization.

## 🚀 Features

- **Efficient Fine-tuning**: Uses QLoRa technique with 4-bit quantization to reduce memory usage
- **Medical Domain Focus**: Trained on medical chatbot conversations
- **Optimized Training**: Configured with cosine learning rate scheduling and gradient accumulation
- **Model Deployment**: Ready-to-use adapter weights for inference
- **Hugging Face Integration**: Pushes trained adapters to Hugging Face Hub

## 🛠️ Technology Stack

- **Base Model**: Qwen/Qwen2-0.5B-Instruct
- **Fine-tuning Method**: QLoRa (Quantized Low-Rank Adaptation)
- **Framework**: Transformers + PEFT + TRL
- **Quantization**: 4-bit quantization using BitsAndBytesConfig
- **Training**: Supervised Fine-Tuning (SFT)
- **Dataset**: ruslanmv/ai-medical-chatbot

## 📋 Requirements

Install the required dependencies:

```bash
pip install torch transformers datasets peft trl bitsandbytes accelerate
```

Or use the provided requirements.txt:

```bash
pip install -r requirements.txt
```

## 🔧 Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (minimum 8GB VRAM recommended)
- **RAM**: Sufficient system memory for dataset processing
- **Storage**: ~5GB for model weights and training artifacts

## 📁 Project Structure

```
.
├── finetuning_with_ruslanmv.ipynb    # Main fine-tuning notebook
├── requirements.txt                   # Python dependencies
├── README.md                         # Project documentation
└── qwen2-medical-adapter/            # Generated LoRA adapter weights
```

## 🚀 Quick Start

### 1. Environment Setup

Ensure you have CUDA available:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### 2. Model Configuration

The project uses QLoRa configuration:
- **Quantization**: 4-bit (nf4)
- **Double Quantization**: Enabled
- **Compute Dtype**: bfloat16

### 3. Dataset Preparation

The model is fine-tuned on the `ruslanmv/ai-medical-chatbot` dataset:
- **Training Split**: 900 samples
- **Evaluation Split**: 100 samples
- **Format**: Chat template with system, user, and assistant messages

### 4. Training Configuration

Key training parameters:
- **Epochs**: 3
- **Batch Size**: 4 (per device)
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 2e-4
- **Optimizer**: paged_adamw_8bit (QLoRa-specific)
- **Scheduler**: Cosine with warmup ratio of 0.1

### 5. LoRA Configuration

```python
peft_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Alpha parameter
    lora_dropout=0.05,       # Dropout rate
    bias="none",             # Bias training
    task_type="CAUSAL_LM",   # Task type
    target_modules="all-linear"  # Target modules
)
```

## 📖 Usage

### Training

Run the complete notebook to fine-tune the model:

```bash
jupyter notebook finetuning_with_ruslanmv.ipynb
```

The training process includes:
1. GPU setup and verification
2. Model and tokenizer loading with 4-bit quantization
3. Dataset loading and preprocessing
4. Training configuration setup
5. Model fine-tuning with SFTTrainer
6. Model evaluation and testing
7. Saving LoRA adapter weights

### Inference

Load the fine-tuned model for inference:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load base model with quantization
bnb_config_inf = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    quantization_config=bnb_config_inf,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./qwen2-medical-adapter")
model = model.merge_and_unload()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Generate response
messages = [
    {"role": "system", "content": "You are a helpful AI medical assistant. Provide safe and informative advice. Do not diagnose. Always recommend consulting a medical professional."},
    {"role": "user", "content": "Hello, what is the capital of bangladesh?"}
]

prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7
)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### Loading from Hugging Face Hub

Use the pre-trained adapter from Hugging Face:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = PeftModel.from_pretrained(model, "fahad1770/qwen2-medical-lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
```

## ⚠️ Important Notes

### Medical Disclaimer

This model is designed for educational purposes and general medical information. **Important**:
- The model should NOT be used for actual medical diagnosis
- Always recommend consulting with qualified medical professionals
- The model provides general information only
- Medical decisions should never be based solely on AI responses

### Model Limitations

- **Knowledge Cutoff**: Limited to training data
- **Bias**: May reflect biases present in training data
- **Accuracy**: Not 100% reliable for medical information
- **Context**: Limited context window for conversations

## 📊 Training Results

The model is configured to save:
- **Checkpoints**: Every 50 steps (maximum 3 saved)
- **Best Model**: Loaded automatically at end of training
- **TensorBoard Logs**: Available for monitoring training progress
- **Final Adapter**: Saved to `./qwen2-medical-adapter`

## 🔗 Resources

- **Base Model**: [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
- **Dataset**: [ruslanmv/ai-medical-chatbot](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot)
- **QLoRa Paper**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **PEFT Library**: [Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project follows the same license as the base Qwen2 model. Please check the Hugging Face model card for specific licensing terms.

## 🙏 Acknowledgments

- **Qwen Team**: For the excellent base model
- **Hugging Face**: For the transformers and PEFT libraries
- **Dataset Creator**: ruslanmv for the medical chatbot dataset
- **Research Community**: For QLoRa and efficient fine-tuning techniques

---

**Note**: This is a research and educational project. The model should be used responsibly and with appropriate medical disclaimers.