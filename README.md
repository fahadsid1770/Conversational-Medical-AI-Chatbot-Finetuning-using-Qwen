# Medical Conversational AI Model Development with Continuous Learning (CLHF)

This project features a high-performance medical assistant based on the Qwen2-0.5B-Instruct architecture. It is optimized via 4-bit QLoRA and implements a sophisticated continuous learning pipeline that refines model behavior through real-time human preference data.

## Project Overview

The core objective of this system is to bridge the gap between static fine-tuning and dynamic user alignment. By integrating a feedback-driven optimization loop, the model can adapt to specific medical consultation nuances based on direct user interactions (Like/Dislike).

## Core Technical Components

### 1. Initial Supervised Fine-Tuning (SFT)
The base model is initially fine-tuned on specialized medical datasets using Quantized Low-Rank Adaptation (QLoRA). This ensures the model acquires essential medical knowledge and conversational structure while maintaining a minimal memory footprint, suitable for deployment on consumer-grade hardware.

### 2. High-Efficiency Inference Pipeline
The inference engine is built on FastAPI and utilizes 4-bit quantization for rapid response generation. It supports multi-turn conversations and maintains context through a structured chat template system.

### 3. Continuous Learning via KTO
Unlike traditional static models, this system captures every user interaction. When a user provides feedback (Like/Dislike) in the interface, the data is stored in a local SQLite database and periodically used to re-align the model's weights using Kahneman-Tversky Optimization (KTO).

---

## Technical Analysis: KTO vs. DPO

A critical architectural decision in this project was the selection of Kahneman-Tversky Optimization (KTO) over Direct Preference Optimization (DPO).

### The Challenge of Real-World Feedback
Standard preference optimization (DPO) requires paired data: for every prompt, the system must provide a "chosen" response and a "rejected" response simultaneously. In a live production environment, this is rarely feasible as users typically interact with a single generated output.

### The KTO Solution
KTO is mathematically designed to handle **unpaired preference data**. It treats each interaction as a binary signal of desirability.

| Feature | Direct Preference Optimization (DPO) | Kahneman-Tversky Optimization (KTO) |
| :--- | :--- | :--- |
| **Data Requirement** | Paired (Chosen + Rejected) | Unpaired (Single label: Good or Bad) |
| **UX Compatibility** | Requires A/B testing or ranking | Fits standard Like/Dislike interaction |
| **Deployment Fit** | Best for offline datasets | Best for continuous online learning |
| **Robustness** | Sensitive to the quality of the "rejected" pair | Highly robust to noisy, real-world signals |

**Conclusion**: KTO allows this system to learn from every single click without the logistical overhead of generating synthetic pairs or forcing users into complex ranking tasks. This makes the learning loop truly continuous and aligned with natural human-computer interaction.

---

## Project Structure

```text
app/
├── api/            # RESTful endpoints for Chat, Feedback, and Training
├── core/           # Configuration, Model Management, and Database Logic
├── schemas/        # Pydantic data validation models
└── training/       # Implementation of SFT and KTO training loops
```

## System Requirements and Deployment

### Installation
```bash
pip install -r requirements.txt
```

### Execution
To initialize the API and the automated database setup, execute:
```bash
python -m app.main
```

### Operational Workflow
1. **Interaction**: The user queries the model via the `/v1/chat` endpoint.
2. **Feedback**: The interface submits user sentiment to `/v1/feedback`.
3. **Optimization**: The system administrator triggers the continuous learning loop via `/v1/train/continuous/start`, which processes pending feedback and updates the active model weights without downtime.

## Medical Disclaimer

This software is developed for research and educational purposes. It is not intended for clinical use, medical diagnosis, or as a substitute for professional medical advice. Users should always consult with qualified healthcare professionals for medical concerns.

---
*Developed for the advancement of interactive medical AI systems.*
