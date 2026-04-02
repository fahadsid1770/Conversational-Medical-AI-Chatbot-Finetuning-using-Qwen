from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.schemas.chat import (
    ChatRequest, ChatResponse, TrainRequest, 
    TrainStatusResponse, ModelInfoResponse, FeedbackRequest
)
from app.core.model_manager import model_manager
from app.training.trainer import run_training, training_status
from app.core.config import settings
from app.core.db import insert_feedback
from app.training.continuous_trainer import run_continuous_training
import os

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for medical queries."""
    history = None
    if request.history:
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]
    
    response = model_manager.generate_response(
        user_query=request.query,
        history=history,
        system_prompt=request.system_prompt
    )
    
    return ChatResponse(
        response=response,
        model=settings.MODEL_ID
    )

@router.post("/feedback")
async def receive_feedback(request: FeedbackRequest):
    """Receive user feedback for model responses."""
    # We store the raw prompt. The trainer will apply the template if needed.
    # This is more robust than storing the templated prompt.
    success = insert_feedback(
        prompt=request.prompt,
        model_response=request.model_response,
        is_liked=request.is_liked
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store feedback.")
    return {"message": "Feedback recorded successfully."}

@router.post("/train/continuous/start")
async def start_continuous_training(background_tasks: BackgroundTasks):
    """Start the continuous training loop (KTO) in the background."""
    if training_status["status"] == "training":
        raise HTTPException(status_code=400, detail="Training is already in progress.")
    
    background_tasks.add_task(run_continuous_training)
    return {"message": "Continuous training triggered in background."}

@router.post("/train/start")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start the fine-tuning process as a background task."""
    if training_status["status"] == "training":
        raise HTTPException(status_code=400, detail="Training is already in progress.")
    
    params = request.dict(exclude_none=True)
    background_tasks.add_task(run_training, params)
    
    return {"message": "Training started in background."}

@router.get("/train/status", response_model=TrainStatusResponse)
async def get_train_status():
    """Get the current training status."""
    return TrainStatusResponse(**training_status)

@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the currently loaded model."""
    adapter_path = settings.ADAPTER_PATH if os.path.exists(settings.ADAPTER_PATH) else None
    return ModelInfoResponse(
        model_id=settings.MODEL_ID,
        active_adapter=adapter_path,
        device=str(model_manager.model.device) if model_manager.model else settings.DEVICE,
        is_loaded=model_manager.model is not None
    )

@router.post("/model/reload")
async def reload_model():
    """Reload the model and adapter."""
    success = model_manager.load_model(
        settings.ADAPTER_PATH if os.path.exists(settings.ADAPTER_PATH) else None
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reload model.")
    return {"message": "Model reloaded successfully."}
