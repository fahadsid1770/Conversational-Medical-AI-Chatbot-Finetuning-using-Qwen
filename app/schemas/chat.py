from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    query: str = Field(..., description="The patient's medical query")
    history: Optional[List[ChatMessage]] = Field(None, description="Previous chat history")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The AI assistant's response")
    model: str = Field(..., description="The model used for generation")

class TrainRequest(BaseModel):
    dataset_name: Optional[str] = Field(None, description="HuggingFace dataset name")
    num_epochs: Optional[int] = Field(None, description="Number of training epochs")
    learning_rate: Optional[float] = Field(None, description="Learning rate")

class TrainStatusResponse(BaseModel):
    status: str
    last_loss: Optional[float] = None
    current_step: int
    total_steps: int
    error: Optional[str] = None

class ModelInfoResponse(BaseModel):
    model_id: str
    active_adapter: Optional[str]
    device: str
    is_loaded: bool

class FeedbackRequest(BaseModel):
    prompt: str = Field(..., description="The user's original query or prompt")
    model_response: str = Field(..., description="The model's generated response")
    is_liked: bool = Field(..., description="True if liked/chosen, False if disliked/rejected")

