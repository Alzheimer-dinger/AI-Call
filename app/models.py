from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class UserInfo(BaseModel):
    username: str

class MemoryItem(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]

class RetrievedMemory(BaseModel):
    score: float
    metadata: Dict[str, Any]

class WebSocketMessage(BaseModel):
    type: str
    data: Optional[bytes] = None
    error: Optional[str] = None

class AudioConfig(BaseModel):
    sample_rate: int
    channels: int = 1
    format: str = "pcm"

