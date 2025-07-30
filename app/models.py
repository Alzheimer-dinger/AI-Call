from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime

class UserInfo(BaseModel):
    username: str
    full_name: Optional[str] = None
    email: Optional[str] = None

class Memory(BaseModel):
    id: str
    content: str
    category: str
    importance: str = "medium"
    user_id: str
    created_at: datetime
    metadata: Dict[str, Any] = {}

class ConversationTurn(BaseModel):
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

class MemorySearchResult(BaseModel):
    score: float
    metadata: Dict[str, Any]

class FunctionCallRequest(BaseModel):
    name: str
    args: Dict[str, Any]

class AddMemoryRequest(BaseModel):
    content: str
    category: str
    importance: str = "medium"
    metadata: Dict[str, Any] = {}

