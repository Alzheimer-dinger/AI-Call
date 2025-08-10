import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

# 'speaker' 필드를 위한 열거형(Enum) 정의
class SpeakerEnum(str, Enum):
    """대화의 화자를 나타내는 열거형"""
    PATIENT = 'patient'
    AI = 'ai'

# 대화의 각 턴을 나타내는 모델
class ConversationTurn(BaseModel):
    """하나의 대화 턴을 나타내는 모델"""
    speaker: SpeakerEnum = Field(..., description="화자 (patient 또는 ai)")
    content: str = Field(..., description="대화 내용")

# 전체 대화 기록을 나타내는 메인 모델
class ConversationLog(BaseModel):
    """
    하나의 대화 세션 전체 기록을 나타내는 Pydantic 모델
    """
    session_id: uuid.UUID = Field(uuid.uuid1(), description="세션의 고유 식별자 (UUID)")
    user_id: str = Field(..., description="사용자의 고유 식별자 (UUID)")
    start_time: datetime = Field(..., description="대화 시작 시간 (ISO 8601 형식)")
    end_time: datetime = Field(..., description="대화 종료 시간 (ISO 8601 형식)")
    conversation: List[ConversationTurn] = Field(..., description="전체 대화 내용 리스트")
    audio_recording_url: Optional[str] = Field(None, description="음성 녹음 파일 URL (GCS)")