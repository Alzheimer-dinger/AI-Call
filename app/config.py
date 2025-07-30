import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Google API 설정
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # Pinecone 설정
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "alzheimer-memories")
    PINECONE_DIMENSION: int = 768
    PINECONE_METRIC: str = "cosine"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # 모델 설정
    EMBEDDING_MODEL: str = "models/embedding-001"
    LIVE_MODEL: str = "models/gemini-2.0-flash-exp"
    
    # 데이터 파일 경로
    MEMORIES_FILE_PATH: str = "data/memories.json"

    # 오디오 설정
    TARGET_SAMPLE_RATE: int = 24000

    # 시스템 인스트럭션
    SYSTEM_INSTRUCTION: str = """당신은 알츠하이머 환자를 위한 친근하고 따뜻한 AI 어시스턴트입니다.

주요 역할:
1. 환자와 자연스럽고 편안한 대화를 나누세요
2. 환자의 기억을 도와주고, 중요한 정보를 기억해주세요
3. 반복적인 질문에도 인내심을 가지고 친절하게 답해주세요
4. 환자가 혼란스러워할 때 차분하게 안내해주세요

대화 스타일:
- 따뜻하고 친근한 말투를 사용하세요
- 간단하고 명확한 문장으로 대화해주세요
- 환자의 감정을 이해하고 공감해주세요
- 필요시 천천히, 반복해서 설명해주세요"""

    class Config:
        env_file = ".env"

settings = Settings()
