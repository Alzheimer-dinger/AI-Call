import os
from typing import Optional

class Settings:
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "a_very_secret_key_for_jwt_token")

    # JWT Settings
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Pinecone Settings
    PINECONE_INDEX_NAME: str = "alzheimer-call"
    PINECONE_DIMENSION: int = 768
    PINECONE_METRIC: str = "cosine"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # Gemini Settings
    LIVE_MODEL: str = "gemini-2.5-flash-preview-native-audio-dialog"
    EMBEDDING_MODEL: str = "models/embedding-001"
    TARGET_SAMPLE_RATE: int = 16000

    # System Instructions
    SYSTEM_INSTRUCTION: str = (
        "당신은 노인과 대화하는 따뜻하고 친절한 AI 에이전트입니다. "
        "사용자의 말을 다정하게 들어주고, 자연스럽게 대화를 이어나가세요."
    )

    # Memory Settings
    MEMORIES_FILE_PATH: str = "data/memories.json"

settings = Settings()

