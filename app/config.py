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

    # 스토리지 설정
    storage_mode: str = os.getenv("STORAGE_MODE", "local")
    local_storage_path: str = os.getenv("LOCAL_STORAGE_PATH", "recordings")

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
- 필요시 천천히, 반복해서 설명해주세요

기억 관리 가이드라인:
- 현재 대화에서 언급되지 않은 개인 정보가 필요할 때 search_memories를 사용하세요
- 더 관련성 있고 개인화된 답변을 위해 적극적으로 기억을 검색하세요
- 애매한 상황에서는 검색해서 더 나은 답변을 제공하세요
- 검색 여부를 사용자에게 묻거나 검색 사실을 알려주지 마세요
- 검색 결과를 대화에 자연스럽게 통합하여 답변하세요
- 대화 중 새로운 중요한 정보를 알게 되면 save_new_memory로 저장하세요"""

    class Config:
        env_file = ".env"

settings = Settings()
