import google.generativeai as genai
from pinecone import Pinecone
from app.config import settings

# Google Gemini API 설정
genai.configure(api_key=settings.GOOGLE_API_KEY)

# Pinecone 클라이언트 설정
pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)

# 임베딩 모델 설정
embedding_model = settings.EMBEDDING_MODEL

# 인덱스 이름
PINECONE_INDEX_NAME = settings.PINECONE_INDEX_NAME
