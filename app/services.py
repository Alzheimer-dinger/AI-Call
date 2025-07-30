import json
import time
from app.core import pinecone, PINECONE_INDEX_NAME
from app.services.memory_service import memory_service
from pinecone import ServerlessSpec

def setup_pinecone():
    """Pinecone 인덱스를 확인하고, 없으면 생성한 후 데이터를 저장합니다."""
    if PINECONE_INDEX_NAME not in pinecone.list_indexes().names():
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,  # Gemini embedding-001
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # 인덱스가 준비될 때까지 잠시 대기
        time.sleep(5)
        print("Index created. Now uploading initial data...")
        upload_initial_memories()
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

def upload_initial_memories():
    """초기 기억 데이터를 업로드합니다."""
    try:
        # memories.json 파일이 있다면 로드
        with open("data/memories.json", "r", encoding="utf-8") as f:
            memories = json.load(f)

        for item in memories:
            memory_service.add_memory(
                user_id=item.get("user_id", "system"),
                content=item["content"],
                metadata=item.get("metadata", {})
            )
        print(f"Uploaded {len(memories)} initial memories to Pinecone.")

    except FileNotFoundError:
        print("No initial memories file found. Skipping initial data upload.")
    except Exception as e:
        print(f"Error uploading initial memories: {e}")

# 기존 함수들은 memory_service로 이관되었으므로 제거하거나 래퍼로 사용
def get_embedding(text: str):
    """래퍼 함수 - memory_service 사용 권장"""
    return memory_service.get_embedding(text)

def retrieve_memories(query: str, top_k: int = 3):
    """래퍼 함수 - memory_service 사용 권장"""
    return memory_service.retrieve_memories(query, top_k)
