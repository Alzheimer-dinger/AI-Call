import json
import time
from app.core import pinecone, genai, embedding_model, PINECONE_INDEX_NAME
from app.config import settings
from pinecone import ServerlessSpec

def get_embedding(text: str):
    """주어진 텍스트를 Gemini API를 사용하여 임베딩합니다."""
    return genai.embed_content(model=embedding_model, content=text)["embedding"]

def setup_pinecone():
    """Pinecone 인덱스를 확인하고, 없으면 생성한 후 데이터를 저장합니다."""
    if PINECONE_INDEX_NAME not in pinecone.list_indexes().names():
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=settings.PINECONE_DIMENSION,
            metric=settings.PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION
            )
        )
        # 인덱스가 준비될 때까지 잠시 대기
        time.sleep(1)
        print("Index created. Now uploading data...")
        upload_memories()
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

def upload_memories():
    """JSON 파일에서 기억 데이터를 읽어 Pinecone에 업로드합니다."""
    index = pinecone.Index(PINECONE_INDEX_NAME)
    try:
        with open(settings.MEMORIES_FILE_PATH, "r", encoding="utf-8") as f:
            memories = json.load(f)
    except FileNotFoundError:
        print(f"Memories file not found: {settings.MEMORIES_FILE_PATH}")
        return

    vectors_to_upsert = []
    for item in memories:
        embedding = get_embedding(item["content"])
        vector = {
            "id": item["id"],
            "values": embedding,
            "metadata": item["metadata"]
        }
        vectors_to_upsert.append(vector)
    
    # 배치로 업로드하여 효율성 증대
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)
    print(f"Uploaded {len(vectors_to_upsert)} vectors to Pinecone.")

def retrieve_memories(query: str, top_k: int = 3):
    """주어진 쿼리와 가장 유사한 기억을 Pinecone에서 검색합니다."""
    index = pinecone.Index(PINECONE_INDEX_NAME)
    query_embedding = get_embedding(query)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # 검색된 결과에서 content와 metadata를 추출하여 반환
    retrieved_memories = []
    if results.get("matches"):
        for match in results["matches"]:
            retrieved_memories.append({
                "score": match.get("score"),
                "metadata": match.get("metadata")
            })
            
    return retrieved_memories
