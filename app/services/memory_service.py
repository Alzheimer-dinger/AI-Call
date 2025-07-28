import json
import time
from typing import List
from app.core import pinecone, genai
from app.config import settings
from app.models import MemoryItem, RetrievedMemory
from pinecone import ServerlessSpec

class MemoryService:
    def __init__(self):
        self.index_name = settings.PINECONE_INDEX_NAME

    def get_embedding(self, text: str) -> List[float]:
        """주어진 텍스트를 Gemini API를 사용하여 임베딩합니다."""
        return genai.embed_content(model=settings.EMBEDDING_MODEL, content=text)["embedding"]

    def setup_pinecone(self) -> None:
        """Pinecone 인덱스를 확인하고, 없으면 생성한 후 데이터를 저장합니다."""
        if self.index_name not in pinecone.list_indexes().names():
            print(f"Creating Pinecone index: {self.index_name}")
            pinecone.create_index(
                name=self.index_name,
                dimension=settings.PINECONE_DIMENSION,
                metric=settings.PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION
                )
            )
            time.sleep(1)
            print("Index created. Now uploading data...")
            self.upload_memories()
        else:
            print(f"Pinecone index '{self.index_name}' already exists.")

    def upload_memories(self) -> None:
        """JSON 파일에서 기억 데이터를 읽어 Pinecone에 업로드합니다."""
        index = pinecone.Index(self.index_name)

        try:
            with open(settings.MEMORIES_FILE_PATH, "r", encoding="utf-8") as f:
                memories_data = json.load(f)
        except FileNotFoundError:
            print(f"Memories file not found: {settings.MEMORIES_FILE_PATH}")
            return

        vectors_to_upsert = []
        for item in memories_data:
            memory_item = MemoryItem(**item)
            embedding = self.get_embedding(memory_item.content)
            vector = {
                "id": memory_item.id,
                "values": embedding,
                "metadata": memory_item.metadata
            }
            vectors_to_upsert.append(vector)

        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
        print(f"Uploaded {len(vectors_to_upsert)} vectors to Pinecone.")

    def retrieve_memories(self, query: str, top_k: int = 3) -> List[RetrievedMemory]:
        """주어진 쿼리와 가장 유사한 기억을 Pinecone에서 검색합니다."""
        index = pinecone.Index(self.index_name)
        query_embedding = self.get_embedding(query)

        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        retrieved_memories = []
        if results.get("matches"):
            for match in results["matches"]:
                retrieved_memories.append(RetrievedMemory(
                    score=match.get("score", 0.0),
                    metadata=match.get("metadata", {})
                ))

        return retrieved_memories

memory_service = MemoryService()

