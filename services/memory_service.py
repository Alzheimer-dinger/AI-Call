import json
import time
import uuid
from typing import List, Dict, Any
from dataclasses import dataclass
from google import genai
import os
from pinecone import Pinecone, ServerlessSpec

@dataclass
class MemorySearchResult:
    score: float
    metadata: Dict

class MemoryService:
    def __init__(self):
        # Pinecone 설정
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "alzheimer-memories")
        self.dimension = int(os.getenv("PINECONE_DIMENSION", "768"))
        self.metric = os.getenv("PINECONE_METRIC", "cosine")
        self.cloud = os.getenv("PINECONE_CLOUD", "aws")
        self.region = os.getenv("PINECONE_REGION", "us-east-1")
        
        # 임베딩 모델 설정
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "models/text-embedding-001")
        
        # Google Genai 클라이언트 설정
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            self.genai_client = genai.Client(vertexai=False, api_key=gemini_api_key)
        else:
            print("Warning: GEMINI_API_KEY not found. Embedding functions will be disabled.")
            self.genai_client = None
        
        # Pinecone 클라이언트 초기화
        if self.pinecone_api_key:
            self.pinecone = Pinecone(api_key=self.pinecone_api_key)
        else:
            print("Warning: PINECONE_API_KEY not found. Memory functions will be disabled.")
            self.pinecone = None

    def get_embedding(self, text: str) -> List[float]:
        """주어진 텍스트를 Gemini API를 사용하여 임베딩합니다."""
        if not self.genai_client:
            print("Genai client not initialized")
            return []
            
        try:
            result = self.genai_client.models.embed_content(model=self.embedding_model, contents=text)
            if result.embeddings and len(result.embeddings) > 0:
                return result.embeddings[0].values
            else:
                print("[DEBUG] No embeddings found in result")
                return []
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def setup_pinecone(self) -> None:
        """Pinecone 인덱스를 확인하고, 없으면 생성합니다."""
        if not self.pinecone:
            return
            
        try:
            if self.index_name not in self.pinecone.list_indexes().names():
                print(f"Creating Pinecone index: {self.index_name}")
                self.pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region
                    )
                )
                time.sleep(2)
                print("Index created successfully.")
            else:
                print(f"Pinecone index '{self.index_name}' already exists.")
        except Exception as e:
            print(f"Error setting up Pinecone: {e}")

    def retrieve_memories(self, query: str, top_k: int = 3, user_id: str = None) -> List[MemorySearchResult]:
        """주어진 쿼리와 가장 유사한 기억을 Pinecone에서 검색합니다."""
        print(f"[DEBUG] retrieve_memories called with query='{query}', user_id='{user_id}'")
        
        if not self.pinecone:
            print("[DEBUG] Pinecone client not initialized")
            return []
            
        try:
            index = self.pinecone.Index(self.index_name)
            print(f"[DEBUG] Getting embedding for query: '{query}'")
            query_embedding = self.get_embedding(query)
            
            if not query_embedding:
                print("[DEBUG] Failed to get embedding")
                return []
                
            print(f"[DEBUG] Got embedding with length: {len(query_embedding)}")

            if user_id:
                print(f"[DEBUG] Searching with user_id filter: {user_id}")
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter={"user_id": {"$eq": user_id}}
                )
            else:
                print("[DEBUG] Searching without user_id filter")
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )

            print(f"[DEBUG] Pinecone query returned: {results}")
            
            retrieved_memories = []
            if results.get("matches"):
                print(f"[DEBUG] Found {len(results['matches'])} matches")
                for match in results["matches"]:
                    retrieved_memories.append(MemorySearchResult(
                        score=match.get("score", 0.0),
                        metadata=match.get("metadata", {})
                    ))
            else:
                print("[DEBUG] No matches found in Pinecone results")

            return retrieved_memories
        except Exception as e:
            print(f"[DEBUG] Error retrieving memories: {e}")
            import traceback
            traceback.print_exc()
            return []

    def add_memory(self, user_id: str, content: str, metadata: Dict[str, Any]) -> str:
        """새로운 기억을 Pinecone에 추가합니다."""
        if not self.pinecone:
            return ""
            
        try:
            index = self.pinecone.Index(self.index_name)
            memory_id = str(uuid.uuid4())
            
            # 메타데이터에 user_id와 content 추가
            metadata["user_id"] = user_id
            metadata["content"] = content
            
            # 임베딩 생성
            embedding = self.get_embedding(content)
            if not embedding:
                return ""
            
            # 벡터 생성
            vector = {
                "id": memory_id,
                "values": embedding,
                "metadata": metadata
            }
            
            # Pinecone에 업서트
            index.upsert(vectors=[vector])
            print(f"Memory added for user {user_id}: {memory_id}")
            
            return memory_id
        except Exception as e:
            print(f"Error adding memory: {e}")
            return ""

# 전역 인스턴스 생성
memory_service = MemoryService()