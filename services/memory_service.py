import json
import time
import uuid
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import os
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

@dataclass
class MemorySearchResult:
    score: float
    metadata: Dict

class MemoryService:
    def __init__(self):
        # Pinecone 설정
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "alzheimer-memories")
        self.dimension = int(os.getenv("PINECONE_DIMENSION", "1024"))  # multilingual-e5-large dimension
        self.metric = os.getenv("PINECONE_METRIC", "cosine")
        self.cloud = os.getenv("PINECONE_CLOUD", "aws")
        self.region = os.getenv("PINECONE_REGION", "us-east-1")
        
        # Pinecone 임베딩 모델 설정 (multilingual-e5-large 사용)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "multilingual-e5-large")
        
        # Pinecone 클라이언트 초기화
        if self.pinecone_api_key:
            self.pinecone = Pinecone(api_key=self.pinecone_api_key)
        else:
            logger.warning("PINECONE_API_KEY not found. Memory functions will be disabled.")
            self.pinecone = None

    def get_embedding(self, text: str) -> List[float]:
        """주어진 텍스트를 Pinecone inference API를 사용하여 임베딩합니다."""
        if not self.pinecone:
            logger.warning("Pinecone client not initialized")
            return []
            
        try:
            # Pinecone inference API 사용
            result = self.pinecone.inference.embed(
                model=self.embedding_model,
                inputs=[text],
                parameters={"input_type": "query", "truncate": "END"}
            )
            
            # EmbeddingsList 객체 처리
            if hasattr(result, '__iter__') and len(result) > 0:
                first_embedding = result[0]
                
                # DenseEmbedding 객체에서 값 추출
                if hasattr(first_embedding, 'to_dict'):
                    embedding_dict = first_embedding.to_dict()
                    if 'values' in embedding_dict:
                        return embedding_dict['values']
                    elif 'embedding' in embedding_dict:
                        return embedding_dict['embedding']
                
                # 직접 속성 접근 시도
                if hasattr(first_embedding, 'values'):
                    return first_embedding.values
                elif hasattr(first_embedding, 'embedding'):
                    return first_embedding.embedding
            
            logger.debug("No embeddings found in result")
            return []
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    def setup_pinecone(self) -> None:
        """Pinecone 인덱스를 확인하고, 없으면 생성합니다."""
        if not self.pinecone:
            return
            
        try:
            if self.index_name not in self.pinecone.list_indexes().names():
                logger.info(f"Creating Pinecone index: {self.index_name}")
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
                logger.info("Index created successfully.")
            else:
                logger.info(f"Pinecone index '{self.index_name}' already exists.")
        except Exception as e:
            logger.error(f"Error setting up Pinecone: {e}")

    def retrieve_memories(self, query: str, top_k: int = 3, user_id: str = None) -> List[MemorySearchResult]:
        """주어진 쿼리와 가장 유사한 기억을 Pinecone에서 검색합니다."""
        logger.debug(f"retrieve_memories called with query='{query}', user_id='{user_id}'")
        
        if not self.pinecone:
            logger.debug("Pinecone client not initialized")
            return []
            
        try:
            index = self.pinecone.Index(self.index_name)
            logger.debug(f"Getting embedding for query: '{query}'")
            query_embedding = self.get_embedding(query)
            
            if not query_embedding:
                logger.debug("Failed to get embedding")
                return []
                
            logger.debug(f"Got embedding with length: {len(query_embedding)}")

            if user_id:
                logger.debug(f"Searching with user_id filter: {user_id}")
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter={"user_id": {"$eq": user_id}}
                )
            else:
                logger.debug("Searching without user_id filter")
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )

            logger.debug(f"Pinecone query returned: {results}")
            
            retrieved_memories = []
            if results.get("matches"):
                logger.debug(f"Found {len(results['matches'])} matches")
                for match in results["matches"]:
                    retrieved_memories.append(MemorySearchResult(
                        score=match.get("score", 0.0),
                        metadata=match.get("metadata", {})
                    ))
            else:
                logger.debug("No matches found in Pinecone results")

            return retrieved_memories
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
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
            logger.info(f"Memory added for user {user_id}: {memory_id}")
            
            return memory_id
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return ""

# 전역 인스턴스 생성
memory_service = MemoryService()