import json
import uuid
from datetime import datetime
from typing import List, Dict, Any
from app.services.memory_service import memory_service
from app.config import settings
from app.core import genai, pinecone

class RAGService:
    def __init__(self):
        self.conversation_history = {}  # user_id별 대화 기록
        self.base_system_instruction = settings.SYSTEM_INSTRUCTION

    async def get_enhanced_system_instruction(self, user_id: str, query: str = None) -> str:
        """사용자별 관련 기억을 검색하여 강화된 시스템 인스트럭션을 생성합니다."""
        try:
            # 최근 대화 내용을 기반으로 검색 쿼리 생성
            search_query = query or self._get_recent_conversation_context(user_id)

            if search_query:
                # 관련 기억 검색
                retrieved_memories = memory_service.retrieve_memories(search_query, top_k=5)

                if retrieved_memories:
                    # 기억 정보를 시스템 인스트럭션에 추가
                    memory_context = self._format_memories_for_context(retrieved_memories)
                    enhanced_instruction = f"""{self.base_system_instruction}

다음은 사용자와 관련된 중요한 기억들입니다. 이 정보를 참고하여 더 개인화된 대화를 나누세요:

{memory_context}

이 기억들을 자연스럽게 대화에 활용하되, 직접적으로 언급하지 말고 맥락에 맞게 사용하세요."""
                    return enhanced_instruction

            return self.base_system_instruction

        except Exception as e:
            print(f"Error generating enhanced system instruction: {e}")
            return self.base_system_instruction

    def _get_recent_conversation_context(self, user_id: str, last_n: int = 3) -> str:
        """최근 대화 내용을 가져와서 검색 쿼리로 사용합니다."""
        if user_id not in self.conversation_history:
            return ""

        recent_messages = self.conversation_history[user_id][-last_n:]
        return " ".join([msg["content"] for msg in recent_messages if msg["role"] == "user"])

    def _format_memories_for_context(self, memories: List) -> str:
        """검색된 기억들을 컨텍스트 형태로 포맷팅합니다."""
        formatted_memories = []
        for memory in memories:
            metadata = memory.metadata
            score = memory.score

            # 스코어가 임계값 이상인 경우만 포함
            if score > 0.7:
                memory_text = f"- {metadata.get('content', '')}"
                if metadata.get('category'):
                    memory_text += f" (카테고리: {metadata['category']})"
                if metadata.get('date'):
                    memory_text += f" (날짜: {metadata['date']})"
                formatted_memories.append(memory_text)

        return "\n".join(formatted_memories)

    async def save_conversation_turn(self, user_id: str, role: str, content: str):
        """대화 턴을 저장합니다."""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []

        conversation_turn = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        self.conversation_history[user_id].append(conversation_turn)

        # 대화 기록이 너무 길어지면 오래된 것 제거
        if len(self.conversation_history[user_id]) > 50:
            self.conversation_history[user_id] = self.conversation_history[user_id][-30:]

    async def process_user_message(self, user_id: str, message: str) -> str:
        """사용자 메시지를 처리하고 RAG 기반 응답을 생성합니다."""
        # 사용자 메시지 저장
        await self.save_conversation_turn(user_id, "user", message)

        # 관련 기억 검색
        retrieved_memories = memory_service.retrieve_memories(message, top_k=3)

        # 컨텍스트가 포함된 시스템 인스트럭션 생성
        enhanced_instruction = await self.get_enhanced_system_instruction(user_id, message)

        return enhanced_instruction

    async def add_new_memory(self, user_id: str, content: str, metadata: Dict[str, Any]):
        """새로운 기억을 추가합니다."""
        memory_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        # 임베딩 생성 및 Pinecone에 저장
        embedding = memory_service.get_embedding(content)

        vector = {
            "id": memory_id,
            "values": embedding,
            "metadata": {
                **metadata,
                "user_id": user_id,
                "content": content,
                "created_at": datetime.now().isoformat()
            }
        }

        index = pinecone.Index(memory_service.index_name)
        index.upsert(vectors=[vector])

        print(f"New memory added for user {user_id}: {memory_id}")
        return memory_id

    async def search_memories_for_function_call(self, user_id: str, query: str, top_k: int = 3) -> List[Dict]:
        """Function call에서 사용할 기억 검색 (사용자별 필터링 포함)"""
        memories = memory_service.retrieve_memories(query, top_k * 2)  # 더 많이 검색 후 필터링

        # 사용자별 필터링
        user_memories = []
        for memory in memories:
            if memory.metadata.get('user_id') == user_id or not memory.metadata.get('user_id'):
                user_memories.append({
                    "content": memory.metadata.get('content', ''),
                    "category": memory.metadata.get('category', ''),
                    "score": memory.score,
                    "date": memory.metadata.get('date', '')
                })

                if len(user_memories) >= top_k:
                    break

        return user_memories

rag_service = RAGService()
