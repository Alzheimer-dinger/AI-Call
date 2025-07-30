import json
import uuid
from datetime import datetime
from typing import List, Dict, Any
from app.core import pinecone
from app.services.memory_service import memory_service
from app.config import settings

class RAGService:
    def __init__(self):
        print("RAGService instance initialized.")
        self.conversation_history = {}  # user_id별 대화 기록
        self.base_system_instruction = f"""{settings.SYSTEM_INSTRUCTION}

추가 기능:
- 대화 중에 중요한 정보를 발견하면 search_memories 함수를 사용해서 관련 기억을 찾아보세요
- ���용자가 새로운 중요한 정보를 말하면 save_new_memory 함수를 사용해서 저장하세요
- 기억 검색 결과를 자연스럽게 대화에 녹여서 활용하세요"""

    async def get_enhanced_system_instruction(self, user_id: str, query: str = None) -> str:
        """사용자별 관련 기억을 검색하여 강화된 시스템 인스트럭션을 생성합니다."""
        try:
            # 최근 대화 내용을 기반으로 검색 쿼리 생성
            search_query = query or self._get_recent_conversation_context(user_id)

            if search_query:
                # 사용자별 관련 기억 검색
                retrieved_memories = memory_service.retrieve_memories(
                    search_query, 
                    top_k=5, 
                    user_id=user_id
                )

                if retrieved_memories:
                    # 기억 정보를 시스템 인��트럭션에 추가
                    memory_context = self._format_memories_for_context(retrieved_memories)
                    enhanced_instruction = f"""{self.base_system_instruction}

다음은 {user_id}님과 관련된 중요한 기억들입니다:

{memory_context}

이 기억들을 참고하여 더 개인화된 대화를 나누세요. 하지만 직접적으로 "기억에 따르면..."이라고 말하지 말고, 자연스럽게 대화에 ��용하세요."""
                    return enhanced_instruction

            return self.base_system_instruction

        except Exception as e:
            print(f"Error generating enhanced system instruction: {e}")
            return self.base_system_instruction

    def _get_recent_conversation_context(self, user_id: str, last_n: int = 3) -> str:
        """최근 대화 내용을 가져와서 검색 쿼리로 사용합니다."""
        if user_id not in self.conversation_history:
            # 기본 검색어 반환
            return "가족 취미 건강 일상"

        recent_messages = self.conversation_history[user_id][-last_n:]
        context = " ".join([msg["content"] for msg in recent_messages if msg["role"] == "user"])
        return context if context else "가족 취미 건강 일상"

    def _format_memories_for_context(self, memories: List) -> str:
        """검색된 기억들을 컨텍스트 형태로 포맷팅합니다."""
        formatted_memories = []
        for memory in memories:
            metadata = memory.metadata
            score = memory.score

            # 스코어가 임계값 이상인 경우만 포함 (관련성 높은 결과만)
            if score > 0.7:
                memory_text = f"- {metadata.get('content', '')}"
                if metadata.get('category'):
                    memory_text += f" (카테고리: {metadata['category']})"
                if metadata.get('date'):
                    memory_text += f" (날짜: {metadata['date']})"
                formatted_memories.append(memory_text)

        return "\n".join(formatted_memories) if formatted_memories else "저장된 기억이 없습니다."

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

        # 관련 기억 검색 (사용자별 필터링 포함)
        retrieved_memories = memory_service.retrieve_memories(message, top_k=3, user_id=user_id)

        # 컨텍스트가 포함된 시스템 인스트럭션 생성
        enhanced_instruction = await self.get_enhanced_system_instruction(user_id, message)

        return enhanced_instruction

    async def add_new_memory(self, user_id: str, content: str, metadata: Dict[str, Any]):
        """새로운 기억을 추가합니다."""
        try:
            index = pinecone.Index(settings.PINECONE_INDEX_NAME)
            memory_id = str(uuid.uuid4())
            embedding = memory_service.get_embedding(content)
            metadata["user_id"] = user_id
            metadata["content"] = content
            vector = {
                "id": memory_id,
                "values": embedding,
                "metadata": metadata
            }
            index.upsert(vectors=[vector])
            print(f"New memory added for user {user_id}: {memory_id}")
            return memory_id
        except Exception as e:
            print(f"Error adding new memory: {e}")
            raise

    async def search_memories_for_function_call(self, user_id: str, query: str, top_k: int = 3) -> List[Dict]:
        """Function call에서 사용할 기억 검색 (사용자별 필터링 포함)"""
        memories = memory_service.retrieve_memories(query, top_k, user_id)

        # 결과를 딕셔너리 형태로 변환
        user_memories = []
        for memory in memories:
            user_memories.append({
                "content": memory.metadata.get('content', ''),
                "category": memory.metadata.get('category', ''),
                "score": memory.score,
                "date": memory.metadata.get('date', ''),
                "importance": memory.metadata.get('importance', 'medium')
            })

        return user_memories

    async def add_sample_memories(self, user_id: str):
        """샘플 기억 데이터를 추가합니다."""
        sample_memories = [
            {"content": "손녀의 이름은 서연이고, 5살이다.", "category": "family", "date": "2023-05-12"},
            {"content": "매주 수요일 오전에 공원에서 산책하는 것을 좋아한다.", "category": "hobby", "date": "2024-01-10"},
            {"content": "어릴 적 고향은 부산이었고, 바다를 자주 보러 갔다.", "category": "life", "date": "1960-07-20"},
            {"content": "가장 좋아하는 음식은 된장찌개이다.", "category": "preference", "date": "2022-11-30"},
            {"content": "키우는 강아지 이름은 '바둑이'이고, 매일 저녁 6시에 밥을 줘야 한다.", "category": "pet", "date": "2023-08-01"}
        ]

        for memory in sample_memories:
            await self.add_new_memory(user_id, memory["content"], {"category": memory["category"], "date": memory["date"]})

        print(f"Sample memories added for user {user_id}")

# 전역 인스턴스 생성
rag_service = RAGService()
