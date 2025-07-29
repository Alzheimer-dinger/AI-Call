import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from app.config import settings
from app.models import UserInfo
from app.services.memory_service import memory_service
from app.services.rag_service import rag_service

class WebSocketService:
    def __init__(self):
        self.client = genai.Client()
        self.model = settings.LIVE_MODEL
        self.target_sample_rate = settings.TARGET_SAMPLE_RATE

    def _get_function_declarations(self):
        """Gemini가 사용��� 수 있는 function declarations를 정의합니다."""
        return [
            {
                "name": "search_memories",
                "description": "사용자와 관련된 기억을 검색합니다. 대화 중 관련 정보가 필요할 때 사용하세요.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "검색할 키워드나 문장"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "검색할 결과 개수 (기본값: 3)",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "save_new_memory",
                "description": "대화 중 새로운 중요한 정보를 기억으로 저장합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "저장할 기억 내용"
                        },
                        "category": {
                            "type": "string",
                            "description": "기억의 카테고리 (예: 가족, 취미, 건강, 일상 등)"
                        },
                        "importance": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "기억의 중요도"
                        }
                    },
                    "required": ["content", "category"]
                }
            }
        ]

    async def handle_live_audio_session(self, websocket: WebSocket, user: dict):
        """Gemini Live와의 오디오 세션을 처리합니다."""
        try:
            # 기본 시스템 인스트럭션 (function calling 안내 포함)
            system_instruction = f"""{settings.SYSTEM_INSTRUCTION}

대화 중 사용자와 관련된 정보가 필요하면 search_memories 함수를 사용하여 기억을 검색할 수 있습니다.
사용자가 새로운 중요한 정보를 말하면 save_new_memory 함수를 사용하여 저장할 수 있습니다.
함수 사용 시 사용자에게 "기억을 찾고 있어요" 같은 말은 하지 마세요. 자연스럽게 처리하세요."""

            config = {
                "response_modalities": ["AUDIO"],
                "system_instruction": system_instruction,
                "tools": [{"function_declarations": self._get_function_declarations()}]
            }

            async with self.client.aio.live.connect(model=self.model, config=config) as session:
                print(f"Persistent Gemini Live session established for {user['username']}")

                tasks = [
                    asyncio.create_task(self._forward_to_gemini(websocket, session, user)),
                    asyncio.create_task(self._forward_to_client(websocket, session, user))
                ]

                try:
                    await asyncio.gather(*tasks)
                except (WebSocketDisconnect, Exception) as e:
                    print(f"Task interrupted: {e}")
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            print(f"An unexpected error occurred in live session: {e}")
            raise

    async def _forward_to_gemini(self, websocket: WebSocket, session, user: dict):
        """클라이언트로부터 PCM 데이터를 받아 Gemini로 전송"""
        try:
            while True:
                pcm_data = await websocket.receive_bytes()
                print(f"Received {len(pcm_data)} bytes from client")

                blob = types.Blob(
                    data=pcm_data,
                    mime_type=f"audio/pcm;rate={self.target_sample_rate}"
                )
                await session.send_realtime_input(audio=blob)

        except WebSocketDisconnect:
            print("Client disconnected during audio forwarding.")
            raise
        except Exception as e:
            print(f"Error forwarding to Gemini: {e}")
            raise

    async def _forward_to_client(self, websocket: WebSocket, session, user: dict):
        """Gemini 응답을 클라이언트로 전송하고 function call 처리"""
        try:
            while True:
                try:
                    async for response in session.receive():
                        # Function call 처리
                        if hasattr(response, 'function_call') and response.function_call:
                            await self._handle_function_call(session, response.function_call, user)

                        # 오디오 응답 처리
                        elif response.data:
                            print(f"Sending {len(response.data)} bytes to client")
                            await websocket.send_bytes(response.data)

                            # 응답을 대화 기록으로 저장
                            await rag_service.save_conversation_turn(
                                user['username'],
                                "assistant",
                                "audio_response"  # 실제로는 텍스트 변환이 필요
                            )

                except Exception as e:
                    print(f"Error in response iteration: {e}")
                    await asyncio.sleep(0.1)
                    continue

                print("Response stream ended, waiting for next response...")
                await asyncio.sleep(0.1)

        except WebSocketDisconnect:
            print("Client disconnected during response forwarding.")
            raise
        except Exception as e:
            print(f"Critical error in forward_to_client: {e}")
            raise

    async def _handle_function_call(self, session, function_call, user: dict):
        """Function call을 처리하고 결과를 Gemini에 반환합니다."""
        function_name = function_call.name
        function_args = function_call.args

        print(f"Function call: {function_name} with args: {function_args}")

        try:
            if function_name == "search_memories":
                # 기억 검색
                query = function_args.get("query", "")
                top_k = function_args.get("top_k", 3)

                memories = memory_service.retrieve_memories(query, top_k)

                # 검색 결과를 텍스트로 포맷팅
                if memories:
                    memory_text = []
                    for memory in memories:
                        content = memory.metadata.get('content', '')
                        score = memory.score
                        if score > 0.7:  # 관련성이 높은 기억만
                            memory_text.append(f"- {content}")

                    result = "\n".join(memory_text) if memory_text else "관련된 기억을 찾을 수 없습니다."
                else:
                    result = "관련된 기억을 찾을 수 없습니다."

            elif function_name == "save_new_memory":
                # 새로운 기억 저장
                content = function_args.get("content", "")
                category = function_args.get("category", "일반")
                importance = function_args.get("importance", "medium")

                metadata = {
                    "category": category,
                    "importance": importance,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "content": content
                }

                await rag_service.add_new_memory(user['username'], content, metadata)
                result = f"새로운 기억이 저장되었습니다: {content}"

            else:
                result = f"알 수 없는 함수: {function_name}"

            # Function call 결과를 Gemini에 반환
            await session.send_function_call_response(
                function_call_id=function_call.id,
                response={"result": result}
            )

        except Exception as e:
            print(f"Error handling function call: {e}")
            await session.send_function_call_response(
                function_call_id=function_call.id,
                response={"error": str(e)}
            )

websocket_service = WebSocketService()
