import asyncio
import json
from datetime import datetime
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
        """Gemini가 사용할 수 있는 function declarations를 정의합니다."""
        return [
            {
                "name": "search_memories",
                "description": "사���자와 관련된 기억을 검색합니다. 대화 중 관련 정보가 필요할 때 사용하세요.",
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
                            "description": "기억의 중요도",
                            "default": "medium"
                        }
                    },
                    "required": ["content", "category"]
                }
            }
        ]

    async def handle_live_audio_session(self, websocket: WebSocket, user: dict):
        try:
            # 사용자별 강화된 시스템 인스트럭션 생성
            enhanced_instruction = await rag_service.get_enhanced_system_instruction(
                user['username']
            )

            print(f"[SESSION START] Live session starting for {user['username']}")
            print(f"[SYSTEM PROMPT] {enhanced_instruction}")

            config = {
                "response_modalities": ["AUDIO"],
                "system_instruction": enhanced_instruction,
                "tools": [{"function_declarations": self._get_function_declarations()}],
                "output_audio_transcription": {},
                "input_audio_transcription": {},
                "generation_config": {
                    "candidate_count": 1,
                    "max_output_tokens": 8192,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40
                }
            }

            async with self.client.aio.live.connect(model=self.model, config=config) as session:
                print(f"[SESSION OPEN] Live session established for {user['username']}")

                tasks = [
                    asyncio.create_task(self._forward_to_gemini(websocket, session, user)),
                    asyncio.create_task(self._forward_to_client(websocket, session, user))
                ]

                try:
                    await asyncio.gather(*tasks)
                except (WebSocketDisconnect, Exception) as e:
                    print(f"[SESSION INTERRUPTED] {e}")
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    print(f"[SESSION CLOSE] Live session ended for {user['username']}")

        except Exception as e:
            print(f"[SESSION ERROR] {e}")
            raise

    async def _forward_to_gemini(self, websocket: WebSocket, session, user: dict):
        """클라이언트로부터 PCM 데이터를 받아 Gemini로 전송"""
        try:
            while True:
                pcm_data = await websocket.receive_bytes()
                blob = types.Blob(
                    data=pcm_data,
                    mime_type=f"audio/pcm;rate={self.target_sample_rate}"
                )
                await session.send_realtime_input(audio=blob)

        except WebSocketDisconnect:
            raise
        except Exception as e:
            print(f"[ERROR] Audio forwarding failed: {e}")
            raise

    async def _forward_to_client(self, websocket: WebSocket, session, user: dict):
        """Gemini 응답을 클라이언트로 전송하고 function call 처리"""
        try:
            while True:
                try:
                    async for response in session.receive():
                        # Function call 처리
                        if hasattr(response, 'function_calls') and response.function_calls:
                            for function_call in response.function_calls:
                                await self._handle_function_call(session, function_call, user)

                        elif hasattr(response, 'function_call') and response.function_call:
                            await self._handle_function_call(session, response.function_call, user)

                        # 오디오 응답 처리
                        elif hasattr(response, 'data') and response.data:
                            await websocket.send_bytes(response.data)

                        # 텍스트 응답 처리 (전사 기능)
                        elif hasattr(response, 'text') and response.text:
                            await rag_service.save_conversation_turn(
                                user['username'], "assistant", response.text
                            )
                        
                        # server_content가 있는 응답 처리 (전사 기능)
                        elif hasattr(response, 'server_content') and response.server_content:
                            server_content = response.server_content
                            
                            # 사용자 입력 전사 처리
                            if hasattr(server_content, 'input_transcription') and server_content.input_transcription:
                                transcription = server_content.input_transcription
                                if hasattr(transcription, 'text') and transcription.text:
                                    await rag_service.save_conversation_turn(
                                        user['username'], "user", transcription.text
                                    )
                                    
                                    # 클라이언트에 전사 데이터 전송
                                    transcription_message = {
                                        "type": "transcription",
                                        "speaker": "user",
                                        "text": transcription.text
                                    }
                                    await websocket.send_text(json.dumps(transcription_message))
                            
                            # AI 응답 전사 처리
                            if hasattr(server_content, 'output_transcription') and server_content.output_transcription:
                                transcription = server_content.output_transcription
                                if hasattr(transcription, 'text') and transcription.text:
                                    await rag_service.save_conversation_turn(
                                        user['username'], "assistant", transcription.text
                                    )
                                    
                                    # 클라이언트에 전사 데이터 전송
                                    transcription_message = {
                                        "type": "transcription",
                                        "speaker": "ai",
                                        "text": transcription.text
                                    }
                                    await websocket.send_text(json.dumps(transcription_message))
                            
                            # 텍스트 응답 처리 (model_turn에서 텍스트 부분)
                            if hasattr(server_content, 'model_turn') and server_content.model_turn:
                                model_turn = server_content.model_turn
                                if hasattr(model_turn, 'parts'):
                                    for part in model_turn.parts:
                                        if hasattr(part, 'text') and part.text:
                                            await rag_service.save_conversation_turn(
                                                user['username'], "assistant", part.text
                                            )
                                            
                                            # 클라이언트에 텍스트 응답 전송
                                            transcription_message = {
                                                "type": "transcription",
                                                "speaker": "ai",
                                                "text": part.text
                                            }
                                            await websocket.send_text(json.dumps(transcription_message))
                                        
                                        # executable_code 처리 - 이를 무시하고 계속 진행
                                        elif hasattr(part, 'executable_code') and part.executable_code:
                                            print(f"[EXECUTABLE CODE] Ignoring code execution: {part.executable_code.code[:100]}...")
                                            # 코드 실행은 하지 않고 무시

                except Exception as e:
                    print(f"[ERROR] Response processing failed: {e}")
                    await asyncio.sleep(0.1)
                    continue

                await asyncio.sleep(0.1)

        except WebSocketDisconnect:
            raise
        except Exception as e:
            print(f"[ERROR] Response forwarding failed: {e}")
            raise

    async def _handle_function_call(self, session, function_call, user: dict):
        """Function call을 처리하고 결과를 Gemini에 반환합니다."""
        function_name = function_call.name
        function_args = function_call.args if hasattr(function_call, 'args') else {}

        print(f"[FUNCTION CALL] {function_name}({function_args})")

        try:
            if function_name == "search_memories":
                # 기억 검색 (사용자별 필터링 적용)
                query = function_args.get("query", "")
                top_k = function_args.get("top_k", 3)

                memories = memory_service.retrieve_memories(query, top_k, user['username'])

                # 검색 결과를 텍스트로 포맷팅
                if memories:
                    memory_text = []
                    for memory in memories:
                        content = memory.metadata.get('content', '')
                        score = memory.score

                        if score > 0.5:  # 관련성 임계값을 낮춤
                            category = memory.metadata.get('category', '')
                            date = memory.metadata.get('date', '')
                            memory_info = f"- {content}"
                            if category:
                                memory_info += f" (분류: {category})"
                            if date:
                                memory_info += f" (날짜: {date})"
                            memory_text.append(memory_info)

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
                    "date": datetime.now().strftime("%Y-%m-%d")
                }

                memory_id = memory_service.add_memory(user['username'], content, metadata)
                result = f"기억이 저장되었습니다: {content}"

            else:
                result = f"알 수 없는 함수: {function_name}"

            print(f"[FUNCTION RESULT] {result}")

            # Function call 결과를 Gemini에 반환
            function_response = types.FunctionResponse(
                name=function_name,
                id=getattr(function_call, 'id', None),
                response={"result": result}
            )

            await session.send(function_response)

        except Exception as e:
            print(f"[FUNCTION ERROR] {function_name}: {e}")

            try:
                # 오류 응답 전송
                error_response = types.FunctionResponse(
                    name=function_name,
                    id=getattr(function_call, 'id', None),
                    response={"error": str(e)}
                )
                await session.send(error_response)
            except Exception as send_error:
                print(f"[ERROR] Failed to send function error response: {send_error}")

# 전역 인스턴스 생성
websocket_service = WebSocketService()
