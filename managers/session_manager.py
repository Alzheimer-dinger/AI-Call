import asyncio
import datetime
import base64
import traceback
from typing import List, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect

from settings import ResponseType, SEND_SAMPLE_RATE, MEMORY_RELEVANCE_THRESHOLD, MAX_MEMORY_RESULTS
from managers.websocket_manager import PayloadManager
from services.memory_service import memory_service

from google.genai.types import FunctionResponse

class SessionManager:
    """개별 세션을 관리하는 클래스"""
    
    def __init__(self, websocket: WebSocket, session):
        self.websocket = websocket
        self.session = session
        self.audio_queue = asyncio.Queue()
        
        # 세션 정보
        self.session_id: str = str(datetime.datetime.now().timestamp())
        self.user_id: str = "guest_user"  # 기본 사용자 ID
        self.start_time: datetime.datetime = datetime.datetime.now()
        self.end_time: datetime.datetime = None
        self.conversation: List[Dict] = []  # 타입 수정
        self.input_audio_chunks: List = []

    async def add_audio(self, message):
        """오디오 메시지를 큐에 추가"""
        await self.audio_queue.put(message)
        if message is not None:
            self.input_audio_chunks.append(message)
    
    def add_transcription(self, speaker: str, content: List[str]):
        """대화 내용을 기록에 추가"""
        # 스피커 타입 정규화
        if speaker == ResponseType.INPUT_TRANSCRIPT:
            speaker = "user"
        elif speaker == ResponseType.OUTPUT_TRANSCRIPT:
            speaker = "ai"

        content_text = ''.join(content) if isinstance(content, list) else content
        if content_text.strip():  # 빈 내용은 저장하지 않음
            self.conversation.append({
                "speaker": speaker, 
                "content": content_text,
                "timestamp": datetime.datetime.now().isoformat()
            })

    async def save_session(self):
        """세션 정보를 DB에 저장"""
        session_data = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.datetime.now().isoformat(),
            "conversation": self.conversation
        }
        
        # TODO: DB에 세션 정보를 저장
        print(f"세션 저장됨: {self.session_id}, 대화 수: {len(self.conversation)}")
    
    async def save_conversation_turn(self, role: str, content: str):
        """대화 턴을 저장합니다"""
        conversation_turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.conversation.append(conversation_turn)

    async def handle_function_call(self, function_name: str, args: Dict[str, Any]) -> str:
        """함수 호출을 처리"""
        try:
            if function_name == "search_memories":
                return await self._handle_search_memories(args)
            elif function_name == "save_new_memory":
                return await self._handle_save_memory(args)
            else:
                return f"알 수 없는 함수입니다: {function_name}"
                
        except Exception as e:
            print(f"Function call error: {e}")
            traceback.print_exc()
            return f"함수 실행 중 오류가 발생했습니다: {str(e)}"

    async def _handle_search_memories(self, args: Dict[str, Any]) -> str:
        """메모리 검색 처리"""
        query = args.get("query", "")
        if not query:
            return "검색어가 제공되지 않았습니다."
            
        memories = memory_service.retrieve_memories(query, top_k=5, user_id=self.user_id)
        print(f"[DEBUG] Retrieved {len(memories)} memories")
                
        
        if not memories:
            return f"'{query}'와 관련된 기억을 찾을 수 없습니다."
        
        if memories:
            for i, mem in enumerate(memories):
                print(f"[DEBUG] Memory {i}: score={mem.score}, content={mem.metadata.get('content', '')[:50]}")
        
        if not memories:
            print(f"[DEBUG] No memories found for query '{query}'")
            return f"'{query}'와 관련된 기억을 찾을 수 없습니다."
        
        

        # 높은 스코어만 필터링 (관련성 있는 결과만)
        relevant_memories = [m for m in memories if m.score > 0.6]
        print(f"[DEBUG] Filtered to {len(relevant_memories)} relevant memories (score > 0.6)")
        
        if not relevant_memories:
            print(f"[DEBUG] No relevant memories found (all scores <= 0.6)")
            return f"'{query}'와 관련된 기억을 찾을 수 없습니다."
        
        result = f"'{query}'와 관련된 기억들:\n"
        for memory in relevant_memories:
            content = memory.metadata.get('content', '')
            category = memory.metadata.get('category', '')
            date = memory.metadata.get('date', '')
            
            result += f"- {content}"
            if category:
                result += f" (카테고리: {category})"
            if date:
                result += f" (날짜: {date})"
            result += f" (관련도: {memory.score:.2f})\n"
        
        return result

    async def _handle_save_memory(self, args: Dict[str, Any]) -> str:
        """메모리 저장 처리"""
        content = args.get("content", "")
        category = args.get("category", "general")
        importance = args.get("importance", "medium")
        
        if not content:
            return "저장할 내용이 제공되지 않았습니다."
        
        metadata = {
            "category": category,
            "importance": importance,
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "session_id": self.session_id
        }
        
        memory_id = memory_service.add_memory(self.user_id, content, metadata)
        
        if memory_id:
            return f"새로운 기억이 저장되었습니다: '{content}' (카테고리: {category})"
        else:
            return "기억 저장 중 오류가 발생했습니다."

    async def receive_client_message(self):
        """클라이언트로부터 메시지 수신"""
        try:
            while True:
                # FastAPI WebSocket 방식으로 수정
                message = await self.websocket.receive_bytes()
                await self.add_audio(message)
        except WebSocketDisconnect as e:
            print("오디오 수신 중 WebSocket 연결이 종료되었습니다.")
            raise e
        except Exception as e:
            print(f"메시지 수신 중 오류: {e}")
            raise e
        finally:
            await self.add_audio(None)  # 스트림 종료 신호

    async def forward_to_gemini(self):
        """오디오 데이터를 Gemini로 전달"""
        while True:
            data = await self.audio_queue.get()
            
            if data is None:  # 종료 신호
                break
                
            try:
                await self.session.send_realtime_input(
                    media={
                        "data": data,
                        "mime_type": f"audio/pcm;rate={SEND_SAMPLE_RATE}",
                    }
                )

            except Exception as e:
                print(f"Gemini로 데이터 전송 중 오류: {e}")
            
            self.audio_queue.task_done()
    
    async def process_gemini_response(self):
        """Gemini 응답 처리"""
        while True:
            input_transcriptions = []
            output_transcriptions = []

            try:
                async for response in self.session.receive():
                    # 세션 재개 처리
                    if response.session_resumption_update:
                        update = response.session_resumption_update
                        if update.resumable and update.new_handle:
                            print(f"새 세션 핸들: {update.new_handle}")
                    
                    # 연결 종료 예정 알림
                    if response.go_away is not None:
                        print(f"연결 종료 예정: {response.go_away.time_left}")

                    server_content = response.server_content
                    if not server_content:
                        continue

                    # 중단 처리
                    if server_content.interrupted:
                        print("응답이 중단되었습니다.")
                        await self.websocket.send_text(
                            PayloadManager.to_payload(ResponseType.INTERRUPT, "")
                        )
                        continue

                    # 도구 호출 처리
                    if response.tool_call:
                        await self._handle_tool_calls(response.tool_call)
                        continue
                    
                    # 오디오 응답 처리
                    if server_content.model_turn:
                        await self._handle_audio_response(server_content.model_turn)

                    # 전사 처리
                    await self._handle_transcriptions(
                        server_content, input_transcriptions, output_transcriptions
                    )
                    
                    # 턴 완료 처리
                    if server_content.turn_complete:
                        await self.websocket.send_text(
                            PayloadManager.to_payload(ResponseType.TURN_COMPLETE, True)
                        )
                        print("Gemini 응답 완료")
                        
                        # 대화 기록 저장
                        if input_transcriptions:
                            self.add_transcription(ResponseType.INPUT_TRANSCRIPT, input_transcriptions)
                            input_transcriptions.clear()
                        
                        if output_transcriptions:
                            self.add_transcription(ResponseType.OUTPUT_TRANSCRIPT, output_transcriptions)
                            output_transcriptions.clear()
            
            except Exception as e:
                print(f"Gemini 응답 처리 중 오류: {e}")
                traceback.print_exc()

    async def _handle_tool_calls(self, tool_call):
        """도구 호출 처리"""
        print(f"[TOOL CALL] Processing {len(tool_call.function_calls)} function calls")
        
        function_responses = []
        for fc in tool_call.function_calls:
            function_name = fc.name
            function_args = fc.args if hasattr(fc, 'args') else {}
            
            print(f"[FUNCTION CALL] {function_name}({function_args})")
            
            try:
                if function_name == "search_memories":
                    query = function_args.get("query", "")
                    top_k = function_args.get("top_k", 3)
                    
                    print(f"[DEBUG] search_memories called with query: '{query}', top_k: {top_k}")
                    
                    if not query:
                        result = "검색어가 제공되지 않았습니다."
                    else:
                        memories = memory_service.retrieve_memories(query, top_k, self.user_id)
                        print(f"[DEBUG] Retrieved {len(memories)} memories")
                        
                        if memories:
                            memory_text = []
                            for memory in memories:
                                content = memory.metadata.get('content', '')
                                score = memory.score
                                print(f"[DEBUG] Memory: score={score}, content={content[:50]}")
                                
                                if score > 0.001:
                                    category = memory.metadata.get('category', '')
                                    date = memory.metadata.get('date', '')
                                    memory_info = f"- {content}"
                                    if category:
                                        memory_info += f" (분류: {category})"
                                    if date:
                                        memory_info += f" (날짜: {date})"
                                    memory_text.append(memory_info)
                            
                            if memory_text:
                                result = "검색된 기억:\n" + "\n".join(memory_text)
                            else:
                                result = "관련된 기억을 찾을 수 없습니다."
                        else:
                            result = "관련된 기억을 찾을 수 없습니다."
                
                elif function_name == "save_new_memory":
                    content = function_args.get("content", "")
                    category = function_args.get("category", "일반")
                    importance = function_args.get("importance", "medium")
                    
                    if not content:
                        result = "저장할 내용이 제공되지 않았습니다."
                    else:
                        metadata = {
                            "category": category,
                            "importance": importance,
                            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                            "session_id": self.session_id
                        }
                        
                        memory_id = memory_service.add_memory(self.user_id, content, metadata)
                        
                        if memory_id:
                            result = f"기억이 저장되었습니다: {content}"
                        else:
                            result = "기억 저장 중 오류가 발생했습니다."
                
                else:
                    result = f"알 수 없는 함수: {function_name}"
                
                print(f"[FUNCTION RESULT] {result}")
                
                # FunctionResponse 생성
                function_response = FunctionResponse(
                    id=fc.id,
                    name=fc.name,
                    response={"result": result}
                )
                function_responses.append(function_response)
                
            except Exception as e:
                print(f"[FUNCTION ERROR] {function_name}: {e}")
                traceback.print_exc()
                error_response = FunctionResponse(
                    id=fc.id,
                    name=fc.name,
                    response={"error": str(e)}
                )
                function_responses.append(error_response)
        
        # Send all function responses
        if function_responses:
            await self.session.send_tool_response(function_responses=function_responses)
    

    async def _handle_audio_response(self, model_turn):
        """오디오 응답 처리"""
        for part in model_turn.parts:
            if part.inline_data:
                encoded_audio = base64.b64encode(part.inline_data.data).decode('utf-8')
                try:
                    await self.websocket.send_text(
                        PayloadManager.to_payload(ResponseType.AUDIO, encoded_audio)
                    )
                except Exception as e:
                    print(f"오디오 전송 중 오류: {e}")
        

    async def _handle_transcriptions(self, server_content, input_transcriptions, output_transcriptions):
        """전사 내용 처리"""
        try:
            # 입력 전사
            if hasattr(server_content, 'input_transcription') and server_content.input_transcription:
                if server_content.input_transcription.text:
                    input_transcriptions.append(server_content.input_transcription.text)
                    await self.websocket.send_text(
                        PayloadManager.to_payload(
                            ResponseType.INPUT_TRANSCRIPT, 
                            server_content.input_transcription.text
                        )
                    )

            # 출력 전사
            if hasattr(server_content, 'output_transcription') and server_content.output_transcription:
                if server_content.output_transcription.text:
                    output_transcriptions.append(server_content.output_transcription.text)
                    await self.websocket.send_text(
                        PayloadManager.to_payload(
                            ResponseType.OUTPUT_TRANSCRIPT, 
                            server_content.output_transcription.text
                        )
                    )
        except Exception as e:
            print(f"전사 내용 전송 중 오류: {e}")