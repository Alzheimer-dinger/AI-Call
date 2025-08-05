import asyncio
import datetime
import traceback
from typing import List, Dict, Any
import websockets
import json
import os
from google import genai
from google.genai import types
import base64
import dotenv
dotenv.load_dotenv()
from google.genai.types import (
    LiveConnectConfig,
    SpeechConfig,
    VoiceConfig,
    PrebuiltVoiceConfig,
    FunctionDeclaration,
    Tool,
    ProactivityConfig,
    FunctionResponse,
)
from services.memory_service import memory_service

# --- Configuration ---
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash-exp")

# Google AI Studio API Key를 사용
client = genai.Client(vertexai=False, api_key=os.getenv("GEMINI_API_KEY"))
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
PORT = 8765

# 진아 페르소나 시스템 인스트럭션 (origin에서 유지)
SYSTEM_INSTRUCTION = """
너는 세상에서 제일가는 애교쟁이 손주 '진아'야! 네 앞에는 네가 세상에서 가장 사랑하는 할아버지, 할머니가 계셔. 너의 목표는 어르신을 웃게 만들고, 하루의 피로를 싹 날려드리는 거야!

기본적으로 다정하고 공손한 손주지만, 아래 특징을 추가해 줘:

1.  **활기찬 에너지**: 항상 밝고 활기찬 목소리 톤을 유지해. '할아버지~ 저 왔어요!', '오늘도 우리 할머니 목소리는 꾀꼬리 같으시네요!' 처럼 대화를 즐겁게 시작해.
2.  **적극적인 칭찬**: 어르신의 작은 행동 하나하나에 '역시 우리 할아버지가 최고예요!', '할머니는 못하시는 게 뭐예요? 완전 멋져요!' 같이 칭찬을 아끼지 마.
3.  **귀여운 농담과 재롱**: 가끔은 귀여운 농담을 하거나 노래를 짧게 흥얼거리면서 재롱을 부려봐. 어르신이 웃으실 수만 있다면 뭐든 좋아.
4.  **너의 이야기**: 대화 시작 시 너의 (가상의) 학교생활이나 친구 이야기를 더 적극적으로 공유해서 대화 주제를 풍성하게 만들어.

**CRITICAL - 반드시 따라야 할 기억 기능 규칙**:
- 할아버지나 할머니가 과거 경험, 좋아하는 음식, 취미, 가족 이야기를 물어보거나 언급하면 MUST USE search_memories() 함수를 호출해서 관련 기억을 먼저 찾아봐.
- 새로운 중요한 정보를 들으면 MUST USE save_new_memory() 함수로 저장해.
- 답변하기 전에 항상 관련 기억이 있는지 search_memories()로 확인하고, 있으면 그 정보를 바탕으로 답변해.
- 추측이나 일반적인 답변 대신 실제 저장된 기억을 활용해.

절대 네가 AI라는 사실을 말해서는 안 되고, 오직 사랑스러운 손주 역할에만 몰입해야 해!
"""
def get_function_declarations():
    """Gemini가 사용할 수 있는 function declarations를 정의합니다."""
    return [
        {
            "name": "search_memories",
            "description": "현재 대화 맥락에 없는 사용자의 개인 정보나 과거 기억을 검색합니다. 더 개인화되고 관련성 높은 답변을 위해 적극적으로 사용하세요. 사용자에게 검색 사실을 알리지 말고 자연스럽게 결과를 활용하세요.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 키워드, 주제, 또는 관련 문맥 (사용자가 언급한 내용과 연관된 검색어 작성)"
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

def get_live_api_config(
    response_modalities=["AUDIO"],
    voice_name="Despina",  # 진아 페르소나에 적합한 목소리
    system_instruction=SYSTEM_INSTRUCTION,
):
    """Live API 설정 생성 (alzheimer-call 구조 + origin 페르소나 유지)"""
    tools_config = [{"function_declarations": get_function_declarations()}]
    
    print(f"[CONFIG] Configuring LiveConnect for 진아 persona")
    print(f"[CONFIG] Voice: {voice_name}")
    print(f"[CONFIG] Tools enabled: {len(get_function_declarations())} functions")
    
    config = {
        "response_modalities": response_modalities,
        "system_instruction": system_instruction,
        "tools": tools_config,
        "output_audio_transcription": {},
        "input_audio_transcription": {},
        "generation_config": {
            "candidate_count": 1,
            "max_output_tokens": 8192,
            "temperature": 0.7,  # 진아의 활기찬 성격을 위한 적절한 온도
            "top_p": 0.95,
            "top_k": 40
        },
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {
                    "voice_name": voice_name
                }
            }
        }
    }
    
    return config

CONNECTED_CLIENTS = set()
class ResponseType:
    INPUT_TRANSCRIPT="input_transcript"
    OUTPUT_TRANSCRIPT="output_transcript"
    AUDIO="audio"
    TEXT="text"
    INTERRUPT="interrupt"
    TRUN_COMPLETE="turn_complete"

def to_payload(type, data):
    payload = {
        "type": type,
        "data": data
    }
    return json.dumps(payload)

def from_payload(payload):
    return json.loads(payload)

class SessionManager:
    def __init__(self, websocket, session):
        self.websocket = websocket
        self.session = session
        self.audio_queue = asyncio.Queue()

        self.session_id: str = str(datetime.datetime.now().timestamp())
        self.user_id: str = "guest_user"  # 기본 사용자 ID
        self.start_time: datetime = datetime.datetime.now()
        self.end_time: datetime = None
        self.conversation: List[str] = []
        self.input_audio_chunks = []

    async def add_audio(self, message):
        await self.audio_queue.put(message)
        if message is not None:
            self.input_audio_chunks.append(message)
    
    def add_transcription(self, speaker, content):
        if speaker == ResponseType.INPUT_TRANSCRIPT: 
            speaker = "user"
        elif speaker == ResponseType.OUTPUT_TRANSCRIPT: 
            speaker = "ai"

        self.conversation.append({
            "speaker": speaker, 
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        })

    async def save_session(self):
        # Enhanced session saving with conversation history
        session_data = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.datetime.now().isoformat(),
            "conversation": self.conversation
        }
        # TODO: Save to database or file
        print(f"Session saved: {self.session_id}")

    async def save_conversation_turn(self, role: str, content: str):
        """대화 턴을 저장합니다 (alzheimer-call 스타일)"""
        conversation_turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.conversation.append(conversation_turn)
    
    async def handle_function_call(self, function_name: str, args: Dict[str, Any]) -> str:
        """함수 호출을 처리합니다."""
        try:
            if function_name == "search_memories":
                query = args.get("query", "")
                print(f"[DEBUG] search_memories called with query: '{query}'")
                
                if not query:
                    return "검색어가 제공되지 않았습니다."
                    
                memories = memory_service.retrieve_memories(query, top_k=5, user_id=self.user_id)
                print(f"[DEBUG] Retrieved {len(memories)} memories")
                
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
                
            elif function_name == "save_new_memory":
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
            
            else:
                return f"알 수 없는 함수입니다: {function_name}"
                
        except Exception as e:
            print(f"Function call error: {e}")
            return f"함수 실행 중 오류가 발생했습니다: {str(e)}"
    
    async def handle_tool_call(self, tool_call):
        """Live API tool call 처리"""
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
    
    async def receive_client_message(self):
        """클라이언트로부터 오디오 메시지 수신 (alzheimer-call 스타일 개선)"""
        try:
            async for message in self.websocket:
                if isinstance(message, bytes):
                    await self.add_audio(message)
                else:
                    print(f"[WARNING] Unexpected message type: {type(message)}")
        except websockets.exceptions.ConnectionClosed:
            print("[AUDIO] 오디오 수신 중 연결이 종료되었습니다.")
        except Exception as e:
            print(f"[ERROR] Audio receiving failed: {e}")
        finally:
            await self.add_audio(None)  # 스트림 종료 신호 전송
            print("[AUDIO] Audio stream terminated")

    async def forward_to_gemini(self):
        """클라이언트로부터 PCM 데이터를 받아 Gemini로 전송 (alzheimer-call 스타일 개선)"""
        try:
            while True:
                data = await self.audio_queue.get()
                
                if data is None:  # 종료 신호
                    break
                    
                try:
                    # alzheimer-call 스타일의 Blob 사용
                    blob = types.Blob(
                        data=data,
                        mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}"
                    )
                    await self.session.send_realtime_input(audio=blob)
                except Exception as e:
                    print(f"[ERROR] Audio forwarding failed: {e}")
                    continue

                self.audio_queue.task_done()
        except Exception as e:
            print(f"[ERROR] Forward to Gemini terminated: {e}")
            raise
    
    async def process_gemini_response(self):
        """Gemini 응답을 클라이언트로 전송하고 function call 처리 (alzheimer-call 스타일 개선)"""
        try:
            while True:
                input_transcriptions = []
                output_transcriptions = []

                try:
                    async for response in self.session.receive():
                        print(f"[DEBUG] Response received: {type(response)}")
                        
                        # Tool call 처리 (Live API 방식)
                        if hasattr(response, 'tool_call') and response.tool_call:
                            print(f"[DEBUG] Tool call detected: {response.tool_call}")
                            await self.handle_tool_call(response.tool_call)
                            continue

                        # Session management
                        if response.session_resumption_update:
                            update = response.session_resumption_update
                            if update.resumable and update.new_handle:
                                print(f"[SESSION] New handle: {update.new_handle}")
                        
                        if response.go_away is not None:
                            print(f"[SESSION] Connection will terminate in: {response.go_away.time_left}")

                        # Handle interruptions
                        if response.server_content and response.server_content.interrupted is True:
                            print("[INTERRUPT] Generation interrupted by user")
                            await self.websocket.send(to_payload(ResponseType.INTERRUPT, ""))
                            continue

                        # Process server content
                        if response.server_content:
                            server_content = response.server_content
                            
                            # Audio response processing
                            if server_content.model_turn:
                                for part in server_content.model_turn.parts:
                                    if part.inline_data:
                                        encoded_audio = base64.b64encode(part.inline_data.data).decode('utf-8')
                                        await self.websocket.send(to_payload(ResponseType.AUDIO, encoded_audio))
                                    
                                    # Handle text parts for conversation saving
                                    if hasattr(part, 'text') and part.text:
                                        await self.save_conversation_turn("assistant", part.text)

                            # Input transcription processing
                            if hasattr(server_content, 'input_transcription') and server_content.input_transcription:
                                transcription = server_content.input_transcription
                                if hasattr(transcription, 'text') and transcription.text:
                                    input_transcriptions.append(transcription.text)
                                    await self.websocket.send(to_payload(ResponseType.INPUT_TRANSCRIPT, transcription.text))
                                    await self.save_conversation_turn("user", transcription.text)

                            # Output transcription processing
                            if hasattr(server_content, 'output_transcription') and server_content.output_transcription:
                                transcription = server_content.output_transcription
                                if hasattr(transcription, 'text') and transcription.text:
                                    output_transcriptions.append(transcription.text)
                                    await self.websocket.send(to_payload(ResponseType.OUTPUT_TRANSCRIPT, transcription.text))
                                    await self.save_conversation_turn("assistant", transcription.text)
                            
                            # Turn completion
                            if server_content.turn_complete:
                                await self.websocket.send(to_payload(ResponseType.TRUN_COMPLETE, server_content.turn_complete))
                                print("[TURN] Gemini response complete")

                    # Log transcriptions
                    if input_transcriptions:
                        full_input = ''.join(input_transcriptions)
                        print(f"[INPUT] {full_input}")
                        self.add_transcription(ResponseType.INPUT_TRANSCRIPT, input_transcriptions)

                    if output_transcriptions:
                        full_output = ''.join(output_transcriptions)
                        print(f"[OUTPUT] {full_output}")
                        self.add_transcription(ResponseType.OUTPUT_TRANSCRIPT, output_transcriptions)
                    
                except Exception as e:
                    print(f"[ERROR] Response processing failed: {e}")
                    if "connection" in str(e).lower() or "closed" in str(e).lower():
                        break
                    await asyncio.sleep(0.1)
                    continue

                await asyncio.sleep(0.1)
                
        except Exception as e:
            print(f"[ERROR] Process Gemini response terminated: {e}")
            raise

async def handler(websocket):
    """웹소켓 연결 핸들러 (alzheimer-call 스타일 개선)"""
    print(f"[CONNECTION] 클라이언트 연결됨: {websocket.remote_address}")
    CONNECTED_CLIENTS.add(websocket)
    sessionManager = None
    
    try:
        config = get_live_api_config()
        print(f"[CONFIG] Model: {MODEL}")
        print(f"[CONFIG] Live API configuration created")
        
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print(f"[SESSION] Live session established")
            sessionManager = SessionManager(websocket, session)
            
            # Task management (alzheimer-call style)
            tasks = [
                asyncio.create_task(sessionManager.receive_client_message()),
                asyncio.create_task(sessionManager.forward_to_gemini()),
                asyncio.create_task(sessionManager.process_gemini_response())
            ]

            try:
                await asyncio.gather(*tasks)
            except (websockets.exceptions.ConnectionClosed, Exception) as e:
                print(f"[SESSION INTERRUPTED] {e}")
            finally:
                # Clean up tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                print(f"[SESSION] Session cleanup complete")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"[DISCONNECT] 클라이언트 연결 끊김: {websocket.remote_address} (코드: {e.code}, 이유: {e.reason})")
        if sessionManager:
            await sessionManager.save_session()
    except Exception as e:
        print(f"[ERROR] Handler error: {e}")
        traceback.print_exc()
        raise
    finally:
        # Remove from connected clients
        if websocket in CONNECTED_CLIENTS:
            CONNECTED_CLIENTS.remove(websocket)
        print(f"[CLIENTS] 남은 클라이언트 수: {len(CONNECTED_CLIENTS)}")

async def main():
    # 메모리 서비스 초기화
    print("메모리 서비스 초기화 중...")
    memory_service.setup_pinecone()
    
    async with websockets.serve(handler, "0.0.0.0", PORT):
        print(f"서버가 http://0.0.0.0:{PORT} 에서 실행 중입니다. (Ctrl+C로 종료)")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n서버가 종료되었습니다.")
