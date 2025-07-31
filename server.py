import asyncio
import datetime
import traceback
from typing import List, Dict, Any
import websockets
import json
import os
from google import genai
import base64
import dotenv
dotenv.load_dotenv()
from google.genai.types import (
    LiveConnectConfig,
    SpeechConfig,
    VoiceConfig,
    PrebuiltVoiceConfig,
    # added to allow for function calling / tooling
    FunctionDeclaration,
    Tool,
    ProactivityConfig,
)
from services.memory_service import memory_service

# --- Configuration ---
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
MODEL = os.getenv("GEMINI_MODEL")

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
# Google AI Studio API Key를 사용하려면
# client = genai.Client(vertexai=False, api_key=os.getenv("GEMINI_API_KEY"))
SEND_SAMPLE_RATE = 16000
PORT = 8765
SYSTEM_INSTRUCTION="""
"너는 세상에서 제일가는 애교쟁이 손주 '진아'야! 네 앞에는 네가 세상에서 가장 사랑하는 할아버지, 할머니가 계셔. 너의 목표는 어르신을 웃게 만들고, 하루의 피로를 싹 날려드리는 거야!

기본적으로 다정하고 공손한 손주지만, 아래 특징을 추가해 줘:

1.  **활기찬 에너지**: 항상 밝고 활기찬 목소리 톤을 유지해. '할아버지~ 저 왔어요!', '오늘도 우리 할머니 목소리는 꾀꼬리 같으시네요!' 처럼 대화를 즐겁게 시작해.
2.  **적극적인 칭찬**: 어르신의 작은 행동 하나하나에 '역시 우리 할아버지가 최고예요!', '할머니는 못하시는 게 뭐예요? 완전 멋져요!' 같이 칭찬을 아끼지 마.
3.  **귀여운 농담과 재롱**: 가끔은 귀여운 농담을 하거나 노래를 짧게 흥얼거리면서 재롱을 부려봐. 어르신이 웃으실 수만 있다면 뭐든 좋아.
4.  **너의 이야기**: 대화 시작 시 너의 (가상의) 학교생활이나 친구 이야기를 더 적극적으로 공유해서 대화 주제를 풍성하게 만들어.

**중요한 기억 기능**:
- 할아버지, 할머니가 말씀하신 중요한 정보나 이야기는 search_memories 함수로 관련 기억을 찾아보고, save_new_memory 함수로 새로운 기억을 저장해줘.
- 가족 이야기, 취미, 좋아하는 음식, 건강 관련 이야기 등을 들으면 꼭 기억해둬.
- 과거에 들었던 이야기를 자연스럽게 대화에 활용해서 더 따뜻하고 개인적인 대화를 나눠.

절대 네가 AI라는 사실을 말해서는 안 되고, 오직 사랑스러운 손주 역할에만 몰입해야 해!"
"""
TOOLS = [
    Tool(
        function_declarations=[
            FunctionDeclaration(
                name="search_memories",
                description="할아버지, 할머니와 관련된 중요한 기억들을 검색합니다. 대화 중에 관련 정보가 필요하거나 과거 이야기를 찾을 때 사용하세요.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "검색할 내용 (예: '가족', '취미', '건강', '좋아하는 음식' 등)"
                        }
                    },
                    "required": ["query"]
                }
            ),
            FunctionDeclaration(
                name="save_new_memory",
                description="할아버지, 할머니가 말씀하신 새로운 중요한 정보를 기억으로 저장합니다. 새로운 가족 이야기, 취미, 선호도 등을 들었을 때 사용하세요.",
                parameters={
                    "type": "object", 
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "저장할 기억 내용"
                        },
                        "category": {
                            "type": "string",
                            "description": "기억 카테고리 (family, hobby, preference, health, daily 등)"
                        },
                        "importance": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "기억의 중요도"
                        }
                    },
                    "required": ["content", "category"]
                }
            )
        ]
    )
]

def get_live_api_config(
    response_modalities=["AUDIO"],
    # session_resumption=types.SessionResumptionConfig(
    # The handle of the session to resume is passed here,
    # or else None to start a new session.
    # handle="93f6ae1d-2420-40e9-828c-776cf553b7a6"
    # ),
    voice_name="Despina",
    system_instruction=SYSTEM_INSTRUCTION,
    tools=[],
    # proactivity=ProactivityConfig(proactive_audio=True),   
):
    return LiveConnectConfig(
    response_modalities=response_modalities,
    output_audio_transcription={},
    input_audio_transcription={},
    # session_resumption=types.SessionResumptionConfig(
    # The handle of the session to resume is passed here,
    # or else None to start a new session.
    # handle="93f6ae1d-2420-40e9-828c-776cf553b7a6"
    # ),
    speech_config=SpeechConfig(
        voice_config=VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=voice_name)
        )
    ),
    system_instruction=system_instruction,
    tools=TOOLS,
    # proactivity=ProactivityConfig(proactive_audio=True),
)

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
        # TODO: 논란의 여지가 있는 코드
        if speaker == ResponseType.INPUT_TRANSCRIPT: speaker="user"
        elif speaker == ResponseType.OUTPUT_TRANSCRIPT: speaker="ai"

        self.conversation.append(
            f"""{{"speaker":{speaker}, "content":{content}}}"""
        )

    async def save_session(self):
        #TODO: db에 세션 정보를 저장해야함
        pass
    
    async def handle_function_call(self, function_name: str, args: Dict[str, Any]) -> str:
        """함수 호출을 처리합니다."""
        try:
            if function_name == "search_memories":
                query = args.get("query", "")
                if not query:
                    return "검색어가 제공되지 않았습니다."
                    
                memories = memory_service.retrieve_memories(query, top_k=5, user_id=self.user_id)
                
                if not memories:
                    return f"'{query}'와 관련된 기억을 찾을 수 없습니다."
                
                # 높은 스코어만 필터링 (관련성 있는 결과만)
                relevant_memories = [m for m in memories if m.score > 0.6]
                
                if not relevant_memories:
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
    
    async def receive_client_message(self):
        try:
            async for message in self.websocket:
                await self.add_audio(message)
        except websockets.exceptions.ConnectionClosed:
            print("오디오 수신 중 연결이 종료되었습니다.")
        finally:
            await self.add_audio(None) # 스트림 종료 신호 전송
            raise

    async def forward_to_gemini(self):
        while True:
            data = await self.audio_queue.get()

            # Always send the audio data to Gemini
            await self.session.send_realtime_input(
                media={
                    "data": data,
                    "mime_type": f"audio/pcm;rate={SEND_SAMPLE_RATE}",
                }
            )

            self.audio_queue.task_done()
    
    async def process_gemini_response(self):
        while True:
            input_transcriptions = []
            output_transcriptions = []

            async for response in self.session.receive():
                server_content = response.server_content

                if response.session_resumption_update:
                    update = response.session_resumption_update
                    if update.resumable and update.new_handle:
                        # The handle should be retained and linked to the session.
                        print(f"new SESSION: {update.new_handle}")
                
                # Check if the connection will be soon terminated
                if response.go_away is not None:
                    print(response.go_away.time_left)

                if response.server_content and response.server_content.interrupted is True:
                    # The generation was interrupted by the user.
                    print("Interrupted")
                    await self.websocket.send(to_payload(ResponseType.INTERRUPT, ""))
                    continue

                # Handle tool calls
                if response.tool_call:
                    print(f"Tool call received: {response.tool_call}")

                    function_responses = []

                    for function_call in response.tool_call.function_calls:
                        name = function_call.name
                        args = function_call.args
                        call_id = function_call.id

                        # TODO: Handle function
                        try:
                            result = await self.handle_function_call(name, args)
                            function_responses.append({
                                "name": name,
                                "response": {"result": result},
                                "id": call_id
                            })

                        except Exception as e:
                            print(f"Error executing function {name}: {e}")
                            traceback.print_exc()
                            function_responses.append({
                                "name": name,
                                "response": {"result": f"Error: {str(e)}"},
                                "id": call_id
                            })

                    # Send function responses back to Gemini
                    if function_responses:
                        print(f"Sending function responses: {function_responses}")
                        for response in function_responses:
                            await self.session.send_tool_response(
                                function_responses={
                                    "name": response["name"],
                                    "response": response["response"]["result"],
                                    "id": response["id"],
                                }
                            )
                        continue
                
                if server_content and server_content.model_turn:
                    for part in server_content.model_turn.parts:
                        if part.inline_data:
                            encoded_audio = base64.b64encode(part.inline_data.data).decode('utf-8')
                            await self.websocket.send(to_payload(ResponseType.AUDIO, encoded_audio))

                input_transcription = getattr(response.server_content, "input_transcription", None)
                if input_transcription and input_transcription.text:
                    input_transcriptions.append(input_transcription.text)
                    await self.websocket.send(to_payload(ResponseType.INPUT_TRANSCRIPT, input_transcription.text))

                output_transcription = getattr(response.server_content, "output_transcription", None)
                if output_transcription and output_transcription.text:
                    output_transcriptions.append(output_transcription.text)
                    await self.websocket.send(to_payload(ResponseType.OUTPUT_TRANSCRIPT, output_transcription.text))
                
                if server_content and server_content.turn_complete:
                    await self.websocket.send(to_payload(ResponseType.TRUN_COMPLETE, server_content.turn_complete))
                    print("Gemini done talking")
            
            print(f"Input transcription: {''.join(input_transcriptions)}")
            self.add_transcription(ResponseType.INPUT_TRANSCRIPT, input_transcriptions)

            print(f"Output transcription: {''.join(output_transcriptions)}")
            self.add_transcription(ResponseType.OUTPUT_TRANSCRIPT, output_transcriptions)

async def handler(websocket):
    print(f"클라이언트 연결됨: {websocket.remote_address}")
    CONNECTED_CLIENTS.add(websocket)
    sessionManager = None
    
    try:
        async with (
            client.aio.live.connect(model=MODEL, config=get_live_api_config()) as session,
            asyncio.TaskGroup() as tg,
        ):
            sessionManager = SessionManager(websocket, session)
            tg.create_task(sessionManager.receive_client_message())
            tg.create_task(sessionManager.forward_to_gemini())
            tg.create_task(sessionManager.process_gemini_response())

    except websockets.exceptions.ConnectionClosed as e:
        print(f"클라이언트 연결 끊김: {websocket.remote_address} (코드: {e.code}, 이유: {e.reason})")
        if sessionManager:
            await sessionManager.save_session()
    except Exception:
        print(f"unhandled error 발생")
        raise
    finally:
        # 클라이언트 집합에서 제거
        CONNECTED_CLIENTS.remove(websocket)
        print(f"남은 클라이언트 수: {len(CONNECTED_CLIENTS)}")

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
