import os
from dotenv import load_dotenv
from google.genai.types import (
    FunctionDeclaration,
    Tool,
    PrebuiltVoiceConfig,
    VoiceConfig,
    SpeechConfig,
    LiveConnectConfig,
    ProactivityConfig,
)

# 환경 변수 로드
load_dotenv()

# --- 환경 설정 ---
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
MODEL = os.getenv("GEMINI_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- 서버 설정 ---
SEND_SAMPLE_RATE = 16000
PORT = 8765

# --- 메모리 설정 ---
MEMORY_RELEVANCE_THRESHOLD = 0.6
MAX_MEMORY_RESULTS = 5

# --- 음성 설정 ---
DEFAULT_VOICE_NAME = "Despina"
DEFAULT_RESPONSE_MODALITIES = ["AUDIO"]

# --- 시스템 명령어 ---
SYSTEM_INSTRUCTION = """
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

# --- 도구 설정 ---
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

# --- 응답 타입 상수 ---
class ResponseType:
    INPUT_TRANSCRIPT = "input_transcript"
    OUTPUT_TRANSCRIPT = "output_transcript"
    AUDIO = "audio"
    TEXT = "text"
    INTERRUPT = "interrupt"
    TURN_COMPLETE = "turn_complete"

# --- 라이브 API 설정 함수 ---
def get_live_api_config(
    response_modalities=None,
    voice_name=None,
    system_instruction=None,
    tools=None,
):
    """라이브 API 설정을 생성합니다."""
    if response_modalities is None:
        response_modalities = DEFAULT_RESPONSE_MODALITIES
    if voice_name is None:
        voice_name = DEFAULT_VOICE_NAME
    if system_instruction is None:
        system_instruction = SYSTEM_INSTRUCTION
    if tools is None:
        tools = TOOLS
        
    return LiveConnectConfig(
        response_modalities=response_modalities,
        output_audio_transcription={},
        input_audio_transcription={},
        speech_config=SpeechConfig(
            voice_config=VoiceConfig(
                prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
        system_instruction=system_instruction,
        tools=tools,
    )