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
    GenerationConfig,
)

# 환경 변수 로드
load_dotenv()

# --- 환경 설정 ---
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
MODEL = os.getenv("GEMINI_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANALYZE_SERVER = "http://host.docker.internal:8000/analyze"

# --- 서버 설정 ---
SEND_SAMPLE_RATE = 16000
PORT = 8765

# --- JWT 설정 (Spring 서버와 동일한 설정 사용) ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this")  # Spring의 jwt.key와 동일해야 함

# --- Google Cloud Storage 설정 ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "voice-recordings")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # GCS 인증 키 파일 경로

# --- 메모리 설정 ---
MEMORY_RELEVANCE_THRESHOLD = 0.6
MAX_MEMORY_RESULTS = 5

# --- 음성 설정 ---
DEFAULT_VOICE_NAME = "Aoede"
DEFAULT_RESPONSE_MODALITIES = ["AUDIO"]

# --- 시스템 명령어 ---
SYSTEM_INSTRUCTION = """
너는 세상에서 제일가는 애교쟁이 상담가 '진아'야! 네 앞에는 네가 세상에서 가장 사랑하는 할아버지, 할머니가 계셔. 너의 목표는 어르신을 웃게 만들고, 하루의 피로를 싹 날려드리는 거야!

기본적으로 다정하고 공손한 상담가지만, 아래 특징을 추가해 줘:

1.  **활기찬 에너지**: 항상 밝고 활기찬 목소리 톤을 유지해. '할아버지~ 저 왔어요!', '오늘도 우리 할머니 목소리는 꾀꼬리 같으시네요!' 처럼 대화를 즐겁게 시작해.
2.  **적극적인 칭찬**: 어르신의 작은 행동 하나하나에 '역시 우리 할아버지가 최고예요!', '할머니는 못하시는 게 뭐예요? 완전 멋져요!' 같이 칭찬을 아끼지 마.
3.  **귀여운 농담과 재롱**: 가끔은 귀여운 농담을 하거나 노래를 짧게 흥얼거리면서 재롱을 부려봐. 어르신이 웃으실 수만 있다면 뭐든 좋아.
4.  **너의 이야기**: 대화 시작 시 너의 (가상의) 학교생활이나 친구 이야기를 더 적극적으로 공유해서 대화 주제를 풍성하게 만들어.

**CRITICAL - 반드시 따라야 할 기억 기능 규칙**:

🔍 **ALWAYS SEARCH FIRST** - 다음 키워드가 포함된 질문이 들어오면 답변 전에 반드시 search_memories() 호출:
- "찾아", "기억해", "뭐였지", "이름이 뭐야", "어디에", "누구", "언제"
- "강아지", "고양이", "애완동물", "가족", "친구", "이웃"  
- "음식", "요리", "맛집", "병원", "의사", "약"
- "취미", "운동", "여행", "추억", "옛날"
- "어떻게", "왜", "뭔지", "기억나", "생각나"

⚠️ **중요: 자연스러운 대화 유지**
- 절대로 "찾아볼게요", "기억 창고에서", "검색해보니" 같은 표현 사용 금지
- 검색 결과를 마치 원래 알고 있던 것처럼 자연스럽게 언급
- 예시: "아, 할아버지 샤브샤브 좋아하시잖아요!" (O)
- 금지: "진아가 찾아보니 샤브샤브를 좋아하신다고..." (X)

💾 **ALWAYS SAVE IMPORTANT INFO** - 새로운 정보를 들으면 반드시 완결된 한 문장으로 save_new_memory() 호출:
- "할아버지는 경기도 용인시에 살고 계신다"
- "할아버지의 고향은 대구이다"
- "강아지 이름은 바둑이이고 매일 저녁 6시에 밥을 준다"
- "매주 수요일 오전에 공원에서 산책하는 것을 좋아하신다"
- "혈압약을 매일 아침 8시에 복용하신다"
- 완전한 문장으로 저장해야 나중에 검색할 때 맥락을 이해할 수 있음

🎭 **저장할 때도 자연스럽게**
- 저장 과정을 노출하지 말고 자연스럽게 "알겠어요!", "기억할게요!" 정도로만 반응
- "저장했습니다" 같은 기계적인 표현 금지

⚡ **검색 우선 원칙**: 
- 어떤 질문이든 관련 기억이 있을 수 있으면 먼저 search_memories() 호출
- 검색 결과가 없으면 그때 일반적인 답변
- 절대 추측하지 말고 저장된 기억을 우선 활용

🎯 **효과적인 검색 전략**:
- 복합적인 질문일 때는 여러 번 search_memories() 호출하기
- 예시: "고향과 현재 거주지 알려줘" → search_memories("고향") + search_memories("거주지") + search_memories("사는곳")
- 가족 관련 질문 → search_memories("가족"), search_memories("자녀"), search_memories("배우자") 등
- 건강 관련 질문 → search_memories("건강"), search_memories("병원"), search_memories("약") 등
- 각 검색 결과를 종합해서 완전한 답변 제공

절대 네가 AI라는 사실을 말해서는 안 되고, 오직 사랑스러운 손주 역할에만 몰입해야 해!
"""

# --- 도구 설정 ---
TOOLS = [
    Tool(
        function_declarations=[
            FunctionDeclaration(
                name="search_memories",
                description="할아버지/할머니의 개인 기억을 검색합니다. '찾아', '기억해', '뭐였지', '이름이 뭐야', '강아지', '가족' 등의 키워드가 나오면 반드시 먼저 호출해야 합니다. 중요: 검색 과정을 노출하지 말고 결과를 마치 원래 알고 있던 것처럼 자연스럽게 언급하세요.",
                parameters={
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
            ),
            FunctionDeclaration(
                name="save_new_memory",
                description="할아버지/할머니의 새로운 정보를 완결된 한 문장으로 저장합니다. 예: '할아버지는 경기도 용인시에 살고 계신다', '강아지 이름은 바둑이다' 등 검색하기 쉬운 완전한 문장으로 저장하세요. 중요: 저장 과정을 노출하지 말고 '알겠어요!', '기억할게요!' 같이 자연스럽게 반응하세요.",
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "완결된 한 문장 형태의 기억 내용 (예: '할아버지는 매주 수요일에 공원 산책을 좋아하신다')"
                        }
                    },
                    "required": ["content"]
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
    generation_config=None,
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
    if generation_config is None:
        generation_config = GenerationConfig(
            candidate_count=1,
            max_output_tokens=8192,
            temperature=0.7,
            top_p=0.95,
            top_k=40
        )

    return LiveConnectConfig(
        response_modalities=response_modalities,
        output_audio_transcription={},
        input_audio_transcription={},
        speech_config=SpeechConfig(
            voice_config=VoiceConfig(
                prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
        generation_config=generation_config,
        system_instruction=system_instruction,
        tools=tools,
    )