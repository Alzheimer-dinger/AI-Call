from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# --- 애플리케이션 시작 전에 환경 변수 로드 --- #
load_dotenv()

# 설정과 초기화 모듈 먼저 로드
from app.config import settings
from app.core import genai

# 라우터 임포트는 설정 이후에 수행
from app.api import router as api_router
from app.auth import router as auth_router
from app.services import setup_pinecone

app = FastAPI(
    title="Alzheimer Call AI Agent",
    version="0.1.0"
)

@app.on_event("startup")
def on_startup():
    """애플리케이션 시작 시 실행됩니다."""
    print("Starting up the application...")
    setup_pinecone()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(api_router, prefix="/api")
app.include_router(auth_router, prefix="/api/auth")

@app.get("/")
def read_root():
    with open("static/index.html") as f:
        return Response(content=f.read(), media_type="text/html")

@app.get("/memory")
def read_memory_page():
    with open("static/memory.html") as f:
        return Response(content=f.read(), media_type="text/html")
