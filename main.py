import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google import genai

# 설정 및 유틸리티 import
from settings import (
    PROJECT_ID,
    LOCATION,
    MODEL,
    GEMINI_API_KEY,
    PORT,
    get_live_api_config,
)
from managers.websocket_manager import ConnectionManager
from managers.session_manager import SessionManager
from auth.websocket_auth import websocket_auth

# --- 클라이언트 초기화 ---
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
# Google AI Studio API Key를 사용하려면:
# client = genai.Client(vertexai=False, api_key=GEMINI_API_KEY)

# --- 전역 변수 ---
connection_manager = ConnectionManager()

async def handle_realtime_session(websocket: WebSocket):
    """실시간 세션 처리 핸들러"""
    # JWT 인증 먼저 수행
    user_id = await websocket_auth.authenticate_websocket(websocket)
    
    await connection_manager.connect(websocket)
    print(f"인증된 클라이언트 연결됨: {websocket.client}, 사용자 ID: {user_id}")
    session_manager = None
    
    try:
        async with (
            client.aio.live.connect(model=MODEL, config=get_live_api_config()) as session,
            asyncio.TaskGroup() as task_group,
        ):
            session_manager = SessionManager(websocket, session, user_id)
            
            # 병렬 태스크 생성
            task_group.create_task(session_manager.receive_client_message())
            task_group.create_task(session_manager.forward_to_gemini())
            task_group.create_task(session_manager.process_gemini_response())

    except WebSocketDisconnect as e:
        print(f"클라이언트 연결 끊김: {websocket.client} (코드: {e.code}, 이유: {e.reason})")
    except Exception as e:
        print(f"처리되지 않은 오류 발생: {e}")
        raise
    finally:
        # 세션 정보 저장
        if session_manager:
            await session_manager.save_session()
        
        # 연결 관리자에서 제거
        connection_manager.disconnect(websocket)
        print(f"남은 클라이언트 수: {len(connection_manager.active_connections)}")

# --- FastAPI 애플리케이션 ---
app = FastAPI(
    title="실시간 음성 채팅 API",
    description="Gemini Live API를 사용한 실시간 음성 채팅 서비스",
    version="1.0.0"
)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # 허용할 출처 리스트
    allow_credentials=True,
    allow_methods=["*"],            # 모든 HTTP 메서드 허용
    allow_headers=["*"],            # 모든 헤더 허용
)

@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "message": "실시간 음성 채팅 API가 실행 중입니다.",
        "connected_clients": len(connection_manager.active_connections),
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "connected_clients": len(connection_manager.active_connections),
        "timestamp": asyncio.get_event_loop().time()
    }

@app.websocket("/ws/realtime")
async def realtime_websocket_endpoint(websocket: WebSocket):
    """실시간 음성 채팅 WebSocket 엔드포인트"""
    try:
        await handle_realtime_session(websocket)
    except Exception as e:
        print(f"WebSocket 오류: {e}")
        if not websocket.client_state.value == 3:  # DISCONNECTED 상태가 아닌 경우만
            await websocket.close(code=1011, reason="Internal server error")

@app.websocket("/ws/test")
async def test_websocket_endpoint(websocket: WebSocket):
    """테스트용 간단한 WebSocket 엔드포인트"""
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await connection_manager.broadcast(f"메시지: {data}")
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        await connection_manager.broadcast("클라이언트가 연결을 끊었습니다.")

if __name__ == "__main__":
    import uvicorn
    
    print(f"서버를 포트 {PORT}에서 시작합니다...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info"
    )