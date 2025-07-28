import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status
from app.auth import get_current_user
from app.services.websocket_service import websocket_service

router = APIRouter()

@router.websocket("/ws/live")
async def live_audio_endpoint(websocket: WebSocket, token: str = Query(...)):
    """클라이언트에서 직접 변환된 PCM 데이터를 Gemini Live로 중계합니다."""
    user = await get_current_user(token)
    if user is None:
        return await websocket.close(code=status.WS_1008_POLICY_VIOLATION)

    await websocket.accept()
    print(f"WebSocket accepted for {user['username']}")

    try:
        await websocket_service.handle_live_audio_session(websocket, user)
    except Exception as e:
        print(f"An unexpected error occurred in live endpoint: {e}")
    finally:
        print("Live audio endpoint cleanup complete.")
