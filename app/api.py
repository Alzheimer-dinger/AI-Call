import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status, Depends, HTTPException
from app.auth import get_current_user
from app.services.websocket_service import websocket_service
from app.services.rag_service import rag_service
from app.models import UserInfo
from typing import Dict, Any

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

@router.post("/memory")
async def add_memory(
    content: str,
    metadata: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """새로운 기억을 추가합니다."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        await rag_service.add_new_memory(current_user['username'], content, metadata)
        return {"message": "Memory added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add memory: {str(e)}")

@router.get("/memory/search")
async def search_memories(
    query: str,
    top_k: int = 5,
    current_user: dict = Depends(get_current_user)
):
    """기억을 검색합니다."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        from app.services.memory_service import memory_service
        memories = memory_service.retrieve_memories(query, top_k, current_user['username'])

        # 결과를 직렬화 가능한 형태로 변환 (스코어 기준으로 정렬)
        serializable_memories = []
        for memory in memories:
            # 스코어가 임계값 이상인 것만 포함
            if memory.score > 0.5:  # API에서도 임계값 적용
                serializable_memories.append({
                    "score": memory.score,
                    "content": memory.metadata.get('content', ''),
                    "category": memory.metadata.get('category', ''),
                    "date": memory.metadata.get('date', ''),
                    "importance": memory.metadata.get('importance', 'medium')
                })

        # 스코어 순으로 정렬 (높은 것부터)
        serializable_memories.sort(key=lambda x: x['score'], reverse=True)
        
        return {"memories": serializable_memories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search memories: {str(e)}")

@router.post("/memory/samples")
async def add_sample_memories(
    current_user: dict = Depends(get_current_user)
):
    """현재 사용자를 위해 샘플 기억을 추가합니다."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        await rag_service.add_sample_memories(current_user['username'])
        return {"message": "Sample memories added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add sample memories: {str(e)}")

@router.get("/conversation/history")
async def get_conversation_history(
    current_user: dict = Depends(get_current_user)
):
    """대화 기록을 조회합니다."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    user_id = current_user['username']
    history = rag_service.conversation_history.get(user_id, [])
    return {"history": history}
