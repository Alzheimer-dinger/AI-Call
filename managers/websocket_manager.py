from fastapi import WebSocket
from typing import List, Set
import json

class ConnectionManager:
    """WebSocket 연결을 관리하는 클래스"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """새로운 WebSocket 연결을 수락하고 관리 목록에 추가"""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """WebSocket 연결을 관리 목록에서 제거"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """특정 WebSocket에 개인 메시지 전송"""
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        """모든 연결된 WebSocket에 브로드캐스트"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"브로드캐스트 중 오류 발생: {e}")

class PayloadManager:
    """페이로드 관련 유틸리티를 관리하는 클래스"""
    
    @staticmethod
    def to_payload(message_type: str, data) -> str:
        """데이터를 JSON 페이로드로 변환"""
        payload = {
            "type": message_type,
            "data": data
        }
        return json.dumps(payload)

    @staticmethod
    def from_payload(payload: str) -> dict:
        """JSON 페이로드를 파싱"""
        return json.loads(payload)