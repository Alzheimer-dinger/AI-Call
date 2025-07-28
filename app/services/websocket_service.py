import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from app.config import settings
from app.models import UserInfo

class WebSocketService:
    def __init__(self):
        self.client = genai.Client()
        self.model = settings.LIVE_MODEL
        self.config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": settings.SYSTEM_INSTRUCTION,
        }
        self.target_sample_rate = settings.TARGET_SAMPLE_RATE

    async def handle_live_audio_session(self, websocket: WebSocket, user: dict):
        """Gemini Live와의 오디오 세션을 처리합니다."""
        try:
            async with self.client.aio.live.connect(model=self.model, config=self.config) as session:
                print(f"Persistent Gemini Live session established for {user['username']}")

                tasks = [
                    asyncio.create_task(self._forward_to_gemini(websocket, session)),
                    asyncio.create_task(self._forward_to_client(websocket, session))
                ]

                try:
                    await asyncio.gather(*tasks)
                except (WebSocketDisconnect, Exception) as e:
                    print(f"Task interrupted: {e}")
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            print(f"An unexpected error occurred in live session: {e}")
            raise

    async def _forward_to_gemini(self, websocket: WebSocket, session):
        """클라이언트로부터 PCM 데이터를 받아 Gemini로 전송"""
        try:
            while True:
                pcm_data = await websocket.receive_bytes()
                print(f"Received {len(pcm_data)} bytes from client")

                blob = types.Blob(
                    data=pcm_data,
                    mime_type=f"audio/pcm;rate={self.target_sample_rate}"
                )
                await session.send_realtime_input(audio=blob)

        except WebSocketDisconnect:
            print("Client disconnected during audio forwarding.")
            raise
        except Exception as e:
            print(f"Error forwarding to Gemini: {e}")
            raise

    async def _forward_to_client(self, websocket: WebSocket, session):
        """Gemini 응답을 클라이언트로 전송"""
        try:
            while True:
                try:
                    async for response in session.receive():
                        if response.data:
                            print(f"Sending {len(response.data)} bytes to client")
                            await websocket.send_bytes(response.data)
                except Exception as e:
                    print(f"Error in response iteration: {e}")
                    await asyncio.sleep(0.1)
                    continue

                print("Response stream ended, waiting for next response...")
                await asyncio.sleep(0.1)

        except WebSocketDisconnect:
            print("Client disconnected during response forwarding.")
            raise
        except Exception as e:
            print(f"Critical error in forward_to_client: {e}")
            raise

websocket_service = WebSocketService()

