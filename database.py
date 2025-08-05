import motor.motor_asyncio
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv()

# --- MongoDB 연결 설정 ---
# 실제 환경에서는 환경 변수 등을 사용하여 관리하는 것이 좋습니다.
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "db")

class Database:
    def __init__(self, uri: str, database_name: str):
        # 비동기 클라이언트 생성
        self.client = motor.motor_asyncio.AsyncIOMotorClient(uri)
        # 데이터베이스 가져오기
        self.db = self.client[database_name]
    
    def get_collection(self, collection_name: str):
        """지정된 이름의 컬렉션을 반환합니다."""
        return self.db[collection_name]

# 데이터베이스 인스턴스 생성 (애플리케이션 전역에서 사용)
db = Database(MONGO_CONNECTION_STRING, DB_NAME)

# 대화 기록을 저장할 컬렉션
# 이 conversation_collection 객체를 다른 파일에서 import하여 사용합니다.
transcripts_collection = db.get_collection("transcripts")