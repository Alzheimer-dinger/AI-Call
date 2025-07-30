
import os
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

# --- 설정 --- #
# TODO: 비밀키 셋업
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "a_very_secret_key_for_jwt_token")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

router = APIRouter()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if token is None:
        return None # WebSocket에서는 헤더 대신 쿼리 파라미터로 토큰을 받을 것이므로, 여기서는 None을 허용
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return {"username": username}
    except JWTError:
        raise credentials_exception

@router.post("/token")
async def login_for_access_token():
    """클라이언트에게 임시 액세스 토큰 발급"""
    # TODO: 인증 구현
    access_token = create_access_token(data={"sub": "guest_user"})
    return {"access_token": access_token, "token_type": "bearer"}
