import jwt
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from settings import JWT_SECRET_KEY

class JWTAuth:
    def __init__(self):
        self.secret_key = JWT_SECRET_KEY
        self.algorithm = "HS256"
    
    def validate_token(self, token: str) -> bool:
        """Validate JWT token"""
        try:
            jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return True
        except jwt.InvalidTokenError:
            return False
    
    def get_token_payload(self, token: str) -> Optional[Dict[str, Any]]:
        """Get token payload if valid"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.InvalidTokenError:
            return None
    
    def get_user_id(self, token: str) -> Optional[str]:
        """Extract user ID from token"""
        payload = self.get_token_payload(token)
        return payload.get("id") if payload else None
    
    def get_role(self, token: str) -> Optional[str]:
        """Extract role from token"""
        payload = self.get_token_payload(token)
        return payload.get("role") if payload else None
    
    def is_access_token(self, token: str) -> bool:
        """Check if token is an access token"""
        payload = self.get_token_payload(token)
        return payload and payload.get("sub") == "AccessToken"
    
    def extract_token_from_header(self, authorization: str) -> Optional[str]:
        """Extract token from Authorization header"""
        if not authorization:
            return None
        
        if authorization.startswith("Bearer "):
            return authorization[7:]  # Remove "Bearer " prefix
        
        return None
    
    def verify_token_and_get_user_id(self, token: str) -> str:
        """Verify token and return user ID, raise exception if invalid"""
        if not self.validate_token(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        if not self.is_access_token(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        user_id = self.get_user_id(token)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        return user_id

# Global JWT auth instance
jwt_auth = JWTAuth()