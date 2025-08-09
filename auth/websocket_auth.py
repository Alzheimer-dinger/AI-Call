from fastapi import WebSocket, WebSocketException, status
from urllib.parse import parse_qs
from .jwt_auth import jwt_auth

class WebSocketAuthMiddleware:
    """WebSocket authentication middleware for JWT token validation"""
    
    @staticmethod
    async def authenticate_websocket(websocket: WebSocket) -> str:
        """
        Authenticate WebSocket connection and return user_id
        
        Expects JWT token to be passed as:
        1. Query parameter: ?token=jwt_token
        2. Or in headers (if supported by client)
        
        Returns user_id if authentication successful
        Raises WebSocketException if authentication fails
        """
        
        # Try to get token from query parameters
        token = WebSocketAuthMiddleware._get_token_from_query(websocket)
        
        if not token:
            # Try to get token from headers (if available)
            token = WebSocketAuthMiddleware._get_token_from_headers(websocket)
        
        if not token:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Authentication token required"
            )
        
        # Validate token and extract user_id
        try:
            user_id = jwt_auth.verify_token_and_get_user_id(token)
            return user_id
        except Exception as e:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason=f"Authentication failed: {str(e)}"
            )
    
    @staticmethod
    def _get_token_from_query(websocket: WebSocket) -> str:
        """Extract token from query parameters"""
        query_params = parse_qs(websocket.url.query)
        token_list = query_params.get('token', [])
        return token_list[0] if token_list else None
    
    @staticmethod
    def _get_token_from_headers(websocket: WebSocket) -> str:
        """Extract token from headers"""
        # Try Authorization header
        auth_header = websocket.headers.get('authorization')
        if auth_header:
            return jwt_auth.extract_token_from_header(auth_header)
        
        # Try custom token header
        token_header = websocket.headers.get('x-auth-token')
        if token_header:
            return token_header
        
        return None

# Global middleware instance
websocket_auth = WebSocketAuthMiddleware()