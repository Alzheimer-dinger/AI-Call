from .memory_service import memory_service

def setup_pinecone():
    memory_service.setup_pinecone()

def get_embedding(text: str):
    return memory_service.get_embedding(text)

def retrieve_memories(query: str, top_k: int = 3):
    retrieved = memory_service.retrieve_memories(query, top_k)
    # 기존 형식으로 변환
    return [{"score": mem.score, "metadata": mem.metadata} for mem in retrieved]

