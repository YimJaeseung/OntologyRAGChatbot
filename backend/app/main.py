from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# RAG 엔진 로직 (간소화)
app = FastAPI()

# CORS: 프론트엔드(React) 접근 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryReq(BaseModel):
    text: str

@app.post("/api/chat")
async def chat(req: QueryReq):
    # 여기에 실제 RAG 로직 연결 (TypeDB + OpenSearch)
    # 현재는 연결 테스트용 응답 반환
    user_query = req.text
    
    return {
        "answer": f"Backend received: '{user_query}'.\nSystem is ready with TypeDB & OpenSearch."
    }