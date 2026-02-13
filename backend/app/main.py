import os
import time
import json
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
# [수정] 최신 드라이버 구조에 맞게 import
from typedb.driver import TypeDB, Credentials, DriverOptions, TransactionType
from app.etl import DynamicETL

# ---------------------------------------------------------
# [1] 스키마 초기화 함수 (Startup Logic)
# ---------------------------------------------------------
def initialize_schema():
    uri = os.getenv("TYPEDB_ADDRESS", "localhost:1729")
    db_name = "rag_ontology"
    schema_path = os.getenv("SCHEMA_PATH", "/init_data/schema.tql")

    print(f"🔄 Initializing TypeDB at {uri}...")
    print(f"📂 Loading Schema from: {schema_path}")

    # 인증 정보 및 옵션 설정
    creds = Credentials("admin", "password")
    opts = DriverOptions(is_tls_enabled=False)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            with TypeDB.driver(uri, creds, opts) as driver:
                # 1. 데이터베이스가 아예 없으면 생성
                if not driver.databases.contains(db_name):
                    print(f"✨ Creating database '{db_name}'...")
                    driver.databases.create(db_name)
                
                # [Check] 스키마 존재 여부 확인 (document-file 타입 유무로 판단)
                is_schema_initialized = False
                try:
                    with driver.transaction(db_name, TransactionType.READ) as tx:
                        if tx.concepts.get_entity_type("document-file").resolve():
                            is_schema_initialized = True
                except Exception:
                    pass

                if is_schema_initialized:
                    print(f"✅ Database '{db_name}' and schema already exist. Skipping schema initialization.")
                    break
                
                # 2. 스키마 파일 로드 및 적재 (항상 실행하여 업데이트 반영)
                if os.path.exists(schema_path):
                    print(f"📂 Loading Schema from: {schema_path}")
                    with open(schema_path, "r", encoding="utf-8") as f:
                        schema_query = f.read()
                    
                    # SCHEMA 트랜잭션으로 정의 후 반드시 COMMIT
                    with driver.transaction(db_name, TransactionType.SCHEMA) as tx:
                        tx.query(schema_query)
                        tx.commit()
                        print("✅ Schema applied successfully.")
                else:
                    print(f"❌ CRITICAL: Schema file missing at {schema_path}")
            
            # 연결 및 작업 성공 시 루프 탈출
            break
        except Exception as e:
            print(f"⚠️ Connection failed on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                print("❌ All attempts to connect to TypeDB failed.")
                raise e
            time.sleep(5)
# ---------------------------------------------------------
# [2] Lifespan 및 App 설정
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 LIFESPAN START: Initializing schema...")
    try:
        initialize_schema()
    except Exception as e:
        print(f"❌ CRITICAL ERROR DURING INITIALIZATION: {e}")
        raise e

    # Initialize ETL Processor (OpenSearch Connection) with Retry
    global etl_processor
    print("🔄 Connecting to OpenSearch...")
    for i in range(10):
        try:
            etl_processor = DynamicETL()
            print("✅ OpenSearch Connected.")
            break
        except Exception as e:
            print(f"⚠️ OpenSearch connection failed (Attempt {i+1}/10): {e}")
            if i == 9:
                raise e
            time.sleep(5)
    yield
    print("🚀 LIFESPAN END")
    if etl_processor:
        print("🛑 Closing ETL Processor resources...")
        etl_processor.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ETL 인스턴스 생성
etl_processor = None

# ---------------------------------------------------------
# [WebSocket] Connection Manager
# ---------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        # [수정] 통합 파이프라인 호출 (업로드 -> 추출 -> 스키마 -> 저장)
        result = await etl_processor.process_file_pipeline(content, file.filename)
        return result
    except Exception as e:
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

from app.rag import hybrid_search

class ChatRequest(BaseModel):
    text: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"💬 Received Question: {request.text}")
    # rag.py의 하이브리드 검색 호출
    answer = await hybrid_search(request.text, etl_processor)
    return {"answer": answer}

# ---------------------------------------------------------
# [Admin] 관리자 기능 API
# ---------------------------------------------------------

@app.post("/api/admin/analyze")
async def admin_analyze_file(
    file: UploadFile = File(...), 
    client_id: str = Form(default=""), 
    item_id: str = Form(default="")
):
    """1. 파일 업로드 및 분석 (저장 안함, 미리보기용)"""
    try:
        print(f"🔍 Analyze request received. Client ID: {client_id}, Item ID: {item_id}", flush=True)
        content = await file.read()
        
        # 진행률 콜백 함수 정의
        async def progress_callback(progress: float, message: str):
            msg = json.dumps({
                "type": "progress", "item_id": item_id, 
                "progress": progress, "message": message
            })
            await manager.send_personal_message(msg, client_id)

        result = await etl_processor.preview_file_analysis(content, file.filename, progress_callback)
        return result
    except Exception as e:
        print(f"Analyze Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SaveRequest(BaseModel):
    doc_id: str
    filename: str
    chunks: list
    entities: dict
    relations: list
    links: list

@app.post("/api/admin/save")
async def admin_save_data(data: SaveRequest):
    """2. 검토 완료된 데이터 저장 (스키마 업데이트 + DB 적재)"""
    try:
        result = etl_processor.save_analyzed_data(data.model_dump())
        return result
    except Exception as e:
        print(f"Save Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/schema/update")
async def admin_update_schema(data: dict):
    """3. 스키마만 업데이트 (데이터 적재 X)"""
    try:
        # [Log] 요청 수신 확인
        ent_count = len(data.get('entities', {}))
        rel_count = len(data.get('relations', []))
        print(f"📥 [Schema Update] Received request: {ent_count} entities, {rel_count} relations")

        # data expects {'entities': ..., 'relations': ...}
        return etl_processor.update_schema_only(data.get('entities', {}), data.get('relations', []))
    except Exception as e:
        print(f"Schema Update Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/documents")
async def admin_list_documents():
    """문서 목록 조회"""
    try:
        return etl_processor.list_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/schema")
async def admin_get_schema():
    """현재 스키마 구조 조회"""
    try:
        return etl_processor.get_schema_tree()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/export/json")
async def admin_export_json():
    """지식 그래프 전체 내보내기"""
    try:
        return etl_processor.export_graph_data()
    except Exception as e:
        print(f"Export Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/admin/documents/{doc_id}")
async def admin_delete_document(doc_id: str):
    """3. 문서 삭제"""
    try:
        return etl_processor.delete_document(doc_id)
    except Exception as e:
        print(f"Delete Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))