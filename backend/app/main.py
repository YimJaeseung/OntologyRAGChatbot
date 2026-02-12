import os
import time
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
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

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = await etl_processor.process_file(content, file.filename)
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