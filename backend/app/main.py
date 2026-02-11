import os
import time
from contextlib import asynccontextmanager
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
            # [3.7 방식] TypeDB.driver로 연결
            with TypeDB.driver(uri, creds, opts) as driver:
                    # 1. 데이터베이스 존재 여부 확인
                    if not driver.databases.contains(db_name):
                        print(f"✨ First time setup: Creating database '{db_name}'...")
                        driver.databases.create(db_name)
                        
                        # 2. DB가 새로 생성된 경우에만 스키마 적재
                        if os.path.exists(schema_path):
                            with open(schema_path, "r") as f:
                                schema_query = f.read()
                            
                            with driver.transaction(db_name, TransactionType.SCHEMA) as tx:
                                tx.query(schema_query)
                                tx.commit()
                            print("✅ Schema initialized successfully for the first time!")
                            return
                    else:
                        # DB가 이미 존재하면 스키마 적재 과정을 건너뜁니다.
                        print(f"📚 Database '{db_name}' already exists. Skipping schema initialization.")
                        return
        except Exception as e:
            print(f"⏳ Connection failed (Attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)

    print("❌ Failed to initialize schema after multiple attempts.")

# ---------------------------------------------------------
# [2] Lifespan 및 App 설정
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # [확인] 이 로그가 docker logs에 찍히는지 보세요
    print("🚀 LIFESPAN START: Initializing schema...")
    try:
        initialize_schema()
    except Exception as e:
        print(f"❌ CRITICAL ERROR DURING INITIALIZATION: {e}")
    yield
    print("🚀 LIFESPAN END")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ETL 인스턴스 생성
etl_processor = DynamicETL()

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