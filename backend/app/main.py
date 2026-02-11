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
# [3] RAG Chat API (Hybrid Search)
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    text: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
    question = req.text
    print(f"💬 Received Question: {question}")

    # 1. OpenSearch Vector Search (유사도 검색)
    vector_results = []
    try:
        query_vec = etl_processor.get_embedding(question)
        os_query = {
            "size": 3,
            "query": {
                "knn": {
                    "vector_field": {
                        "vector": query_vec,
                        "k": 3
                    }
                }
            }
        }
        os_res = etl_processor.os_client.search(index=etl_processor.index_name, body=os_query)
        vector_results = [hit['_source']['text'] for hit in os_res['hits']['hits']]
        print(f"🔍 OpenSearch Found: {len(vector_results)} chunks")
    except Exception as e:
        print(f"⚠️ OpenSearch Error: {e}")

    # 2. TypeDB Graph Search (키워드 기반 연결 탐색)
    graph_results = []
    try:
        # 간단히 질문에 포함된 단어로 엔티티를 찾고, 연결된 텍스트를 조회
        # (실제로는 LLM으로 엔티티를 추출하면 더 정확합니다)
        words = question.split()
        with TypeDB.driver(os.getenv("TYPEDB_ADDRESS", "localhost:1729"), Credentials("admin", "password"), DriverOptions(is_tls_enabled=False)) as driver:
            with driver.transaction("rag_ontology", TransactionType.READ) as tx:
                for word in words:
                    if len(word) < 2: continue
                    # 해당 단어가 이름에 포함된 자산(Asset)과 연결된 텍스트 조회
                    tql = f"""
                    match 
                    $e isa physical-asset, has name $n; 
                    $n contains "{word}";
                    (target: $e, source: $c) isa mention;
                    $c has content-text $text;
                    get $text;
                    """
                    # 환경에 맞춰 tx.query() 함수 호출 방식으로 수정
                    for ans in tx.query(tql):
                        graph_results.append(ans.get("text").as_attribute().get_value())
        print(f"🕸️ TypeDB Found: {len(graph_results)} related chunks")
    except Exception as e:
        print(f"⚠️ TypeDB Search Error: {e}")

    # 3. Context 결합 및 LLM 답변 생성
    context = "\n\n".join(list(set(vector_results + graph_results)))
    
    system_prompt = "You are an industrial AI assistant. Answer based on the context below."
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    
    try:
        response = etl_processor.llm_client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        answer = response.choices[0].message.content
        return {"answer": answer, "context": context}
    except Exception as e:
        return {"answer": "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다.", "error": str(e)}

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