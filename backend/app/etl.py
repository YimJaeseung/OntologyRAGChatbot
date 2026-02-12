import os
import re
import uuid
import json
import pandas as pd
import pdfplumber
from datetime import datetime
from typing import List, Dict, Optional

# TypeDB 3.7 호환 임포트
from typedb.driver import TypeDB, TransactionType, Credentials, DriverOptions
from opensearchpy import OpenSearch
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# 1. Schema Manager: 동적으로 L3 타입을 관리하는 클래스
# ---------------------------------------------------------

class SchemaManager:
    def __init__(self, uri, db_name):
        self.uri = uri
        self.db_name = db_name
        # 인증 정보 및 옵션 설정
        self.creds = Credentials("admin", "password")
        self.opts = DriverOptions(is_tls_enabled=False)
        
        # schemal.tql에 정의된 L2 Entity 목록
        self.valid_parents = {
            "equipment", "component", "sensor", "site", "zone",
            "document-file", "content-unit",
            "engineer", "operator", "manager",
            "fault", "alarm", "maintenance-activity"
        }
        self._known_types = set(self.valid_parents)

    def sanitize_type_name(self, name: str) -> str:
        slug = name.lower()
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'\s+', '-', slug)
        return slug

    def ensure_l3_type(self, l3_name: str, l2_parent: str) -> str:
        slug_l3 = self.sanitize_type_name(l3_name)
        
        if slug_l3 in self._known_types or slug_l3 == l2_parent:
            return slug_l3

        if l2_parent not in self.valid_parents:
            print(f"⚠️ Invalid parent '{l2_parent}'. Fallback to 'document-file'")
            l2_parent = "document-file"

        # TypeDB 3.7 표준: driver -> transaction
        with TypeDB.driver(self.uri, self.creds, self.opts) as driver:
            # 1. 존재 확인 (쿼리 방식)
            with driver.transaction(self.db_name, TransactionType.READ) as tx:
                try:
                    # 해당 타입이 존재하는지 concepts API로 확인
                    if tx.concepts.get_entity_type(slug_l3).resolve():
                        return slug_l3
                except Exception:
                    pass # 타입이 없으면 아래 정의 로직으로 이동

            # 2. 없으면 정의 (SCHEMA 트랜잭션)
            print(f"🆕 Defining New L3 Type: '{slug_l3}' (sub {l2_parent})")
            try:
                with driver.transaction(self.db_name, TransactionType.SCHEMA) as tx:
                    define_query = f"define entity {slug_l3}, sub {l2_parent};"
                    tx.query(define_query)
                    tx.commit()
                self._known_types.add(slug_l3)
                return slug_l3
            except Exception as e:
                print(f"⚠️ Failed to define type: {e}. Fallback to {l2_parent}")
                return l2_parent

# ---------------------------------------------------------
# 2. Dynamic ETL: 파일 처리 및 데이터 적재
# ---------------------------------------------------------
class DynamicETL:
    def __init__(self):
        self.typedb_uri = os.getenv("TYPEDB_ADDRESS", "localhost:1729")
        self.db_name = os.getenv("TYPEDB_DATABASE", "rag_ontology")
        self.creds = Credentials("admin", "password")
        self.opts = DriverOptions(is_tls_enabled=False)
        
        # [최적화] TypeDB 드라이버를 인스턴스 변수로 유지하여 재사용
        self.driver = TypeDB.driver(self.typedb_uri, self.creds, self.opts)
        
        self.os_client = OpenSearch(
            hosts=[os.getenv("OPENSEARCH_URL", "http://localhost:9200")],
            http_auth=None, use_ssl=False
        )
        self.index_name = "rag-docs"
        self.llm_client = OpenAI(
            base_url=os.getenv("VLLM_API_URL", "http://100.111.233.70:8000/v1"),
            api_key="EMPTY"
        )
        # SchemaManager 생성 시 올바른 변수 전달
        self.schema_mgr = SchemaManager(self.typedb_uri, self.db_name)
        
        # OpenSearch 인덱스 초기화 (Mapping 설정)
        self._initialize_index()

    def _initialize_index(self):
        if not self.os_client.indices.exists(index=self.index_name):
            print(f"⚙️ Creating OpenSearch index '{self.index_name}' with k-NN mapping...")
            body = {
                "settings": {"index.knn": True},
                "mappings": {
                    "properties": {
                        "vector_field": {
                            "type": "knn_vector",
                            "dimension": 1536,  # text-embedding-3-small dimension
                            "method": {"name": "hnsw", "engine": "nmslib"}
                        },
                        "text": {"type": "text"},
                        "chunk_id": {"type": "keyword"}
                    }
                }
            }
            self.os_client.indices.create(index=self.index_name, body=body)
            print("✅ OpenSearch index created.")

    def close(self):
        """리소스 해제"""
        self.driver.close()
        self.os_client.close()

    def get_embedding(self, text: str) -> List[float]:
        try:
            return self.llm_client.embeddings.create(
                input=[text.replace("\n", " ")], 
                model="text-embedding-3-small"
            ).data[0].embedding
        except:
            return [0.0] * 1536 

    def analyze_document_type(self, text_snippet: str) -> dict:
        valid_parents_str = ", ".join(self.schema_mgr.valid_parents)
        prompt = f"""
        Analyze the text snippet from an industrial document.
        Determine the specific 'L3 Type' and its 'L2 Parent' from this list: [{valid_parents_str}].
        Snippet: "{text_snippet[:300]}..."
        Return JSON: {{"l3_name": "string", "l2_parent": "string"}}
        """
        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = response.choices[0].message.content
            clean_json = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except:
            return {"l3_name": "General Doc", "l2_parent": "document-file"}

    def insert_to_typedb(self, tql_query):
        # TypeDB 3.7 표준: driver -> transaction
        with self.driver.transaction(self.db_name, TransactionType.WRITE) as tx:
            tx.query(tql_query)
            tx.commit()

    def insert_to_opensearch(self, chunk_id, text, vector, metadata):
        doc = {
            "chunk_id": chunk_id, "text": text, "vector_field": vector,
            "metadata": metadata, "timestamp": datetime.now()
        }
        self.os_client.index(index=self.index_name, body=doc, id=chunk_id)



    def extract_graph_data(self, text: str) -> Dict:
        """[Level 3] LLM을 통한 엔티티 및 관계 추출"""
        prompt = f"""
        Extract industrial knowledge from the text.
        Return ONLY a JSON object with this structure:
        {{
          "entities": [{{ "name": "Pump A", "type": "equipment" }}],
          "relations": [{{ "from": "Pump A", "to": "System B", "type": "part-of" }}]
        }}
        Text: "{text[:600]}"
        """
        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={ "type": "json_object" }
            )
            return json.loads(response.choices[0].message.content)
        except:
            return {"entities": [], "relations": []}

    def write_graph_to_typedb(self, tx, chunk_id, graph_data):
        """추출된 그래프 데이터를 TypeDB에 적재"""
        # 1. 엔티티 생성 및 청크와 연결 (mention)
        for ent in graph_data.get("entities", []):
            name = ent['name'].replace('"', "'")
            ent_type = ent['type']
            
            # 1. 엔티티 존재 확인
            check_ent = list(tx.query(f'match $e isa {ent_type}, has name "{name}"; get;'))
            
            # 2. 없으면 생성
            if not check_ent:
                tx.query(f'insert $e isa {ent_type}, has name "{name}";')
            
            # 3. 관계 연결 
            link_query = f"""
            match $c isa content-unit, has id-chunk-id "{chunk_id}";
                  $e isa {ent_type}, has name "{name}";
            insert (source: $c, target: $e) isa mention;
            """
            tx.query(link_query)

        # 2. 관계 생성 (part-of, monitors 등)
        for rel in graph_data.get("relations", []):
            rel_type = rel['type'] 
            from_name = rel['from'].replace('"', "'")
            to_name = rel['to'].replace('"', "'")
            
            # schema.tql의 관계 정의에 맞춰 role을 매핑해야 함 (예: assembly)
            # 여기서는 범용적으로 source/target 혹은 parent/child 관계를 시도
            query = f"""
            match 
                $f isa physical-asset, has name "{from_name}";
                $t isa physical-asset, has name "{to_name}";
            insert 
                (child: $f, parent: $t) isa {rel_type};
            """
            try: tx.query(query)
            except: pass

    async def process_file(self, file_content: bytes, filename: str):
        print(f"📂 Processing file: {filename}")
        doc_id = str(uuid.uuid4())
        is_excel = filename.endswith(".xlsx") or filename.endswith(".xls")
        
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "wb") as f:
            f.write(file_content)

        # 텍스트 추출 부분 (기존과 동일)
        if is_excel:
            df = pd.read_excel(temp_path).fillna("")
            snippet = df.head(5).to_string()
            full_text = df.to_string()
        else:
            with pdfplumber.open(temp_path) as pdf:
                full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            snippet = full_text[:1000]

        # 타입 분석 및 문서 엔티티 생성 (기존과 동일)
        analysis = self.analyze_document_type(snippet)
        analyzed_type = self.schema_mgr.ensure_l3_type(analysis.get("l3_name"), analysis.get("l2_parent"))
        
        now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        tql_doc = f"insert $d isa {analyzed_type}, has id-doc-id '{doc_id}', has title '{filename}', has created-date {now};"
        self.insert_to_typedb(tql_doc)

        # --- 핵심 수정: 트랜잭션을 열고 루프 내에서 그래프 추출 수행 ---
        with self.driver.transaction(self.db_name, TransactionType.WRITE) as tx:
                if is_excel and df is not None:
                    for idx, row in df.iterrows():
                        row_id = f"{doc_id}_r{idx}"
                        row_json = row.to_json(force_ascii=False)
                        
                        # 1. 데이터 적재
                        tx.query(f"""
                            match $d isa {analyzed_type}, has id-doc-id "{doc_id}";
                            insert $r isa table-row, has id-chunk-id "{row_id}", 
                            has content-text "{str(row_json).replace('"', "'")}", 
                            has row-index {idx}, has created-date {now};
                            (container: $d, content: $r) isa containment;
                        """)
                        # 2. 그래프 추출 및 연결 (Level 3)
                        graph_data = self.extract_graph_data(row_json)
                        self.write_graph_to_typedb(tx, row_id, graph_data)
                        
                        # 벡터 DB 동기화 (기존)
                        vector = self.get_embedding(row_json)
                        self.insert_to_opensearch(row_id, row_json, vector, {"doc_id": doc_id})

                else:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
                    for idx, text in enumerate(splitter.split_text(full_text)):
                        chunk_id = f"{doc_id}_c{idx}"
                        
                        # 1. 청크 적재
                        tx.query(f"""
                            match $d isa {analyzed_type}, has id-doc-id "{doc_id}";
                            insert $c isa text-chunk, has id-chunk-id "{chunk_id}", 
                            has content-text "{text.replace('"', "'")}", 
                            has page-number {idx}, has created-date {now};
                            (container: $d, content: $c) isa containment;
                        """)
                        # 2. 그래프 추출 및 연결 (Level 3)
                        graph_data = self.extract_graph_data(text)
                        self.write_graph_to_typedb(tx, chunk_id, graph_data)
                        
                        # 벡터 DB 동기화
                        vector = self.get_embedding(text)
                        self.insert_to_opensearch(chunk_id, text, vector, {"doc_id": doc_id})
                
                tx.commit() # 모든 청크와 추출된 지식을 한 번에 커밋

        os.remove(temp_path)
        return {"status": "success", "doc_id": doc_id}