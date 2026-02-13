import os
import uuid
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Optional

# TypeDB 3.7 호환 임포트
from typedb.driver import TypeDB, TransactionType, Credentials, DriverOptions
from opensearchpy import OpenSearch
from openai import OpenAI

# [분리된 모듈 임포트]
from app.schema import SchemaManager
from app.parser import parse_file_content

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
        self.schema_mgr = SchemaManager(self.driver, self.db_name)
        
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
        valid_types = ", ".join(self.schema_mgr.valid_parents)
        prompt = f"""
        Extract industrial knowledge from the text.
        Identify specific entity types (L3) and their parent categories (L2).
        Identify relationships between entities (e.g., connection, part-of, location).
        Parent categories (L2) should be one of: [{valid_types}].
        
        [Constraints]
        - Do NOT extract 'date', 'time', 'level', 'status', 'description' as Entity Types. These are attributes.
        - Do NOT create generic types like 'site-equipment', 'unnamed-level'. Use specific types.
        - 'sub-project' should be classified as 'project'.

        Return ONLY a JSON object with this structure:
        {{
          "entities": [{{ "name": "Pump A", "type": "centrifugal-pump", "parent_type": "equipment" }}],
          "relations": [{{ "from": "Pump A", "to": "System B", "type": "assembly" }}]
        }}
        Text: "{text[:1000]}"
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

    async def process_file_pipeline(self, file_content: bytes, filename: str):
        """
        [통합 파이프라인]
        1. 파일 파싱 & 청킹 (In-Memory)
        2. 지식 추출 (LLM Analysis)
        3. 스키마 업데이트 (Schema Transaction)
        4. 데이터 적재 (Write Transaction)
        """
        print(f"📂 Processing file: {filename}")
        doc_id = str(uuid.uuid4())
        
        # Step 1: 파일 파싱 및 청킹
        raw_chunks = parse_file_content(file_content, filename)
        
        # [OPTIMIZATION] Create embeddings in parallel
        async def create_embedding_task(chunk_text):
            return await asyncio.to_thread(self.get_embedding, chunk_text)

        embedding_tasks = [create_embedding_task(rc['text']) for rc in raw_chunks]
        vectors = await asyncio.gather(*embedding_tasks)
        
        # 청크에 ID 부여 및 임베딩 생성 (Enrichment)
        chunks = []
        for i, rc in enumerate(raw_chunks):
            chunk_id = f"{doc_id}_{'r' if rc['type']=='table-row' else 'c'}{rc['index']}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": rc['text'],
                "type": rc['type'],
                "vector": vectors[i]
            })
            
        print(f"✅ Step 1: Parsed {len(chunks)} chunks.")

        # Step 2: 청크별 지식 추출 (메모리 상에서 수행)
        extracted_data = await self._analyze_chunks(chunks)
        print(f"✅ Step 2: Extracted {len(extracted_data['entities'])} entities.")

        # Step 3: 스키마 업데이트
        self._update_schema_definitions(extracted_data['entities'], extracted_data['relations'])
        print(f"✅ Step 3: Schema updated.")

        # Step 4: DB 적재 (TypeDB + OpenSearch)
        self._save_to_db(doc_id, filename, chunks, extracted_data)
        print(f"✅ Step 4: Data saved to DB.")

        return {"status": "success", "doc_id": doc_id, "chunks": len(chunks), "entities": len(extracted_data['entities'])}

    async def preview_file_analysis(self, file_content: bytes, filename: str):
        """[Admin] 1단계: 파일 파싱 및 지식 추출 (DB 저장 X)"""
        doc_id = str(uuid.uuid4())
        
        # 1. 파싱 및 청킹
        raw_chunks = parse_file_content(file_content, filename)

        # [OPTIMIZATION] Create embeddings in parallel
        async def create_embedding_task(chunk_text):
            return await asyncio.to_thread(self.get_embedding, chunk_text)

        embedding_tasks = [create_embedding_task(rc['text']) for rc in raw_chunks]
        vectors = await asyncio.gather(*embedding_tasks)

        chunks = []
        for i, rc in enumerate(raw_chunks):
            chunk_id = f"{doc_id}_{'r' if rc['type']=='table-row' else 'c'}{rc['index']}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": rc['text'],
                "type": rc['type'],
                "vector": vectors[i] # 벡터 생성은 미리 수행
            })

        # 2. 지식 추출
        extracted_data = await self._analyze_chunks(chunks)
        
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": chunks,
            "entities": extracted_data['entities'],
            "relations": extracted_data['relations'],
            "links": extracted_data['links']
        }

    def save_analyzed_data(self, data: Dict):
        """[Admin] 2단계: 검토된 데이터 스키마 반영 및 DB 저장"""
        # 1. 스키마 업데이트
        self._update_schema_definitions(data['entities'], data.get('relations', []))
        
        # 2. 데이터 구조 재조립 (save_to_db 호환)
        extracted_data = {
            "entities": data['entities'],
            "relations": data['relations'],
            "links": data['links']
        }
        
        # 3. DB 저장
        self._save_to_db(
            doc_id=data['doc_id'], 
            filename=data['filename'], 
            chunks=data['chunks'], 
            extracted_data=extracted_data
        )
        return {"status": "saved", "doc_id": data['doc_id']}

    async def _analyze_chunks(self, chunks: List[Dict]) -> Dict:
        """각 청크에 대해 LLM을 호출하여 엔티티와 관계를 추출"""
        
        # [OPTIMIZATION] Run graph extraction in parallel
        async def extract_task(chunk):
            graph_data = await asyncio.to_thread(self.extract_graph_data, chunk['text'])
            return chunk['chunk_id'], graph_data

        tasks = [extract_task(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        all_entities = {} # name -> {type, parent}
        all_relations = []
        chunk_links = [] # (chunk_id, entity_name)

        for chunk_id, graph_data in results:
            for ent in graph_data.get("entities", []):
                name = ent.get('name')
                if not name: continue
                
                etype = ent.get('type') or "unknown-entity"
                
                # [Filter] 속성(Attribute) 성격의 데이터가 엔티티로 추출되는 것 방지
                if etype.lower() in {"date", "datetime", "time", "status", "description", "comment", "note", "unknown", "level", "alarm-level", "unnamed-level", "site-equipment"}:
                    continue

                parent = ent.get("parent_type") or "physical-asset"
                all_entities[name] = {"type": etype, "parent": parent}
                chunk_links.append((chunk_id, name))
            
            for rel in graph_data.get("relations", []):
                all_relations.append(rel)
        
        return {
            "entities": all_entities,
            "relations": all_relations,
            "links": chunk_links
        }

    def _update_schema_definitions(self, entities: Dict, relations: List[Dict] = None):
        """추출된 엔티티 타입을 확인하고 필요 시 스키마 업데이트"""
        # 1. 엔티티 타입 정의
        for name, info in entities.items():
            # [Fix] Update type with the actual sanitized/renamed type returned by schema manager
            info['type'] = self.schema_mgr.ensure_l3_type(info['type'], info['parent'])
            
        # 2. 관계 타입 정의
        if relations:
            for rel in relations:
                from_name = rel.get('from')
                to_name = rel.get('to')
                rel_type = rel.get('type')
                
                # 엔티티 목록에서 타입 조회
                from_type = entities.get(from_name, {}).get('type')
                to_type = entities.get(to_name, {}).get('type')
                
                if from_type and to_type and rel_type:
                    final_rel_type = self.schema_mgr.ensure_relation_type(rel_type, from_type, to_type)
                    if final_rel_type:
                        rel['type'] = final_rel_type # Update with sanitized/renamed type

    def _save_to_db(self, doc_id: str, filename: str, chunks: List[Dict], extracted_data: Dict):
        """TypeDB와 OpenSearch에 최종 데이터 적재"""
        now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        
        # [Fix] 파일명 내 백슬래시 및 따옴표 이스케이프 처리
        safe_filename = filename.replace('\\', '\\\\').replace("'", "\\'")

        with self.driver.transaction(self.db_name, TransactionType.WRITE) as tx:
            # 1. 문서(Document) 생성
            tx.query(f"insert $d isa document-file, has id-doc-id '{doc_id}', has name '{safe_filename}', has created-date {now};")

            # 2. 청크(Content Unit) 생성 및 문서 연결
            for chunk in chunks:
                # [Fix] 백슬래시 이스케이프 후, 큰따옴표를 작은따옴표로 치환 (TQL 문자열 파싱 오류 방지)
                safe_text = chunk['text'].replace('\\', '\\\\').replace('"', "'")
                # TypeDB 적재
                q_chunk = f"""
                match $d isa document-file, has id-doc-id "{doc_id}";
                insert $c isa {chunk['type']}, has id-chunk-id "{chunk['chunk_id']}", 
                has content-text "{safe_text}", has created-date {now};
                (container: $d, content: $c) isa containment;
                """
                tx.query(q_chunk)
                
                # OpenSearch 적재
                self.insert_to_opensearch(chunk['chunk_id'], chunk['text'], chunk['vector'], {"doc_id": doc_id})

            # 엔티티 생성
            for name, info in extracted_data['entities'].items():
                safe_name = name.replace('\\', '\\\\').replace('"', "'")
                etype = info['type']
                # 존재 확인 후 생성
                q_check = tx.query(f'match $e isa {etype}, has name "{safe_name}"; fetch {{ "id": $e }};')
                if hasattr(q_check, 'resolve'): q_check = q_check.resolve()
                if not list(q_check):
                    tx.query(f'insert $e isa {etype}, has name "{safe_name}";')

            # 청크 연결 (Mention)
            for cid, name in extracted_data['links']:
                safe_name = name.replace('\\', '\\\\').replace('"', "'")
                etype = extracted_data['entities'][name]['type']
                q_link = f"""
                match $c isa content-unit, has id-chunk-id "{cid}";
                      $e isa {etype}, has name "{safe_name}";
                insert (source: $c, target: $e) isa mention;
                """
                try: tx.query(q_link)
                except: pass

            # 관계 생성
            for rel in extracted_data['relations']:
                rtype = rel['type']
                fname = rel['from'].replace('\\', '\\\\').replace('"', "'")
                tname = rel['to'].replace('\\', '\\\\').replace('"', "'")
                
                # [Fix] 관계 저장 로직 유연화 (여러 역할 패턴 시도)
                queries = []
                
                # Case 1: Assembly / Part-of (part, system)
                if rtype in ['part-of', 'assembly', 'composition']:
                    rtype = 'assembly'
                    queries.append(f'match $f has name "{fname}"; $t has name "{tname}"; insert (part: $f, system: $t) isa {rtype};')
                
                # Case 2: Location (located, location)
                if rtype == 'location':
                    queries.append(f'match $f has name "{fname}"; $t has name "{tname}"; insert (located: $f, place: $t) isa {rtype};')

                # Case 3: Generic Connection (source, target) - Default fallback
                queries.append(f'match $f has name "{fname}"; $t has name "{tname}"; insert (source: $f, target: $t) isa {rtype};')
                
                for q in queries:
                    try: 
                        tx.query(q)
                        break # 성공하면 루프 종료
                    except: 
                        pass # 실패하면 다음 패턴 시도
            
            tx.commit()

    def delete_document(self, doc_id: str):
        """[Admin] 문서 및 관련 데이터 삭제"""
        # 1. TypeDB 삭제 (문서 + 포함된 청크)
        # 주의: 연결된 엔티티(장비 등)는 다른 문서에서도 쓸 수 있으므로 삭제하지 않음
        with self.driver.transaction(self.db_name, TransactionType.WRITE) as tx:
            q_del = f"""
            match $d isa document-file, has id-doc-id "{doc_id}";
            (container: $d, content: $c) isa containment;
            delete $d, $c;
            """
            tx.query(q_del)
            tx.commit()

        # 2. OpenSearch 삭제
        query = {
            "query": {
                "term": {
                    "metadata.doc_id.keyword": doc_id
                }
            }
        }
        self.os_client.delete_by_query(index=self.index_name, body=query)
        return {"status": "deleted", "doc_id": doc_id}

    def list_documents(self):
        """[Admin] 저장된 문서 목록 조회"""
        docs = []
        try:
            with self.driver.transaction(self.db_name, TransactionType.READ) as tx:
                q = 'match $d isa document-file, has name $n, has id-doc-id $id, has created-date $date; fetch { "id": $id, "name": $n, "date": $date };'
                results = tx.query(q)
                if hasattr(results, 'resolve'): results = results.resolve()
                for res in results:
                    # TypeDBJSON is dict-like, and fetch with JSON structure returns primitive values.
                    doc_id = res.get("id")
                    name = res.get("name")
                    date = res.get("date")
                    if doc_id:
                        # The date is a datetime object, so we convert it to a string for JSON serialization.
                        docs.append({"id": doc_id, "name": name, "date": str(date)})
            
            print(f"📄 Listed {len(docs)} documents.")
        except Exception as e:
            print(f"⚠️ Error listing documents: {e}")
            # 에러 발생 시 빈 리스트 반환하여 프론트엔드 멈춤 방지
        return docs

    def get_schema_tree(self):
        return self.schema_mgr.get_schema_tree()

    def update_schema_only(self, entities: Dict, relations: List[Dict] = None):
        """[Schema Phase] 데이터 저장 없이 스키마만 업데이트"""
        print("🔄 Starting schema update process...")
        self._update_schema_definitions(entities, relations)
        print("✅ Schema update process completed.")
        return {"status": "schema_updated", "entity_count": len(entities), "relation_count": len(relations or [])}

    def export_graph_data(self) -> Dict:
        """TypeDB의 스키마 구조(Ontology)를 JSON으로 내보내기"""
        # get_schema_tree()는 이제 엔티티와 관계를 모두 포함하는 전체 스키마를 반환합니다.
        schema_tree = self.schema_mgr.get_schema_tree()

        # API 계약을 유지하기 위해 'entities'와 'relations' 키로 분리합니다.
        relations = schema_tree.pop("relations", [])
        entities_hierarchy = schema_tree

        return {
            "entities": entities_hierarchy,
            "relations": relations # get_schema_tree에서 이미 정렬됨
        }