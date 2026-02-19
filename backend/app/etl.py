import os
import uuid
import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Optional

# TypeDB 3.7 호환 임포트
from typedb.driver import TypeDB, TransactionType, Credentials, DriverOptions
from opensearchpy import OpenSearch
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError

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
        self.llm_client = AsyncOpenAI(
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

    async def get_embedding(self, text: str) -> List[float]:
        try:
            response = await self.llm_client.embeddings.create(
                input=[text.replace("\n", " ")], 
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
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



    async def extract_graph_data(self, text: str) -> Dict:
        """[Level 3] LLM을 통한 엔티티 및 관계 추출"""
        valid_types = ", ".join(self.schema_mgr.valid_parents)
        
        # [Fix] System Prompt 분리 및 스키마 명시
        system_prompt = """
        You are an expert Industrial Knowledge Graph Engineer.
        Extract structured knowledge from the text into a JSON format.

        Output JSON Schema:
        {
          "entities": [ { "name": "string", "type": "string", "parent_type": "string" } ],
          "relations": [ { "from": "string", "to": "string", "type": "string" } ]
        }
        
        IMPORTANT: 'entities' must be a list of OBJECTS (dictionaries), NOT a list of lists.
        """

        # [Prompt Engineering] 구성 요소 분리 (재시도 시 단순화를 위해)
        definitions = f"""
        [Definitions]
        - **Equipment**: Physical machines and devices (e.g., Pump, Motor, Robot).
        - **Component**: Parts belonging to equipment (e.g., Bearing, Valve, Cable).
        - **Site**: Physical locations (e.g., Factory, Zone, Room).
        - **Operator**: People or teams who operate equipment.
        - **Manager**: People or departments responsible for sites or projects.
        - **Fault**: A malfunction or defect in a component or equipment.
        - **Alarm**: A signal or warning about a fault or an abnormal condition.

        [Rules]
        1. **Entities**: Identify specific L3 types and their L2 parent from: [{valid_types}].
           - Ignore attributes like dates, IDs, status, or generic terms (e.g., "item", "part").
        2. **Relations**: Identify connections like 'assembly' (part-of), 'location' (at), 'responsibility' (by), 'connection'.
        """
        
        example = """
        
        [Example]
        Input: "The Centrifugal Pump (P-101) in Zone A was inspected by the Maintenance Team. Found a crack in the seal."
        Output: {{
          "entities": [
            {{ "name": "P-101", "type": "centrifugal-pump", "parent_type": "equipment" }},
            {{ "name": "Zone A", "type": "zone", "parent_type": "site" }},
            {{ "name": "Maintenance Team", "type": "team", "parent_type": "operator" }},
            {{ "name": "seal", "type": "seal", "parent_type": "component" }},
            {{ "name": "crack", "type": "crack", "parent_type": "fault" }}
          ],
          "relations": [
            {{ "from": "P-101", "to": "Zone A", "type": "location" }},
            {{ "from": "Maintenance Team", "to": "P-101", "type": "responsibility" }},
            {{ "from": "seal", "to": "P-101", "type": "assembly" }},
            {{ "from": "crack", "to": "seal", "type": "caused-by" }}
          ]
        }}
        """
        
        input_data = f"""
        Text: "{text[:2000]}"
        """

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # [Retry Strategy] 재시도 시 프롬프트 단순화 (예시 제거) 및 시스템 프롬프트 강화
                current_system_prompt = system_prompt
                current_user_prompt = definitions + example + input_data
                
                if attempt > 0:
                    current_system_prompt += "\n\nCRITICAL: Your previous response was invalid JSON. Return ONLY the JSON object. Do not include markdown formatting."
                    # 예시를 제거하여 프롬프트 단순화 (토큰 절약 및 혼란 방지)
                    current_user_prompt = definitions + input_data

                response = await self.llm_client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=[
                        {"role": "system", "content": current_system_prompt},
                        {"role": "user", "content": current_user_prompt}
                    ],
                    temperature=0.1,
                    response_format={ "type": "json_object" }
                )
                data = json.loads(response.choices[0].message.content)
                if not isinstance(data, dict):
                    raise ValueError("LLM returned non-dict JSON")
                return data
            except Exception as e:
                if attempt == max_retries - 1:
                    # print(f"⚠️ Extraction failed after {max_retries} attempts: {e}")
                    return {"entities": [], "relations": []}
                # 재시도 전 잠시 대기 (비동기)
                await asyncio.sleep(1)

    async def extract_graph_data_batch(self, texts: List[str]) -> Dict:
        """[Level 3] LLM을 통한 여러 행의 엔티티 및 관계 일괄 추출"""
        valid_types = ", ".join(self.schema_mgr.valid_parents)
        # Combine the JSON strings of rows into a larger JSON array string
        json_array_of_rows = "[" + ",".join(texts) + "]"
        
        # [Fix] System Prompt 분리 및 스키마 명시
        system_prompt = """
        You are an expert Industrial Knowledge Graph Engineer.
        Analyze the JSON array of table rows. Consolidate knowledge into a single graph.

        Output JSON Schema:
        {
          "entities": [ { "name": "string", "type": "string", "parent_type": "string" } ],
          "relations": [ { "from": "string", "to": "string", "type": "string" } ]
        }
        
        IMPORTANT: 'entities' must be a list of OBJECTS (dictionaries), NOT a list of lists.
        """

        # [Prompt Engineering] 구성 요소 분리
        definitions = f"""
        [Definitions]
        - **Equipment**: Physical machines and devices (e.g., Pump, Motor, Robot).
        - **Component**: Parts belonging to equipment (e.g., Bearing, Valve, Cable).
        - **Site**: Physical locations (e.g., Factory, Zone, Room).
        - **Operator**: People or teams who operate equipment.
        - **Manager**: People or departments responsible for sites or projects.
        - **Fault**: A malfunction or defect in a component or equipment.
        - **Alarm**: A signal or warning about a fault or an abnormal condition.
        
        [Rules]
        1. **Entities**: Extract L3 types and L2 parents from: [{valid_types}].
           - Ignore: Dates, IDs, Part Numbers, Status, Descriptions.
        2. **Relations**: 'assembly' (part-of), 'location', 'responsibility', 'connection'.
        """

        example = """
        [Example]
        Input: ["{{'Item': 'Pump-A', 'Part': 'Seal', 'Location': 'Room-1'}}"]
        Output: {{
          "entities": [
            {{ "name": "Pump-A", "type": "pump", "parent_type": "equipment" }},
            {{ "name": "Seal", "type": "seal", "parent_type": "component" }},
            {{ "name": "Room-1", "type": "room", "parent_type": "site" }}
          ],
          "relations": [
            {{ "from": "Seal", "to": "Pump-A", "type": "assembly" }},
            {{ "from": "Pump-A", "to": "Room-1", "type": "location" }}
          ]
        }}
        """
        
        input_data = f"""
        JSON Data:
        {json_array_of_rows}
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # [Retry Strategy] 재시도 시 프롬프트 단순화
                current_system_prompt = system_prompt
                current_user_prompt = definitions + example + input_data
                
                if attempt > 0:
                    current_system_prompt += "\n\nCRITICAL: Your previous response was invalid JSON. Return ONLY the JSON object."
                    current_user_prompt = definitions + input_data # 예시 제거

                response = await self.llm_client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=[
                        {"role": "system", "content": current_system_prompt},
                        {"role": "user", "content": current_user_prompt}
                    ],
                    temperature=0.1,
                    response_format={ "type": "json_object" },
                    timeout=120,
                    max_tokens=4096 # [Fix] 응답 잘림 방지를 위해 최대 토큰 수 명시
                )
                data = json.loads(response.choices[0].message.content)
                if not isinstance(data, dict):
                    raise ValueError("LLM returned non-dict JSON")
                return data
            except Exception as e:
                print(f"⚠️ Batch LLM extraction failed (Attempt {attempt+1}/{max_retries}): {e}. Retrying...", flush=True)
                await asyncio.sleep(2 * (attempt + 1)) # [Fix] 비동기 sleep으로 변경하여 이벤트 루프 블로킹 방지
        
        print(f"❌ Batch extraction failed after {max_retries} attempts.")
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
        # [Fix] 동시 실행 수 제한 (Semaphore) - 임베딩은 비교적 빠르므로 20개
        sem = asyncio.Semaphore(20)
        async def create_embedding_task(chunk_text):
            async with sem:
                return await self.get_embedding(chunk_text)

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

    async def preview_file_analysis(self, file_content: bytes, filename: str, progress_callback=None):
        """[Admin] 1단계: 파일 파싱 및 지식 추출 (DB 저장 X)"""
        doc_id = str(uuid.uuid4())
        print(f"  ➡️ Step 1/3: Parsing file '{filename}'...", flush=True)
        
        # 1. 파싱 및 청킹
        raw_chunks = parse_file_content(file_content, filename)

        print(f"  ➡️ Step 2/3: Creating embeddings for {len(raw_chunks)} chunks...", flush=True)
        # [OPTIMIZATION] Create embeddings in parallel
        # [Fix] 동시 실행 수 제한
        sem = asyncio.Semaphore(20)
        
        total_embeddings = len(raw_chunks)
        completed_embeddings = 0

        async def create_embedding_task(chunk_text):
            nonlocal completed_embeddings
            async with sem:
                res = await self.get_embedding(chunk_text)
            
            completed_embeddings += 1
            if completed_embeddings % 100 == 0 or completed_embeddings == total_embeddings:
                print(f"    🔹 Embedding Progress: {completed_embeddings}/{total_embeddings} ({(completed_embeddings/total_embeddings)*100:.1f}%)", flush=True)
            return res

        try:
            embedding_tasks = [create_embedding_task(rc['text']) for rc in raw_chunks]
            vectors = await asyncio.gather(*embedding_tasks)
        except Exception as e:
            print(f"  ❌ Embedding creation failed: {e}", flush=True)
            raise

        chunks = []
        for i, rc in enumerate(raw_chunks):
            chunk_id = f"{doc_id}_{'r' if rc['type']=='table-row' else 'c'}{rc['index']}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": rc['text'],
                "type": rc['type'],
                "vector": vectors[i] # 벡터 생성은 미리 수행
            })

        print(f"  ➡️ Step 3/3: Extracting knowledge from chunks...", flush=True)
        # 2. 지식 추출
        extracted_data = await self._analyze_chunks(chunks, progress_callback)
        
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
        # [Modified] 사용자의 요청으로 저장 시 스키마 업데이트 생략
        # self._update_schema_definitions(data['entities'], data.get('relations', []))
        
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

    async def _analyze_chunks(self, chunks: List[Dict], progress_callback=None) -> Dict:
        """각 청크에 대해 LLM을 호출하여 엔티티와 관계를 추출 (엑셀은 배치 처리)"""
        
        tasks = []
        sem = asyncio.Semaphore(5) # [Optimized] 서버 부하 감소를 위해 동시 실행 수 추가 감소

        # 엑셀 행(table-row)과 일반 텍스트 청크 분리
        table_row_chunks = [c for c in chunks if c['type'] == 'table-row']
        other_chunks = [c for c in chunks if c['type'] != 'table-row']

        # 1. 일반 텍스트 청크는 개별적으로 처리
        for chunk in other_chunks:
            async def extract_single_task(c):
                async with sem:
                    graph_data = await self.extract_graph_data(c['text'])
                # 결과를 ( [chunk_id], graph_data ) 튜플로 통일
                return [c['chunk_id']], graph_data
            tasks.append(extract_single_task(chunk))

        # 2. 엑셀 행은 배치로 묶어 처리
        BATCH_SIZE = 10 # [Optimized] 응답 잘림 및 타임아웃 방지를 위해 배치 크기 추가 감소
        if table_row_chunks:
            print(f"📊 Batching {len(table_row_chunks)} table rows into batches of {BATCH_SIZE}...")
        
        for i in range(0, len(table_row_chunks), BATCH_SIZE):
            batch = table_row_chunks[i:i+BATCH_SIZE]
            batch_texts = [c['text'] for c in batch]
            batch_chunk_ids = [c['chunk_id'] for c in batch]
            
            async def extract_batch_task(texts, chunk_ids):
                async with sem:
                    # 배치 추출 함수 호출
                    graph_data = await self.extract_graph_data_batch(texts)
                return chunk_ids, graph_data
            tasks.append(extract_batch_task(batch_texts, batch_chunk_ids))

        # [New] Progress Tracking
        total_tasks = len(tasks)
        completed_tasks = 0
        print(f"🚀 Starting analysis for {total_tasks} tasks...")

        async def wrap_with_progress(task):
            nonlocal completed_tasks
            try:
                res = await task
            except Exception as e:
                print(f"⚠️ Task failed in wrap_with_progress: {e}", flush=True)
                res = ([], {"entities": [], "relations": []})
            
            completed_tasks += 1
            if completed_tasks % 5 == 0 or completed_tasks == total_tasks:
                print(f"⏳ Analysis Progress: {completed_tasks}/{total_tasks} ({(completed_tasks/total_tasks)*100:.1f}%)", flush=True)
                
                # [WebSocket] Send progress update
                if progress_callback:
                    try:
                        await progress_callback((completed_tasks / total_tasks) * 100, f"Analyzing... {completed_tasks}/{total_tasks}")
                    except Exception as e:
                        print(f"⚠️ Progress callback failed: {e}", flush=True)
            return res

        wrapped_tasks = [wrap_with_progress(t) for t in tasks]
        results = await asyncio.gather(*wrapped_tasks)

        all_entities = {} # name -> {type, parent}
        all_relations = []
        chunk_links = [] # (chunk_id, entity_name)

        for chunk_ids, graph_data in results: # chunk_ids는 이제 리스트
            # [Fix] graph_data가 딕셔너리가 아닌 경우(예: 에러 문자열) 방어 코드 추가
            if not isinstance(graph_data, dict):
                print(f"⚠️ Unexpected graph_data type: {type(graph_data)}. Skipping. Value: {str(graph_data)[:100]}", flush=True)
                continue

            for ent in graph_data.get("entities", []):
                # [Fix] LLM이 엔티티를 dict가 아닌 list 등으로 잘못 반환하는 경우 방어
                if not isinstance(ent, dict):
                    print(f"⚠️ Unexpected entity format: {type(ent)}. Skipping. Value: {str(ent)[:100]}", flush=True)
                    continue

                name = ent.get('name')
                if not name: continue
                
                etype = ent.get('type') or "unknown-entity"
                
                if etype.lower() in {"date", "datetime", "time", "status", "description", "comment", "note", "unknown", "level", "alarm-level", "unnamed-level", "site-equipment"}:
                    continue

                parent = ent.get("parent_type") or "physical-asset"
                all_entities[name] = {"type": etype, "parent": parent}
                
                # 해당 엔티티를 찾은 모든 청크와 연결
                for chunk_id in chunk_ids:
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
        # 1. 엔티티 타입 정의 (Batch Optimization)
        type_pairs = [(info['type'], info['parent']) for info in entities.values()]
        resolved_types = self.schema_mgr.ensure_l3_types_batch(type_pairs)
        
        for name, info in entities.items():
            key = (info['type'], info['parent'])
            if key in resolved_types:
                info['type'] = resolved_types[key]
            
        # 2. 관계 타입 정의
        if relations:
            for rel in relations:
                # [Fix] LLM이 관계를 dict가 아닌 list 등으로 잘못 반환하는 경우 방어
                if not isinstance(rel, dict):
                    print(f"⚠️ Unexpected relation format: {type(rel)}. Skipping. Value: {str(rel)[:100]}", flush=True)
                    continue

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
        print(f"💾 Saving document '{safe_filename}' (ID: {doc_id}) to TypeDB...")

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
                self.insert_to_opensearch(
                    chunk['chunk_id'], chunk['text'], chunk['vector'], 
                    {"doc_id": doc_id, "filename": filename} # [Fix] doc_id를 메타데이터에 포함
                )

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

                # Case 3: Responsibility (responsible, subject-area)
                if rtype == 'responsibility':
                    queries.append(f'match $f has name "{fname}"; $t has name "{tname}"; insert (responsible: $f, subject-area: $t) isa {rtype};')

                # Case 4: Generic Connection (source, target) - Default fallback
                queries.append(f'match $f has name "{fname}"; $t has name "{tname}"; insert (source: $f, target: $t) isa {rtype};')
                
                for q in queries:
                    try: 
                        tx.query(q)
                        break # 성공하면 루프 종료
                    except: 
                        pass # 실패하면 다음 패턴 시도
            
            tx.commit()
        print(f"✅ Document '{filename}' saved successfully.")

    def delete_document(self, doc_id: str):
        """[Admin] 문서 및 관련 데이터 삭제"""
        print(f"🗑️ Deleting document {doc_id}...")
        
        # 1. TypeDB 삭제 (문서 + 포함된 청크)
        # 주의: 연결된 엔티티(장비 등)는 다른 문서에서도 쓸 수 있으므로 삭제하지 않음
        with self.driver.transaction(self.db_name, TransactionType.WRITE) as tx:
            # 1-1. Mention 관계 삭제 (Chunk가 Source인 경우)
            q_del_mentions = f"""
            match 
            $d isa document-file, has id-doc-id "{doc_id}";
            $c isa content-unit;
            (container: $d, content: $c) isa containment;
            $m (source: $c) isa mention;
            delete $m;
            """
            try: tx.query(q_del_mentions)
            except Exception as e: print(f"⚠️ Error deleting mentions: {e}")

            # 1-2. Chunk 및 Containment 삭제
            q_del_chunks = f"""
            match 
            $d isa document-file, has id-doc-id "{doc_id}";
            $c isa content-unit;
            $cont (container: $d, content: $c) isa containment;
            delete $c, $cont;
            """
            try: tx.query(q_del_chunks)
            except Exception as e: print(f"⚠️ Error deleting chunks: {e}")

            # 1-3. Document 삭제
            q_del_doc = f"""
            match $d isa document-file, has id-doc-id "{doc_id}";
            delete $d;
            """
            try: tx.query(q_del_doc)
            except Exception as e: print(f"⚠️ Error deleting document entity: {e}")
            
            tx.commit()

        # 2. OpenSearch 삭제
        # [Fix] doc_id를 기준으로 해당 문서의 모든 청크 삭제
        try:
            self.os_client.delete_by_query(
                index=self.index_name, body={"query": {"term": {"metadata.doc_id.keyword": doc_id}}}
            )
        except:
            # Fallback for text field
            self.os_client.delete_by_query(
                index=self.index_name, body={"query": {"match": {"metadata.doc_id": doc_id}}}
            )
            
        return {"status": "deleted", "doc_id": doc_id}

    def list_documents(self):
        """[Admin] 저장된 문서 목록 조회"""
        docs = []
        try:
            with self.driver.transaction(self.db_name, TransactionType.READ) as tx:
                # [Fix] Use attribute projection in fetch to handle optional attributes gracefully
                q = """
                match $d isa document-file;
                fetch { 
                    "id": $d.id-doc-id, 
                    "name": $d.name, 
                    "date": $d.created-date 
                };
                """
                results = tx.query(q)
                if hasattr(results, 'resolve'): results = results.resolve()
                for res in results:
                    # Helper to extract value from potential list or single object
                    def get_val(field):
                        raw = res.get(field)
                        if not raw: return None
                        item = raw[0] if isinstance(raw, list) and raw else raw
                        return item.get("value") if isinstance(item, dict) else item

                    doc_id = get_val("id")
                    name = get_val("name")
                    date = get_val("date")

                    if doc_id:
                        # The date is a datetime object, so we convert it to a string for JSON serialization.
                        docs.append({"id": doc_id, "name": name, "date": str(date) if date else ""})
            
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