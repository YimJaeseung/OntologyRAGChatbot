import asyncio
import json
import re
from typedb.driver import TypeDB, Credentials, DriverOptions, TransactionType
from app.etl import DynamicETL

async def analyze_query(user_query: str, processor: DynamicETL):
    """자연어 질문에서 TypeDB 검색용 핵심 키워드 및 엔티티 추출"""
    prompt = f"""
    Analyze the user question and extract key industrial entities or model names.
    User Question: "{user_query}"
    Return ONLY a JSON list of strings (keywords). 
    Example: ["Pump-A", "Sensor-01"]
    """
    # processor의 llm_client 사용
    response = processor.llm_client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except:
        return user_query.split()

async def search_opensearch(query_text: str, processor: DynamicETL, top_k: int = 3):
    """벡터 유사도 기반의 비정형 청크 검색"""
    # processor의 메서드 및 클라이언트 활용
    query_vector = processor.get_embedding(query_text)
    
    search_query = {
        "size": top_k,
        "query": {
            "knn": {
                "vector_field": {
                    "vector": query_vector,
                    "k": top_k
                }
            }
        }
    }
    
    response = processor.os_client.search(
        index=processor.index_name,
        body=search_query
    )
    
    hits = response['hits']['hits']
    return [{"text": h['_source']['text'], "score": h['_score']} for h in hits]

async def search_typedb(keywords: list, processor: DynamicETL):
    """키워드 기반의 그래프 관계 및 속성 검색"""
    results = []
    # processor의 설정 활용, TypeDB 클래스는 직접 사용
    try:
        with processor.driver.transaction(processor.db_name, TransactionType.READ) as tx:
                for word in keywords:
                    # 정규식 특수문자 이스케이프 처리
                    safe_word = re.escape(word)
                    
                    # [수정] 단일 쿼리로 최적화 (fetch all 방식 제거)
                    # $n(이름)에 대해 like 연산 수행 후, 연결된 텍스트 조회
                    tql = f"""
                    match 
                    $e has name $n;
                    $n like "(?i).*{safe_word}.*";
                    (target: $e, source: $c) isa mention;
                    $c has content-text $text;
                    fetch {{"text": $text}};
                    """
                    
                    try:
                        query_res = tx.query(tql)
                        if hasattr(query_res, 'resolve'):
                            query_res = query_res.resolve()

                        for res in query_res:
                            # fetch with JSON structure returns primitive values.
                            text_val = res.get("text")
                            if text_val:
                                results.append(text_val)
                    except Exception as e:
                        print(f"⚠️ TQL Execution Failed for word '{word}': {e}")
                        continue

    except Exception as e:
        print(f"⚠️ TypeDB Connection Failed: {e}")
        
    return list(set(results))

async def hybrid_search(user_query: str, processor: DynamicETL):
    # 위 함수들을 조합하여 최종 결과 생성 (이전 로직 동일)
    keywords = await analyze_query(user_query, processor)
    os_data, tdb_data = await asyncio.gather(
        search_opensearch(user_query, processor),
        search_typedb(keywords, processor)
    )

    context = "--- Vector Search Results ---\n"
    context += "\n".join([d['text'] for d in os_data])
    context += "\n\n--- Graph Knowledge Results ---\n"
    context += "\n".join(tdb_data)
    
    # 4. LLM 최종 답변 생성
    final_prompt = f"""
    [지시사항]
    아래 제공된 Context를 바탕으로 사용자의 질문에 답변하세요.
    답변은 반드시 '한국어'로 작성해야 합니다.
    중국어나 한자는 절대 사용하지 마세요.
    전문 용어는 한국어 표준 용어를 사용하고, 필요한 경우 괄호 안에 영문 약어를 쓰세요.
    답변의 어조는 단정하고 전문적인 한국어 문체를 사용하세요.

    
    [Context]
    {context}
    
    [Question]
    {user_query}
    
    [Answer]
    """
    
    response = processor.llm_client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {
            "role": "system", 
            "content": (
                "You are an industrial AI assistant. "
                "You must answer in Korean. "
                "Never use Chinese characters or Hanja. "
                "If the retrieved context contains Chinese, translate it to Korean."
            )
        },
        {"role": "user", "content": final_prompt}
    ],
    temperature=0.3  # 창의성보다는 지시 준수력을 위해 약간 낮춤
)
    
    return response.choices[0].message.content