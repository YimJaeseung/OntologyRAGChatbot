from typedb.driver import TypeDB, TransactionType, Credentials, DriverOptions
from opensearchpy import OpenSearch
import os

def check_counts():
    # 1. OpenSearch 데이터 확인
    print(f"📊 --- OpenSearch Data Status ---")
    os_url = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
    try:
        os_client = OpenSearch(hosts=[os_url], http_auth=None, use_ssl=False)
        index_name = "rag-docs"
        if os_client.indices.exists(index=index_name):
            count = os_client.count(index=index_name)["count"]
            print(f"✅ Documents in '{index_name}': {count}")
        else:
            print(f"⚠️ Index '{index_name}' does not exist.")
        os_client.close()
    except Exception as e:
        print(f"❌ OpenSearch Connection Failed: {e}")

    # etl.py에서 사용한 설정값과 동일하게 맞춤 
    uri = "localhost:1729"
    db_name = "rag_ontology"
    creds = Credentials("admin", "password")
    opts = DriverOptions(is_tls_enabled=False)

    print(f"\n📊 --- TypeDB Data Status ---")
    with TypeDB.driver(uri, creds, opts) as driver:
        # 데이터베이스 존재 확인 
        if not driver.databases.contains(db_name):
            print(f"❌ Database '{db_name}' does not exist.")
            return

        with driver.transaction(db_name, TransactionType.READ) as tx:
            # 1. 물리 자산(Entity) 개수 확인 
            asset_q = "match $e isa physical-asset; reduce $count = count;"
            asset_count = next(tx.query(asset_q).resolve()).get("count").as_value()
            
            # 2. 지식 청크(Content Unit) 개수 확인 
            chunk_q = "match $c isa content-unit; reduce $count = count;"
            chunk_count = next(tx.query(chunk_q).resolve()).get("count").as_value()
            
            # 3. 언급(Mention) 관계 개수 확인 
            mention_q = "match $rel (source: $s, target: $t) isa mention; reduce $count = count;"
            mention_count = next(tx.query(mention_q).resolve()).get("count").as_value()

            print(f"✅ Physical Assets: {asset_count}")
            print(f"✅ Knowledge Chunks: {chunk_count}")
            print(f"✅ Mention Relations: {mention_count}")

if __name__ == "__main__":
    check_counts()