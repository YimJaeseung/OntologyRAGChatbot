import re
import os
from typedb.driver import TypeDB, TransactionType, Credentials, DriverOptions

class SchemaManager:
    def __init__(self, driver, db_name):
        self.driver = driver
        self.db_name = db_name
        
        # schemal.tql에 정의된 L2 Entity 목록
        self.valid_parents = {
            "equipment", "component", "sensor", "site", "zone",
            "document-file", # content-unit 제거 (LLM이 오용하지 않도록)
            "engineer", "operator", "manager",
            "fault", "alarm", "maintenance-activity"
        }
        self._known_types = set(self.valid_parents)
        self._known_relations = set()
        self._load_base_schema_types()

    def _load_base_schema_types(self):
        """init_data/schema.tql 파일을 파싱하여 초기에 정의된 관계 목록을 로드합니다."""
        schema_path = os.getenv("SCHEMA_PATH", "/init_data/schema.tql")
        if os.path.exists(schema_path):
            try:
                with open(schema_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    matches = re.findall(r'^\s*relation\s+([a-zA-Z0-9_-]+),', content, re.MULTILINE)
                    for rel_name in matches:
                        self._known_relations.add(rel_name)
            except Exception as e:
                print(f"⚠️ Could not parse base relations from schema.tql: {e}")

    def sanitize_type_name(self, name: str) -> str:
        if not name:
            return ""
        slug = str(name).lower()
        slug = slug.replace('_', '-')
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'[\s-]+', '-', slug)
        return slug.strip('-')

    def ensure_l3_type(self, l3_name: str, l2_parent: str) -> str:
        slug_l3 = self.sanitize_type_name(l3_name)
        slug_parent = self.sanitize_type_name(l2_parent)
        
        # [Hierarchy Enforcement] L1 타입을 적절한 L2 타입으로 매핑
        l1_defaults = {
            "physical-asset": "equipment",
            "person": "operator",
            "event": "maintenance-activity",
            "content": "document-file",
            "content-unit": "document-file" # content-unit으로 들어오면 document-file로 매핑
        }
        if slug_parent in l1_defaults:
            slug_parent = l1_defaults[slug_parent]
        
        if slug_l3 in self._known_types or slug_l3 == slug_parent:
            return slug_l3

        if slug_parent not in self.valid_parents:
            print(f"⚠️ Invalid parent '{slug_parent}'. Fallback to 'document-file'")
            slug_parent = "document-file"

        # 1. 존재 및 충돌 확인
        with self.driver.transaction(self.db_name, TransactionType.READ) as tx:
            try:
                # [Fix] concepts API 대신 쿼리로 존재 확인
                q_check = f"match $x sub {slug_l3}; select $x; limit 1;"
                if list(tx.query(q_check).resolve()):
                     self._known_types.add(slug_l3)
                     return slug_l3
            except Exception: 
                pass # 타입이 없으면 아래 정의 로직으로 이동

        # 2. 없으면 정의 (SCHEMA 트랜잭션)
        print(f"🆕 Defining New L3 Type: '{slug_l3}' (sub {slug_parent})")
        try:
            with self.driver.transaction(self.db_name, TransactionType.SCHEMA) as tx:
                define_query = f"define entity {slug_l3}, sub {slug_parent};"
                tx.query(define_query)
                tx.commit()
            self._known_types.add(slug_l3)
            return slug_l3
        except Exception as e:
            print(f"⚠️ Failed to define type '{slug_l3}': {e}")
            # 이름 충돌 시 '_entity' 접미사 추가하여 재시도 (예: department -> department-entity)
            alt_slug = f"{slug_l3}-entity"
            try:
                with self.driver.transaction(self.db_name, TransactionType.SCHEMA) as tx:
                    tx.query(f"define entity {alt_slug}, sub {slug_parent};")
                    tx.commit()
                self._known_types.add(alt_slug)
                return alt_slug
            except:
                print(f"❌ Failed to define alternative type. Fallback to {slug_parent}")
                return slug_parent

    def ensure_l3_types_batch(self, type_pairs: list) -> dict:
        """
        [Optimization] 배치 단위로 L3 타입을 확인하고 정의하여 트랜잭션 오버헤드 감소
        Args:
            type_pairs: List of (l3_name, l2_parent) tuples
        Returns:
            Dict mapping (l3_name, l2_parent) -> final_slug
        """
        resolved_map = {}
        definitions_needed = {} # slug -> parent_slug

        # 1. 메모리 캐시 확인 및 전처리
        for l3, parent in type_pairs:
            slug_l3 = self.sanitize_type_name(l3)
            slug_parent = self.sanitize_type_name(parent)
            
            # L1 Defaults (ensure_l3_type와 동일 로직)
            l1_defaults = {
                "physical-asset": "equipment",
                "person": "operator",
                "event": "maintenance-activity",
                "content": "document-file",
                "content-unit": "document-file"
            }
            if slug_parent in l1_defaults:
                slug_parent = l1_defaults[slug_parent]
            
            # 이미 확인된 타입이면 스킵
            if slug_l3 in self._known_types or slug_l3 == slug_parent:
                resolved_map[(l3, parent)] = slug_l3
                continue

            if slug_parent not in self.valid_parents:
                slug_parent = "document-file"
            
            resolved_map[(l3, parent)] = slug_l3
            if slug_l3 not in definitions_needed:
                definitions_needed[slug_l3] = slug_parent

        if not definitions_needed:
            return resolved_map

        # 2. DB 존재 여부 확인 및 일괄 정의
        try:
            # 존재하지 않는 타입만 필터링 (Batch Read)
            with self.driver.transaction(self.db_name, TransactionType.READ) as tx:
                missing_slugs = {slug: p_slug for slug, p_slug in definitions_needed.items() 
                                 if not list(tx.query(f"match $x sub {slug}; select $x; limit 1;").resolve())}
            
            # 없는 타입 일괄 정의 (Batch Schema Write)
            if missing_slugs:
                print(f"🆕 Batch Defining {len(missing_slugs)} New L3 Types...")
                with self.driver.transaction(self.db_name, TransactionType.SCHEMA) as tx:
                    for slug, p_slug in missing_slugs.items():
                        tx.query(f"define entity {slug}, sub {p_slug};")
                    tx.commit()
                self._known_types.update(missing_slugs.keys())
                
        except Exception as e:
            print(f"⚠️ Batch definition failed: {e}. Fallback to individual definition.")
            # 실패 시 개별 처리로 폴백
            for l3, parent in type_pairs:
                resolved_map[(l3, parent)] = self.ensure_l3_type(l3, parent)

        return resolved_map

    def get_schema_tree(self) -> dict:
        """현재 정의된 스키마 계층 구조(L2 -> L3)를 조회"""
        tree = {}
        # 조회 시에는 content-unit도 포함하여 구조 확인 가능하게 함
        target_parents = self.valid_parents.union({"content-unit"})
        with self.driver.transaction(self.db_name, TransactionType.READ) as tx:
            for parent in target_parents:
                try:
                    # 해당 부모 타입(L2)의 직계 하위 타입(L3)만 조회 (sub!)
                    q = f"match $x sub! {parent}; select $x;"
                    res = tx.query(q)
                    if hasattr(res, 'resolve'): res = res.resolve()
                    
                    children = []
                    for r in res:
                        c = r.get("x")
                        if c:
                            # TypeDB Driver: Concept -> label -> name
                            # [Fix] 드라이버/객체 버전에 따른 label 접근 방식 호환성 처리
                            try:
                                # Standard TypeDB 3.x
                                name = c.label.name
                            except AttributeError:
                                try:
                                    name = c.get_label().name
                                except:
                                    name = str(c).split(':')[-1].strip()
                            
                            # [Clean Up] 'EntityType(name)' 형태의 문자열 정리
                            if name.startswith("EntityType(") and name.endswith(")"):
                                name = name[11:-1]

                            if name != parent:
                                children.append(name)
                    if children:
                        tree[parent] = sorted(children)
                except Exception as e:
                    print(f"⚠️ Error fetching schema tree for '{parent}': {e}")

            # [New] 관계(Relation) 목록 조회
            relations = []
            if self._known_relations:
                tree["relations"] = sorted(list(self._known_relations))

        return tree

    def ensure_relation_type(self, rel_name: str, from_type: str, to_type: str) -> str:
        """
        관계 타입과 역할을 정의하고, 엔티티 타입에 plays 관계를 설정함.
        """
        slug_rel = self.sanitize_type_name(rel_name)
        slug_from = self.sanitize_type_name(from_type)
        slug_to = self.sanitize_type_name(to_type)
        
        # [Fix] 속성 이름과 충돌하는 경우 엔티티 이름 보정 (예: department -> department-entity)
        def resolve_entity_name(name):
            try:
                with self.driver.transaction(self.db_name, TransactionType.READ) as tx:
                    if tx.concepts.get_attribute_type(name).resolve():
                        return f"{name}-entity"
            except:
                pass
            return name

        slug_from = resolve_entity_name(slug_from)
        slug_to = resolve_entity_name(slug_to)

        if not slug_rel or not slug_from or not slug_to:
            return slug_rel

        # 역할 이름 결정 (하드코딩된 매핑 또는 기본값)
        role_map = {
            "assembly": ("part", "system"),
            "part-of": ("part", "system"),
            "composition": ("part", "system"),
            "connection": ("source", "target"),
            "location": ("located", "place"),
            "containment": ("content", "container"),
            "caused-by": ("source", "target"),
            "alarm": ("source", "target"),
            "manager": ("source", "target"),
            "responsibility": ("responsible", "subject-area")
        }
        
        # 기본값은 source/target
        role1, role2 = role_map.get(slug_rel, ("source", "target"))
        
        # 관계 타입 이름을 표준화 (매핑된 키가 있으면 그것 사용)
        if slug_rel in ["part-of", "composition"]:
            slug_rel = "assembly"
        if slug_rel in ["requester", "responsible", "managed-by"]:
            slug_rel = "responsibility"

        # 1. 존재 확인 및 충돌 처리
        # [Fix] 이미 알려진 관계라면 DB 확인 및 이름 변경 스킵
        if slug_rel in self._known_relations:
            is_relation = True
        else:
            is_relation = False
            is_occupied = False
            try:
                with self.driver.transaction(self.db_name, TransactionType.READ) as tx:
                    # 존재 여부 확인
                    q_check = f"match $x sub {slug_rel}; select $x; limit 1;"
                    if list(tx.query(q_check).resolve()):
                        is_occupied = True
                        # 관계 타입인지 확인 (TQL 제약으로 인해 정확한 확인이 어려울 수 있음)
                        # 여기서는 이름이 점유되었는데 known_relations에 없으면 충돌로 간주할 수도 있으나,
                        # 안전을 위해 DB에서 추가 확인을 시도하거나, 충돌로 처리
                        pass
            except Exception:
                pass
                
            # 이름이 점유되었으나 관계가 아닌 경우 (예: 엔티티와 이름 충돌) -> 이름 변경
            if is_occupied and not is_relation:
                print(f"⚠️ Name '{slug_rel}' is occupied by a non-relation type. Renaming to '{slug_rel}-relation'.")
                slug_rel = f"{slug_rel}-relation"
                is_relation = False # 새 이름은 정의되지 않았다고 가정

        # 2. 관계 타입 정의 (SCHEMA 트랜잭션)
        if not is_relation:
            try:
                with self.driver.transaction(self.db_name, TransactionType.SCHEMA) as tx:
                    print(f"🆕 Defining New Relation: '{slug_rel}' (roles: {role1}, {role2})")
                    q_rel = f"define relation {slug_rel}, relates {role1}, relates {role2};"
                    tx.query(q_rel)
                    tx.commit()
                self._known_relations.add(slug_rel)
            except Exception as e:
                print(f"⚠️ Failed to define relation '{slug_rel}': {e}")
                return slug_rel

        # 3. 엔티티에 plays 역할 부여 (개별 트랜잭션으로 분리하여 SVL42 오류 회피)
        for entity_type, role in [(slug_from, role1), (slug_to, role2)]:
            try:
                with self.driver.transaction(self.db_name, TransactionType.SCHEMA) as tx:
                    tx.query(f"define entity {entity_type}, plays {slug_rel}:{role};")
                    tx.commit()
            except Exception as e:
                # SVL42: Cannot redeclare inherited capability... 는 무시
                if "SVL42" not in str(e):
                    print(f"⚠️ Failed to define plays for '{entity_type}' on '{slug_rel}:{role}': {e}")
        
        return slug_rel