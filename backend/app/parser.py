import os
import pandas as pd
import pdfplumber
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

def sanitize_text(text: str) -> str:
    """Remove surrogate characters to prevent encoding errors."""
    if not isinstance(text, str):
        return str(text)
    return text.encode('utf-8', 'ignore').decode('utf-8')

def parse_file_content(file_content: bytes, filename: str) -> List[Dict]:
    """
    파일 내용을 읽어 텍스트 청크 리스트를 반환합니다.
    반환 형식: [{'text': str, 'index': int, 'type': str}]
    """
    is_excel = filename.endswith(".xlsx") or filename.endswith(".xls")
    temp_path = f"/tmp/{filename}"
    
    # 임시 파일 저장
    with open(temp_path, "wb") as f:
        f.write(file_content)

    chunks = []
    try:
        if is_excel:
            df = pd.read_excel(temp_path).fillna("")
            for idx, row in df.iterrows():
                row_json = row.to_json(force_ascii=False)
                chunks.append({
                    "text": sanitize_text(row_json),
                    "index": idx,
                    "type": "table-row"
                })
        elif filename.endswith(".pdf"):
            with pdfplumber.open(temp_path) as pdf:
                full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
            for idx, text in enumerate(splitter.split_text(full_text)):
                chunks.append({
                    "text": sanitize_text(text),
                    "index": idx,
                    "type": "text-chunk"
                })
        elif filename.endswith(".txt"):
            with open(temp_path, "r", encoding="utf-8") as f:
                full_text = f.read()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
            for idx, text in enumerate(splitter.split_text(full_text)):
                chunks.append({
                    "text": sanitize_text(text),
                    "index": idx,
                    "type": "text-chunk"
                })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return chunks