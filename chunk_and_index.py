# chunk_and_index.py

import os
import sys

# Force UTF-8 output on Windows to prevent UnicodeEncodeError
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Tự động load token từ .env nếu có
try:
    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
except ImportError:
    pass

import json
import redis
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

# --------------------------------------------
# 1. CẤU HÌNH CHUNG
# --------------------------------------------
# Redis Stack đang chạy Docker Desktop trên port 6380 
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
INDEX_NAME = "idx:law"

# Model và kích thước vector
MODEL_NAME = "sentence-transformers/LaBSE"  # 768d
VECTOR_DIMS = 768  # LaBSE có kích thước 768

# --------------------------------------------
# 2. KẾT NỐI REDIS và Embedding Model
# --------------------------------------------
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

def create_redis_index():
    """
    Tạo Redis index với cấu hình phù hợp cho LaBSE.
    Nếu index đã tồn tại với kích thước khác, xóa và tạo lại.
    """
    try:
        # Kiểm tra index hiện tại
        info = r.execute_command("FT.INFO", INDEX_NAME)
        current_dims = None
        for i in range(0, len(info), 2):
            if info[i] == b"attributes":
                attrs = info[i + 1]
                for attr in attrs:
                    if b"VECTOR" in attr:
                        current_dims = int(attr[attr.index(b"DIM") + 1])
                        break
        
        # Nếu kích thước vector khác, xóa index cũ
        if current_dims is not None and current_dims != VECTOR_DIMS:
            print(f"Xóa index cũ ({current_dims}d) để tạo lại với kích thước mới ({VECTOR_DIMS}d)")
            r.execute_command("FT.DROPINDEX", INDEX_NAME)
    except:
        pass

    try:
        # Tạo index mới với HNSW
        r.execute_command(
            "FT.CREATE", INDEX_NAME,
            "ON", "HASH",
            "PREFIX", "1", "doc:",
            "SCHEMA",
            "vector", "VECTOR", "HNSW", "6",
              "TYPE", "FLOAT32",
              "DIM", str(VECTOR_DIMS),
              "DISTANCE_METRIC", "COSINE",
            "text", "TEXT",
            "meta", "TEXT"
        )
        print(f"Đã tạo Redis index '{INDEX_NAME}' với vector {VECTOR_DIMS}d")
    except Exception as e:
        if "Index already exists" not in str(e):
            print(f"Lỗi khi tạo index: {e}")

# Khởi tạo embedding model với CUDA nếu có sẵn
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert = SentenceTransformer(MODEL_NAME, device=device)

# Tạo/Cập nhật Redis index
create_redis_index()

# --------------------------------------------
# 3. HÀM XỬ LÝ TEXT & EMBEDDING
# --------------------------------------------
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Chia một đoạn text dài thành các chunk nhỏ hơn.
    Args:
        text: Văn bản cần chia
        chunk_size: Số token tối đa trong mỗi chunk
        overlap: Số token overlap giữa các chunk liên tiếp
    Returns:
        List[str]: Danh sách các chunk
    """
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        chunks.append(" ".join(chunk))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks

def get_embedding(text: str, batch_size: int = 32) -> bytes:
    """
    Sinh embedding vector cho một đoạn text.
    Args:
        text: Văn bản cần encode
        batch_size: Kích thước batch cho xử lý nhiều văn bản
    Returns:
        bytes: Vector embedding dưới dạng bytes
    """
    emb: np.ndarray = sbert.encode(
        text,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    arr = np.array(emb, dtype=np.float32)
    return arr.tobytes()

# --------------------------------------------
# 4. HÀM INDEX & SEARCH
# --------------------------------------------
def index_all_chunks(txt_folder: str):
    """
    Duyệt qua từng file .txt trong txt_folder, chia chunk và index vào Redis.
    Args:
        txt_folder: Đường dẫn tới thư mục chứa các file .txt
    """
    chunk_counter = 0

    for filename in sorted(os.listdir(txt_folder)):
        if not filename.lower().endswith(".txt"):
            continue

        txt_path = os.path.join(txt_folder, filename)
        with open(txt_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        chunks = chunk_text(full_text, chunk_size=400, overlap=50)
        print(f"→ Đang index file '{filename}' với {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            meta = {
                "source": filename.replace(".txt", ""),
                "chunk_id": i
            }
            meta_str = json.dumps(meta, ensure_ascii=False)

            emb_bytes = get_embedding(chunk)
            key = f"doc:{chunk_counter}"

            r.hset(key, mapping={
                "vector": emb_bytes,
                "text": chunk,
                "meta": meta_str
            })

            chunk_counter += 1
            if chunk_counter % 100 == 0:
                print(f"   → Đã index tổng cộng {chunk_counter} chunks...")

    print(f"✅ Hoàn tất: đã index {chunk_counter} chunks vào Redis Stack.")

def retrieve_similar_chunks(query: str, top_k: int = 3, threshold: float = 0.0) -> List[Dict[str, Any]]:
    """
    Tìm các chunk văn bản tương tự với câu query.
    Args:
        query: Câu truy vấn
        top_k: Số lượng chunk muốn lấy về
        threshold: Ngưỡng similarity tối thiểu (0-1)
    Returns:
        List[Dict]: Danh sách các chunk phù hợp, mỗi chunk là một dict với các key:
            - text: Nội dung văn bản
            - meta: Thông tin meta của chunk
            - score: Điểm similarity
    """
    print(f"[DEBUG] retrieve_similar_chunks called: query='{query[:40]}...', top_k={top_k}, threshold={threshold}", flush=True)
    q_emb = get_embedding(query)
    print(f"[DEBUG] Embedding done, len={len(q_emb)} bytes", flush=True)

    try:
        raw = r.execute_command(
            "FT.SEARCH", INDEX_NAME,
            f"*=>[KNN {top_k} @vector $vec AS vector_score]",
            "PARAMS", 2, "vec", q_emb,
            "RETURN", 3, "text", "meta", "vector_score",
            "DIALECT", 2
        )
        print(f"[DEBUG] FT.SEARCH raw len={len(raw) if raw else 0}, hits={raw[0] if raw else 'N/A'}", flush=True)
    except Exception as e:
        print(f"[DEBUG] ❌ FT.SEARCH error: {e}", flush=True)
        return []

    if not raw or len(raw) <= 1:
        print(f"[DEBUG] ❌ raw is empty or has no results", flush=True)
        return []

    total_hits = int(raw[0])
    results = []
    
    for i in range(1, len(raw), 2):
        props = raw[i + 1]
        if not isinstance(props, list):
            continue
        props_dict = {}
        for j in range(0, len(props), 2):
            key = props[j].decode("utf-8") if isinstance(props[j], bytes) else str(props[j])
            val = props[j + 1]
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            props_dict[key] = val
        
        text = props_dict.get("text", "")
        meta_str = props_dict.get("meta", "{}")
        try:
            meta = json.loads(meta_str)
        except:
            meta = {"raw_meta": meta_str}
        try:
            # Redis KNN trả về cosine DISTANCE [0,1] (0=giống hệt, 1=trái ngược)
            # Chuyển sang cosine SIMILARITY: score = 1 - distance
            cosine_dist = float(props_dict.get("vector_score", 1.0))
            score = 1.0 - cosine_dist  # score=1.0 mà giống, score=0.0 là trái ngược
        except:
            score = 0.0
            
        source = meta.get("source", "Unknown")
        dieu = meta.get("dieu", "")
        passed = score >= threshold
        flag = "✓ PASS" if passed else "✗ skip"
        print(f"[DEBUG] {flag} | sim={score:.4f} dist={cosine_dist:.4f} | {source}{' Điều '+dieu if dieu else ''}", flush=True)
        
        if passed:
            results.append({"text": text, "meta": meta, "score": score})
            
    print(f"[DEBUG] Threshold={threshold:.2f} | {len(results)}/{len(range(1, len(raw), 2))-1} chunks passed", flush=True)
    return results

# --------------------------------------------
# 5. MAIN
# --------------------------------------------
if __name__ == "__main__":
    # Index tất cả file .txt trong thư mục "plain_texts"
    TXT_FOLDER = "plain_texts"
    if not os.path.isdir(TXT_FOLDER):
        print(f"🚨 Thư mục '{TXT_FOLDER}' không tồn tại. Vui lòng tạo và đặt file .txt vào đó.")
    else:
        index_all_chunks(TXT_FOLDER)
