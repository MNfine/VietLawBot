# retrieve_and_answer_lite.py
# Phiên bản nhẹ không sử dụng Docker/Redis
# Đọc trực tiếp từ các file .txt và sử dụng sentence-transformers để tìm kiếm

import os
import json
import re
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import time
import random

# Load các biến môi trường
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --------------------------------------------
# CẤU HÌNH CHUNG
# --------------------------------------------
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

# Số chunk lấy về và ngưỡng similarity
TOP_K = 5
SIMILARITY_THRESHOLD = 0.60  # LaBSE cosine similarity thực: văn bản pháp lý tiếng Việt thường đạt 0.4-0.55
SHOW_DEBUG = True  # Lite version bật debug mặc định để dễ kiểm tra

# Đường dẫn tới thư mục chứa văn bản pháp luật
PLAIN_TEXTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plain_texts")

# Model embedding
MODEL_NAME = "sentence-transformers/LaBSE"

# Cache cho chunks và embeddings (load một lần khi khởi động)
_chunks_cache = None
_embeddings_cache = None
_sbert_model = None

def get_sbert_model():
    """Lazy load sentence transformer model."""
    global _sbert_model
    if _sbert_model is None:
        print("Đang load model embedding LaBSE...")
        _sbert_model = SentenceTransformer(MODEL_NAME)
        print("Đã load xong model embedding.")
    return _sbert_model

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Chia một đoạn text dài thành các chunk nhỏ hơn.
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

def load_all_chunks() -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Load tất cả chunks từ thư mục plain_texts và tạo embeddings.
    Cache để không phải load lại mỗi lần query.
    """
    global _chunks_cache, _embeddings_cache
    
    if _chunks_cache is not None and _embeddings_cache is not None:
        return _chunks_cache, _embeddings_cache
    
    print("Đang load và index văn bản pháp luật...")
    
    all_chunks = []
    
    if not os.path.isdir(PLAIN_TEXTS_DIR):
        print(f"Thư mục '{PLAIN_TEXTS_DIR}' không tồn tại!")
        return [], np.array([])
    
    for filename in sorted(os.listdir(PLAIN_TEXTS_DIR)):
        if not filename.lower().endswith(".txt"):
            continue
        if filename == "fields.meta":
            continue
            
        txt_path = os.path.join(PLAIN_TEXTS_DIR, filename)
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                full_text = f.read()
        except Exception as e:
            print(f"Lỗi đọc file {filename}: {e}")
            continue
            
        chunks = chunk_text(full_text, chunk_size=400, overlap=50)
        
        for i, chunk in enumerate(chunks):
            meta = {
                "source": filename.replace(".txt", ""),
                "chunk_id": i
            }
            all_chunks.append({
                "text": chunk,
                "meta": meta
            })
    
    if not all_chunks:
        print("Không tìm thấy chunks nào!")
        return [], np.array([])
    
    print(f"Đã load {len(all_chunks)} chunks từ {PLAIN_TEXTS_DIR}")
    
    # Tạo embeddings cho tất cả chunks
    sbert = get_sbert_model()
    texts = [c["text"] for c in all_chunks]
    
    print("Đang tạo embeddings cho tất cả chunks...")
    embeddings = sbert.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    _chunks_cache = all_chunks
    _embeddings_cache = embeddings
    
    print(f"Hoàn tất indexing {len(all_chunks)} chunks.")
    return all_chunks, embeddings

def retrieve_similar_chunks(query: str, top_k: int = 3, threshold: float = 0.0) -> List[Dict[str, Any]]:
    """
    Tìm các chunk văn bản tương tự với câu query.
    Sử dụng cosine similarity trực tiếp trong memory.
    """
    chunks, embeddings = load_all_chunks()
    
    if not chunks or embeddings.size == 0:
        return []
    
    # Tạo embedding cho query
    sbert = get_sbert_model()
    query_emb = sbert.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Tính cosine similarity
    similarities = np.dot(embeddings, query_emb)
    
    # Lấy top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score >= threshold:
            chunk = chunks[idx]
            results.append({
                "text": chunk["text"],
                "meta": chunk["meta"],
                "score": score
            })
            if SHOW_DEBUG:
                source = chunk["meta"].get("source", "Unknown")
                print(f"[DEBUG] Chunk score: {score:.4f} | Source: {source}", flush=True)
    
    return results

def gemini_answer(prompt: str) -> str:
    """Gọi Gemini để sinh câu trả lời (có retry/backoff và thông báo lỗi rõ)."""
    if not GEMINI_API_KEY:
        return "Lỗi: GOOGLE_API_KEY đang rỗng. Hãy set biến môi trường GOOGLE_API_KEY trước khi gọi Gemini."

    # Thử danh sách model theo thứ tự (tùy account, có model gọi được/không)
    model_candidates = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-lite"
    ]

    last_err = None

    for model_name in model_candidates:
        try:
            gm = genai.GenerativeModel(model_name)

            # Retry tối đa 3 lần cho mỗi model khi gặp 429 kiểu rate-limit (không phải quota=0)
            for attempt in range(3):
                try:
                    resp = gm.generate_content(prompt)
                    # resp.text đôi khi None nếu bị safety/empty
                    text = getattr(resp, "text", None)
                    return (text or "").strip() or "Gemini trả về rỗng (không có text)."
                except Exception as e:
                    msg = str(e)
                    last_err = e

                    # Nếu là 429 thì backoff (nhưng nếu quota=0 thì vẫn fail)
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                        # exponential backoff + jitter
                        sleep_s = min(2 ** attempt, 8) + random.random()
                        time.sleep(sleep_s)
                        continue
                    else:
                        # lỗi khác thì break luôn để thử model khác
                        break

        except Exception as e:
            last_err = e
            continue

    return f"Lỗi khi gọi Gemini API (đã thử nhiều model): {last_err}"

def parse_mcq(query: str) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    Tách câu hỏi trắc nghiệm thành phần đề bài và các phương án.
    """
    pattern = (
        r"(.*?)(?:\n\s*(?:A\.|A\))\s*(?P<A>.+?))"
        r"(?:\n\s*(?:B\.|B\))\s*(?P<B>.+?))"
        r"(?:\n\s*(?:C\.|C\))\s*(?P<C>.+?))"
        r"(?:\n\s*(?:D\.|D\))\s*(?P<D>.+?))"
    )
    m = re.search(pattern, query, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return query, None
        
    stem = m.group(1).strip()
    options = {
        "A": m.group("A").strip(),
        "B": m.group("B").strip(),
        "C": m.group("C").strip(), 
        "D": m.group("D").strip()
    }
    return stem, options

def format_numbered_list(text: str) -> str:
    """
    Sửa lỗi định dạng các đề mục số bị xuống dòng không hợp lý.
    Chỉ gộp dòng số thứ tự trống (vd: "1.") với duy nhất 1 dòng nội dung tiếp theo.
    """
    number_pattern = r'^(?:\d+[\.\)]|[a-zA-Z][\.\)]|[-*])\s*$'
    
    lines = text.split('\n')
    result = []
    i = 0
    n = len(lines)
    
    while i < n:
        line = lines[i]
        line_stripped = line.strip()
        
        # Nếu dòng hiện tại khớp với số thứ tự trống
        if re.match(number_pattern, line_stripped):
            # Tìm dòng tiếp theo có nội dung
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            
            if j < n:
                # Gộp dòng số với dòng nội dung đó
                combined = f"{line_stripped} {lines[j].strip()}"
                result.append(combined)
                i = j + 1
            else:
                result.append(line)
                i += 1
        else:
            result.append(line)
            i += 1

    text2 = '\n'.join(result)
    # Loại bỏ khoảng trắng thừa ở giữa dòng nhưng KHÔNG thay thế newline bằng space
    text2 = re.sub(r'[ \t]{2,}', ' ', text2)
    return text2

def answer_with_context(query: str) -> str:
    """
    Trả lời câu hỏi dựa trên văn bản luật có liên quan.
    Hỗ trợ cả câu hỏi trắc nghiệm và tự luận.
    """
    # Parse câu hỏi
    stem, options = parse_mcq(query)
    
    # Lấy các chunk văn bản liên quan
    chunks = retrieve_similar_chunks(query, top_k=TOP_K, threshold=SIMILARITY_THRESHOLD)
    
    # Phân tích query để xác định loại luật cần tìm
    query_lower = query.lower()
    is_civil_law = "dân sự" in query_lower
    is_criminal_law = "hình sự" in query_lower
    
    # Kiểm tra xem có chunk nào vượt ngưỡng similarity không
    has_relevant_chunks = any(c["score"] >= SIMILARITY_THRESHOLD for c in chunks)
    
    if not has_relevant_chunks:
        # Fallback sang Gemini ngay nếu không có chunk nào đủ liên quan
        general_prompt = (
            "Bạn là luật sư tư vấn pháp luật Việt Nam. Hãy trả lời câu hỏi sau một cách chính xác, chuyên nghiệp "
            "và dễ hiểu. Trả lời ngắn gọn, súc tích, đúng trọng tâm. Nếu không chắc chắn, hãy khuyến nghị người dùng tham khảo ý kiến luật sư.\n\n"
            f"Câu hỏi: {query}\n\n"
        )
        return "[Tổng quát] " + gemini_answer(general_prompt)

    # Lọc chunks theo score và ưu tiên theo loại luật
    selected = []
    for c in chunks:
        if c["score"] < SIMILARITY_THRESHOLD:
            continue
            
        source = c["meta"].get("source", "").lower()
        is_civil = "luật-91-2015" in source
        is_criminal = "luật-100-2015" in source
        
        if is_civil_law and is_civil:
            selected.append(c)
        elif is_criminal_law and is_criminal:
            selected.append(c)
        elif not is_civil_law and not is_criminal_law:
            selected.append(c)

    # Tạo context từ các chunk được chọn
    context = ""
    for idx, c in enumerate(selected):
        meta = c["meta"]
        source = meta.get("source", "Unknown")
        dieu = meta.get("dieu", "")
        khoan = meta.get("khoan", "")
        header = f"[Đoạn {idx+1} – {source}"
        if dieu:
            header += f", Điều {dieu}"
        if khoan:
            header += f", Khoản {khoan}"
        header += "]\n"
        
        chunk_text = c["text"]
        chunk_text = chunk_text.replace("\n", " ").strip()
        chunk_text = re.sub(r"\s+", " ", chunk_text)
        chunk_text = re.sub(r"\s*\*+\s*", " ", chunk_text)
        context += header + chunk_text.strip() + "\n\n"

    # Tạo prompt phù hợp với loại câu hỏi
    if options:
        opts_text = ""
        for key, val in options.items():
            opts_text += f"{key}. {val}\n"
        prompt = (
            "Bạn là luật sư chuyên về pháp luật Việt Nam với nhiều năm kinh nghiệm. "
            "Hãy trả lời câu hỏi dựa trên các đoạn văn bản luật được cung cấp dưới đây. "
            "Trả lời ngắn gọn, súc tích, đúng trọng tâm. Nếu thông tin không đầy đủ, hãy trả lời dựa trên kiến thức pháp luật tổng quát.\n\n"
            f"Câu hỏi (MCQ): {stem}\n"
            "Phương án:\n" + opts_text + "\n"
            f"Văn bản luật liên quan:\n{context}\n"
            "Hãy chọn 1 trong 4 phương án (A/B/C/D) và giải thích ngắn gọn lý do chọn. "
            "Nếu không tìm thấy thông tin phù hợp trong văn bản luật, "
            "hãy trả lời dựa trên hiểu biết chung về pháp luật Việt Nam."
        )
    else:
        prompt = (
            "Bạn là luật sư chuyên về pháp luật Việt Nam với nhiều năm kinh nghiệm. "
            "Hãy trả lời câu hỏi dựa trên các đoạn văn bản luật được cung cấp dưới đây. "
            "Trả lời ngắn gọn, súc tích, đúng trọng tâm. Nếu thông tin không đầy đủ, hãy trả lời dựa trên kiến thức pháp luật tổng quát.\n\n"
            f"Câu hỏi: {query}\n\n"
            f"Văn bản luật liên quan:\n{context}\n"
            "Nêu kết luận rõ ràng, trích dẫn điều luật nếu có. Nếu không tìm thấy thông tin phù hợp trong văn bản luật, "
            "hãy trả lời dựa trên hiểu biết chung về pháp luật Việt Nam."
        )

    # Gọi Gemini và format câu trả lời
    answer = gemini_answer(prompt)
    answer = format_numbered_list(answer)
    
    # Kiểm tra xem câu trả lời có sử dụng thông tin từ văn bản luật không
    lower_ans = answer.lower()
    keywords_no_direct_info = [
        "xin lỗi, không đủ dữ liệu",
        "xin lỗi, không thể trả lời",
        "rất tiếc, không có đủ thông tin",
        "không tìm thấy thông tin",
        "không có thông tin",
        "không tìm thấy quy định",
        "không có quy định"
    ]
    
    uses_legal_context = (
        "điều" in lower_ans or
        "khoản" in lower_ans or
        "luật" in lower_ans or
        "quy định" in lower_ans or
        "theo" in lower_ans
    )
    
    # Quyết định prefix dựa trên việc có sử dụng thông tin pháp luật không
    prefix = "[LawBot (RAG)] " if uses_legal_context else "[Tổng quát] "
    
    # Thêm disclaimer
    disclaimer = (
        "\n\n---\n"
        "**Khuyến nghị pháp lý:** Nếu bạn hoặc ai đó bạn biết đang gặp vấn đề liên quan đến pháp luật, "
        "hãy tìm kiếm sự tư vấn của luật sư để được hỗ trợ pháp lý tốt nhất. "
        "Thông tin này không thay thế cho tư vấn pháp lý chuyên nghiệp. "
        "Bạn nên tham khảo ý kiến của luật sư hoặc chuyên gia pháp lý để được tư vấn cụ thể trong từng trường hợp."
    )
    
    return prefix + answer + disclaimer

# --------------------------------------------
# Preload chunks khi import module (tùy chọn)
# --------------------------------------------
def preload_index():
    """Gọi hàm này để preload index khi khởi động app."""
    load_all_chunks()

# --------------------------------------------
# PHẦN TEST KHI CHẠY TRỰC TIẾP
# --------------------------------------------
if __name__ == "__main__":
    print("\n=== TEST TRẮC NGHIỆM ===")
    test_mcq = """
Theo quy định của Bộ luật Dân sự, người từ đủ 6 tuổi đến chưa đủ 18 tuổi có thể:
A. Tự mình xác lập và thực hiện giao dịch dân sự
B. Chỉ có thể thực hiện giao dịch thông qua người đại diện
C. Tự mình xác lập và thực hiện giao dịch dân sự nếu được cha mẹ đồng ý
D. Tự mình xác lập, thực hiện các giao dịch dân sự phục vụ nhu cầu sinh hoạt hàng ngày
    """
    print("Câu hỏi trắc nghiệm mẫu:")
    print(test_mcq)
    print("\nKết quả trả lời:")
    print(answer_with_context(test_mcq))

    print("\n=== TEST TỰ LUẬN ===")
    test_essay = "Trình bày các quy định về năng lực hành vi dân sự của người từ đủ 6 tuổi đến chưa đủ 18 tuổi?"
    print("Câu hỏi tự luận mẫu:")
    print(test_essay)
    print("\nKết quả trả lời:")
    print(answer_with_context(test_essay))
