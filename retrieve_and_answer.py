# retrieve_and_answer.py

import os
import json
import re
from typing import Tuple, Optional, Dict, Any

# Load các biến môi trường
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import google.generativeai as genai
from chunk_and_index import retrieve_similar_chunks

# --------------------------------------------
# CẤU HÌNH CHUNG
# --------------------------------------------
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

# Số chunk lấy về và ngưỡng similarity
TOP_K = 3 
SIMILARITY_THRESHOLD = 0.6
SHOW_DEBUG = False  # Thêm flag để kiểm soát việc in debug info

def gemini_answer(prompt: str) -> str:
    """Gọi Gemini để sinh câu trả lời."""
    gm = genai.GenerativeModel("models/gemini-2.0-flash")
    resp = gm.generate_content(prompt)
    return resp.text.strip()

def parse_mcq(query: str) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    Tách câu hỏi trắc nghiệm thành phần đề bài và các phương án.
    Args:
        query: Câu hỏi trắc nghiệm đầy đủ
    Returns:
        tuple: (stem, options) trong đó
            - stem: phần đề bài
            - options: dict các phương án hoặc None nếu không phải MCQ
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
    Hỗ trợ nhiều định dạng số: 1., 1), a., a), i., i), etc.
    """
    # Pattern nhận diện các loại số thứ tự
    number_pattern = r'^(?:\d+\.|\d+\)|\w+\.|\w+\)|[ivxIVX]+\.|\([^)]+\))\s*$'
    
    lines = text.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if re.match(number_pattern, line):
            next_content = []
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line:
                    next_content.append(next_line)
                    if re.match(number_pattern, next_line):
                        break
                    j += 1
                else:
                    j += 1
                    if j < len(lines) and not lines[j].strip():
                        break
            
            if next_content:
                combined = f"{line} {' '.join(next_content)}"
                result.append(combined)
                i = j
                continue
            else:
                result.append(line)
        else:
            result.append(line)
        i += 1

    text2 = '\n'.join(result)
    text2 = re.sub(r'\n{3,}', '\n\n', text2)
    text2 = re.sub(r'\s{2,}', ' ', text2)
    return text2

def answer_with_context(query: str) -> str:
    """
    Trả lời câu hỏi dựa trên văn bản luật có liên quan.
    Hỗ trợ cả câu hỏi trắc nghiệm và tự luận.
    Args:
        query: Câu hỏi cần trả lời
    Returns:
        str: Câu trả lời từ Gemini
    """
    # Parse câu hỏi
    stem, options = parse_mcq(query)
    
    # Lấy các chunk văn bản liên quan
    chunks = retrieve_similar_chunks(query, top_k=TOP_K)
    
    # Phân tích query để xác định loại luật cần tìm
    query_lower = query.lower()
    is_civil_law = "dân sự" in query_lower
    is_criminal_law = "hình sự" in query_lower
    
    # Thông tin tóm tắt về chunks tìm được (chỉ hiện khi debug)
    if chunks and SHOW_DEBUG:
        print(f"\nTìm thấy {len(chunks)} chunk phù hợp (score > {SIMILARITY_THRESHOLD}):")
        for i, c in enumerate(chunks, 1):
            meta = c["meta"]
            source = meta.get("source", "Unknown")
            dieu = meta.get("dieu", "")
            score = c.get("score", 0.0)
            print(f"[{i}] Score: {score:.3f} | {source}{' Điều ' + dieu if dieu else ''}")
        print()
    
    # Lọc chunks theo score và ưu tiên theo loại luật
    selected = []
    for c in chunks:
        if c["score"] < SIMILARITY_THRESHOLD:
            continue
            
        source = c["meta"].get("source", "").lower()
        is_civil = "luật-91-2015" in source
        is_criminal = "luật-100-2015" in source
        
        # Nếu query liên quan đến luật dân sự -> ưu tiên chunks từ BLDS
        if is_civil_law and is_civil:
            selected.append(c)
        # Nếu query liên quan đến luật hình sự -> ưu tiên chunks từ BLHS
        elif is_criminal_law and is_criminal:
            selected.append(c)
        # Nếu query không nêu rõ loại luật -> lấy tất cả chunks có score cao
        elif not is_civil_law and not is_criminal_law:
            selected.append(c)
            
    if SHOW_DEBUG:
        print(f"Đã chọn {len(selected)} chunks cho quá trình RAG.\n")
    
    if not selected:
        return "[Gemini tổng quát] " + gemini_answer(query)

    # Tạo context từ các chunk
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
        
        # Làm sạch text nhưng giữ nguyên cấu trúc
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
            "Trả lời chính xác, ngắn gọn, dễ hiểu và chỉ dựa vào các điều luật liên quan.\n\n"
            f"Câu hỏi (MCQ): {stem}\n"
            "Phương án:\n" + opts_text + "\n"
            f"Văn bản luật liên quan:\n{context}\n"
            "Hãy chọn 1 trong 4 phương án (A/B/C/D) dựa vao văn bản luật ở trên. "
            "Nếu không đủ dữ liệu để trả lời, hãy nói rõ là không đủ dữ liệu. "
            "Nếu có thể, giải thích ngắn gọn."
        )
    else:
        prompt = (
            "Bạn là luật sư chuyên về pháp luật Việt Nam với nhiều năm kinh nghiệm. "
            "Hãy trả lời câu hỏi dựa trên các đoạn văn bản luật được cung cấp dưới đây. "
            "Trả lời chính xác, ngắn gọn, dễ hiểu và chỉ dựa vào các điều luật liên quan.\n\n"
            f"Câu hỏi: {query}\n\n"
            f"Văn bản luật liên quan:\n{context}\n"
            "Trả lời ngắn gọn, rõ ràng, trích dẫn chính xác Điều, Khoản nếu có. "
            "Nếu không đủ dữ liệu để trả lời, hãy nói rõ là không đủ dữ liệu."
        )

    # Gọi Gemini và format câu trả lời
    rag_answer = gemini_answer(prompt)
    rag_answer = format_numbered_list(rag_answer)
    
    # Kiểm tra xem có dùng được thông tin từ văn bản luật không
    lower_ans = rag_answer.lower()
    # Các từ khóa chỉ ra rằng không có thông tin phù hợp trong văn bản luật
    keywords_no_info = [
        "xin lỗi, không đủ dữ liệu",
        "xin lỗi, không thể trả lời",
        "rất tiếc, không có đủ thông tin",
        "không tìm thấy thông tin liên quan",
        "không có thông tin phù hợp"
    ]
    
    # Thêm disclaimer vào cuối câu trả lời
    disclaimer = (
        "\n\nLưu ý: Nếu bạn hoặc ai đó bạn biết đang gặp vấn đề liên quan đến pháp luật, "
        "hãy tìm kiếm sự tư vấn của luật sư để được hỗ trợ pháp lý tốt nhất. "
        "* Thông tin này không thay thế cho tư vấn pháp lý chuyên nghiệp. "
        "Bạn nên tham khảo ý kiến của luật sư hoặc chuyên gia pháp lý để được tư vấn cụ thể trong từng trường hợp."
    )
    
    if not context or any(kw in lower_ans for kw in keywords_no_info):
        print("=> Chuyển sang Gemini tổng quát do không có thông tin phù hợp")
        final_answer = "[Gemini tổng quát] " + gemini_answer(query)
    else:
        print("=> Sử dụng LawBot RAG với context từ văn bản luật")
        final_answer = "[LawBot (RAG)] " + rag_answer
    
    return final_answer + disclaimer

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
