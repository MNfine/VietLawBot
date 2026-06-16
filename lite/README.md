# VietLawBot Lite

Phiên bản nhẹ của VietLawBot - **Không sử dụng Docker/Redis**.

## Đặc điểm

- ✅ Đọc trực tiếp văn bản pháp luật từ thư mục `plain_texts`
- ✅ Sử dụng Sentence Transformers (LaBSE) để tìm kiếm ngữ nghĩa trong memory
- ✅ Gọi Gemini API để sinh câu trả lời
- ✅ Không cần Docker, không cần Redis
- ✅ Dễ dàng triển khai trên máy local

## Cấu trúc

```
lite/
├── app_lite.py                 # Flask application chính
├── retrieve_and_answer_lite.py # Logic RAG không dùng Redis
├── models_lite.py              # Database models
├── requirements_lite.txt       # Dependencies
├── templates/                  # HTML templates
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── verify_email.html
│   ├── news.html
│   └── lawyer.html
└── README.md
```

## Cài đặt

### 1. Tạo virtual environment (khuyến nghị)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac  
source venv/bin/activate
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements_lite.txt
```

### 3. Cấu hình biến môi trường

Tạo file `.env` trong thư mục `lite/`:

```env
# Bắt buộc: Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# Tùy chọn: Email (cho chức năng đăng ký)
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_app_password

# Tùy chọn: Google OAuth
GOOGLE_OAUTH_CLIENT_ID=your_client_id
GOOGLE_OAUTH_CLIENT_SECRET=your_client_secret

# Tùy chọn: HuggingFace (để load model nhanh hơn)
HUGGINGFACE_TOKEN=your_token
```

### 4. Chạy ứng dụng

```bash
python app_lite.py
```

Mở trình duyệt: http://localhost:5003

## Cách hoạt động

1. **Khởi động**: 
   - Load tất cả văn bản từ `../plain_texts/`
   - Chia thành chunks nhỏ (400 tokens, overlap 50)
   - Tạo embeddings cho tất cả chunks (sử dụng LaBSE model)

2. **Khi có câu hỏi**:
   - Tạo embedding cho câu hỏi
   - Tính cosine similarity với tất cả chunks
   - Lấy top-k chunks có similarity cao nhất
   - Nếu có chunk vượt ngưỡng (0.6) → Dùng context + Gemini
   - Nếu không → Chỉ dùng Gemini (tổng quát)

3. **Sinh câu trả lời**:
   - Gemini nhận context từ văn bản luật
   - Trả lời có trích dẫn điều luật nếu có

## So sánh với phiên bản gốc

| Tính năng | Phiên bản gốc | Lite |
|-----------|---------------|------|
| Database vector | Redis Stack | In-memory (numpy) |
| Container | Docker | Không cần |
| Startup time | Nhanh (đã index) | Chậm hơn (load mỗi lần) |
| Memory usage | Thấp | Cao hơn |
| Phù hợp | Production | Development/Demo |

## Lưu ý

- Lần đầu chạy sẽ mất thời gian để download model LaBSE (~1.8GB)
- Embeddings được cache trong memory, restart sẽ phải load lại
- Để giảm memory, có thể giảm số chunks hoặc dùng model nhẹ hơn

## Troubleshooting

### Lỗi "GOOGLE_API_KEY not set"
→ Kiểm tra file `.env` hoặc set biến môi trường

### Lỗi "Model not found"
→ Kiểm tra kết nối internet để download model LaBSE

### Lỗi "Out of memory"
→ Giảm `chunk_size` trong `retrieve_and_answer_lite.py`
