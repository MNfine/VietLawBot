#!/bin/bash
echo "========================================"
echo "   VietLawBot Lite - Starting..."
echo "========================================"
echo ""

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 không được cài đặt"
    exit 1
fi

# Kiểm tra virtual environment
if [ -f "venv/bin/activate" ]; then
    echo "[INFO] Đang kích hoạt virtual environment..."
    source venv/bin/activate
else
    echo "[INFO] Không tìm thấy venv, sử dụng Python global"
fi

# Kiểm tra dependencies
echo "[INFO] Kiểm tra dependencies..."
if ! python3 -c "import flask" 2>/dev/null; then
    echo "[INFO] Đang cài đặt dependencies..."
    pip install -r requirements_lite.txt
fi

echo ""
echo "[INFO] Khởi động VietLawBot Lite trên port 5003..."
echo "[INFO] Mở trình duyệt: http://localhost:5003"
echo ""

python3 app_lite.py
