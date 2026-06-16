@echo off
echo ========================================
echo    VietLawBot Lite - Starting...
echo ========================================
echo.

REM Kiểm tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python không được cài đặt hoặc không có trong PATH
    pause
    exit /b 1
)

REM Kiểm tra virtual environment
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Đang kích hoạt virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] Không tìm thấy venv, sử dụng Python global
)

REM Kiểm tra dependencies
echo [INFO] Kiểm tra dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo [INFO] Đang cài đặt dependencies...
    pip install -r requirements_lite.txt
)

echo.
echo [INFO] Khởi động VietLawBot Lite trên port 5003...
echo [INFO] Mở trình duyệt: http://localhost:5003
echo.

python app_lite.py

pause
