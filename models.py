from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """Mô hình User để lưu thông tin người dùng và quản lý giới hạn câu hỏi"""
    # Các trường cơ bản
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    phone = db.Column(db.String(15), nullable=True)
    
    # Các trường xác thực email
    email_verified = db.Column(db.Boolean, default=False)
    verification_code = db.Column(db.String(6), nullable=True)
    code_expiry = db.Column(db.DateTime, nullable=True)
    
    # Các trường theo dõi câu hỏi
    questions_asked = db.Column(db.Integer, default=0)      # Tổng số câu hỏi đã hỏi
    last_question_date = db.Column(db.Date, default=None)   # Ngày hỏi câu hỏi cuối
    daily_questions = db.Column(db.Integer, default=0)      # Số câu hỏi trong ngày
    
    # Các trường subscription (để sau này mở rộng)
    subscription_expiry = db.Column(db.DateTime, nullable=True)
    subscription_type = db.Column(db.String(20), nullable=True)
    
    # Cấu hình giới hạn câu hỏi
    MAX_ANONYMOUS_QUESTIONS = 5    # Số câu tối đa cho người chưa đăng nhập
    MAX_DAILY_QUESTIONS = 20      # Số câu tối đa mỗi ngày cho người đã đăng nhập
    
    def set_password(self, password):
        """Mã hóa và lưu mật khẩu"""
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        """Kiểm tra mật khẩu"""
        return check_password_hash(self.password_hash, password)
        
    def reset_daily_questions(self):
        """Reset số câu hỏi hàng ngày nếu sang ngày mới"""
        today = datetime.date.today()
        if self.last_question_date != today:
            self.daily_questions = 0
            self.last_question_date = today
            db.session.commit()
            
    def can_ask_question(self):
        """Kiểm tra xem người dùng còn có thể hỏi câu hỏi không"""
        self.reset_daily_questions()  # Reset số câu nếu sang ngày mới
        return self.daily_questions < self.MAX_DAILY_QUESTIONS
        
    def increment_question_count(self):
        """Tăng số câu hỏi sau khi hỏi thành công"""
        self.reset_daily_questions()  # Đảm bảo đã reset nếu sang ngày mới
        self.questions_asked += 1     # Tăng tổng số câu hỏi
        self.daily_questions += 1     # Tăng số câu hỏi trong ngày
        db.session.commit()