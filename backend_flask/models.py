from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    user_id = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(20))
    email = db.Column(db.String(100))

    @property
    def password(self):
        raise AttributeError("비밀번호는 직접 읽을 수 없습니다.")
    
    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)


class DetectionResult(db.Model):
    __tablename__ = 'detection_results'
    id = db.Column(db.Integer, primary_key=True)
    
    event_type    = db.Column(db.String(20))  
    address       = db.Column(db.String(200), default="서울시 관제구역")
    latitude      = db.Column(db.Float, nullable=True)
    longitude     = db.Column(db.Float, nullable=True)
    detected_at   = db.Column(db.DateTime, default=datetime.now)
    is_simulation = db.Column(db.Boolean, default=False, nullable=False)
    video_origin  = db.Column(db.String(50), nullable=True)  # ✅ 추가: 'webcam' | 'fire' | 'reverse' | 'realtime_its' | None

    is_resolved = db.Column(db.Boolean, default=False)
    resolved_at = db.Column(db.DateTime, nullable=True)
    feedback    = db.Column(db.Boolean, nullable=True)
    resolved_by = db.Column(db.String(50), db.ForeignKey('users.name'), nullable=True)
    resolver    = db.relationship('User', backref='resolved_cases')

    fire_detail    = db.relationship('FireResult',    backref='base_result', uselist=False, cascade="all, delete-orphan")
    reverse_detail = db.relationship('ReverseResult', backref='base_result', uselist=False, cascade="all, delete-orphan")
    manual_detail  = db.relationship('ManualResult',  backref='base_result', uselist=False, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.event_type,
            "address": self.address,
            "lat": self.latitude,
            "lng": self.longitude,
            "time": self.detected_at.strftime('%Y-%m-%d %H:%M:%S'),
            "is_simulation": self.is_simulation,
            "video_origin": self.video_origin,  # ✅ 추가
            "is_resolved": self.is_resolved,
            "feedback": self.feedback,
            "resolved_at": self.resolved_at.strftime('%Y-%m-%d %H:%M:%S') if self.resolved_at else None,
            "resolved_by_name": self.resolver.name if self.resolver else self.resolved_by
        }


class FireResult(db.Model):
    __tablename__ = 'fire_results'
    id            = db.Column(db.Integer, primary_key=True)
    result_id     = db.Column(db.Integer, db.ForeignKey('detection_results.id'), nullable=False)
    image_path    = db.Column(db.String(255), nullable=True)
    fire_severity = db.Column(db.String(20)) 

class ReverseResult(db.Model):
    __tablename__ = 'reverse_results'
    id           = db.Column(db.Integer, primary_key=True)
    result_id    = db.Column(db.Integer, db.ForeignKey('detection_results.id'), nullable=False)
    image_path   = db.Column(db.String(255), nullable=True)
    vehicle_info = db.Column(db.String(50)) 

class ManualResult(db.Model):
    __tablename__ = 'manual_results'
    id        = db.Column(db.Integer, primary_key=True)
    result_id = db.Column(db.Integer, db.ForeignKey('detection_results.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=True)
    memo      = db.Column(db.Text, nullable=True)


class MonitoringAction(db.Model):
    """교통 모니터링 팀 — 관제사 대응 조치 기록"""
    __tablename__ = 'monitoring_actions'
    id          = db.Column(db.Integer, primary_key=True)
    camera_id   = db.Column(db.String(64),  nullable=False)
    action_type = db.Column(db.String(64),  nullable=False)   # VSL_DOWN, VMS_SLOW 등
    action_at   = db.Column(db.DateTime,    default=datetime.utcnow)
    acted_by    = db.Column(db.String(64),  nullable=True)    # 관리자 이름


class PlateResult(db.Model):
    __tablename__ = 'plate_results'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # 원본 인식 데이터
    plate_number = db.Column(db.String(50), nullable=False)
    ground_truth = db.Column(db.String(50), nullable=True)
    is_correct   = db.Column(db.Boolean, nullable=True)
    
    # 신뢰도/투표
    confidence = db.Column(db.Float, nullable=True)
    vote_count = db.Column(db.Integer, nullable=True)
    is_fixed   = db.Column(db.Boolean, default=False)
    
    # 파일 관련
    img_path       = db.Column(db.String(255), nullable=True)
    video_filename = db.Column(db.String(255), nullable=True)
    
    # ✅ 추가: 인식을 실행한 관리자 정보
    operator_name = db.Column(db.String(50), db.ForeignKey('users.name'), nullable=True)
    operator      = db.relationship('User', backref='plate_results')
    
    # 시간
    detected_at = db.Column(db.DateTime, default=datetime.now)
    created_at  = db.Column(db.DateTime, default=datetime.now)
    updated_at  = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 관계
    preprocess_results = db.relationship(
        'PlatePreprocessResult',
        backref='original_result',
        cascade="all, delete-orphan"
    )


class PlatePreprocessResult(db.Model):
    """번호판 전처리 결과"""
    __tablename__ = 'plate_preprocess_results'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # 관계
    result_id = db.Column(
        db.Integer, 
        db.ForeignKey('plate_results.id'), 
        nullable=False
    )
    
    # 전처리 정보
    preprocess_method = db.Column(db.String(50), nullable=False)   # 전처리방법
    corrected_text = db.Column(db.String(50), nullable=False)      # 보정후번호판
    
    # 검증
    ground_truth = db.Column(db.String(50), nullable=True)  # 정답번호판
    is_correct = db.Column(db.Boolean, nullable=True)       # 정오여부
    
    # 성능
    elapsed_ms = db.Column(db.Integer, nullable=True)       # 처리시간(ms)
    img_path = db.Column(db.String(255), nullable=True)     # 이미지경로
    
    # 시간
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)