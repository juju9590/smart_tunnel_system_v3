from ultralytics import YOLO
import os

# =========================
# 1. 설정
# =========================
MODEL_NAME = "yolo11n.pt"
DATA_PATH = "../data/data.yaml"

EPOCHS = 50
IMGSZ = 640
BATCH = 8

PROJECT_NAME = "runs/train"
RUN_NAME = "tunnel_final"

# =========================
# 2. 모델 로드
# =========================
print("🚀 모델 로드 중...")
model = YOLO(MODEL_NAME)

# =========================
# 3. 학습 실행
# =========================
print("🔥 학습 시작")

results = model.train(
    data=DATA_PATH,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    workers=2,
    device="cpu",

    # 🔥 성능 관련 옵션
    optimizer="AdamW",     # 기본 SGD보다 안정적
    lr0=0.001,             # 초기 learning rate
    cos_lr=True,           # cosine scheduler

    # 🔥 augmentation (중요)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,

    # 🔥 기타
    project=PROJECT_NAME,
    name=RUN_NAME,
    exist_ok=True,

    # 🔥 로그
    verbose=True
)

print("✅ 학습 완료")

# =========================
# 4. best 모델 출력
# =========================
best_model_path = os.path.join(PROJECT_NAME, RUN_NAME, "weights", "best.pt")

print(f"\n🏆 Best Model: {best_model_path}")