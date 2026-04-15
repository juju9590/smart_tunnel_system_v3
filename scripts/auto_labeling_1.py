import os
import random
import shutil
from ultralytics import YOLO # (YOLOv8, YOLOv11 등)를 설치할 때 가장 표준적인 방법
# pip install ultralytics를 하면 기본적으로 PyTorch가 함께 설치
# ultralytics는 PyTorch라는 엔진으로 돌아가는 소프트웨어

# =========================
# 1. 경로 설정
# =========================
BASE_PATH = r"D:\smart_tunnel_V3\data"

SOURCE_PATH = os.path.join(BASE_PATH, "sample_frames")
OUTPUT_IMG_PATH = os.path.join(BASE_PATH, "raw_frames", "images")
OUTPUT_LABEL_PATH = os.path.join(BASE_PATH, "raw_frames", "labels")

os.makedirs(OUTPUT_IMG_PATH, exist_ok=True)
os.makedirs(OUTPUT_LABEL_PATH, exist_ok=True)

# =========================
# 2. 클래스별 개수 설정
# =========================
LIMITS = {
    "accident": 220,
    "congestion": 440,
    "normal": 440
}

# =========================
# 3. YOLO 모델
# =========================
model = YOLO("yolov8s.pt")  # 또는 yolov8n.pt

# =========================
# 4. 차량 클래스
# =========================
VEHICLE_CLASSES = [2, 5, 7]  # car, bus, truck

# =========================
# 5. 처리 시작
# =========================
img_id = 0

for category, limit in LIMITS.items():
    folder_path = os.path.join(SOURCE_PATH, category)

    all_images = [f for f in os.listdir(folder_path)
                  if f.endswith((".jpg", ".png"))]

    selected = random.sample(all_images, min(limit, len(all_images)))

    print(f"\n📂 {category} → {len(selected)}개 선택")

    for img_name in selected:
        img_path = os.path.join(folder_path, img_name)

        # =========================
        # YOLO 자동 라벨링
        # =========================
        results = model.predict(
            source=img_path,
            conf=0.25,
            save=True,    # 이미지 저장
            save_txt=True, # 라벨 저장
            save_conf=True,
            project="runs/auto_tmp",
            name="predict",
            exist_ok=True
        )

        # =========================
        # 결과 라벨 경로
        # =========================
        label_path = os.path.join(
            "runs/auto_tmp/predict/labels",
            os.path.splitext(img_name)[0] + ".txt"
        )

        # =========================
        # 새 파일명 (이미지명 = 라벨명 => img_00001.jpg, img_00001.txt)
        # =========================
        new_name = f"img_{img_id:05d}.jpg"
        new_label = f"img_{img_id:05d}.txt"

        # =========================
        # 라벨 존재 확인
        # =========================
        if os.path.exists(label_path):

            # =========================
            # 라벨 변환 (차량만)
            # =========================
            with open(label_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                cls = int(parts[0])

                if cls in VEHICLE_CLASSES:
                    parts[0] = "0"
                    new_lines.append(" ".join(parts))

            # =========================
            # 차량 없는 이미지 처리
            # =========================
            if len(new_lines) == 0:
                if random.random() > 0.1:  # 10%만 유지
                    continue

            # =========================
            # 이미지 복사
            # =========================
            shutil.copy(img_path, os.path.join(OUTPUT_IMG_PATH, new_name))

            # =========================
            # 라벨 저장
            # =========================
            with open(os.path.join(OUTPUT_LABEL_PATH, new_label), "w") as f:
                f.write("\n".join(new_lines))

            img_id += 1

print("\n✅ 자동 라벨링 + 이동 완료")