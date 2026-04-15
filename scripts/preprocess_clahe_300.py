# ✔ 상황별 (파일명 기반) 자동 분리
# ✔ 비율 맞춰 랜덤 선택 (정상120 / 혼잡120 / 사고60)
# ✔ CLAHE 적용
# ✔ 라벨 자동 복사 + 이름 변경

import os
import cv2
import random
import shutil

# =========================
# 경로 설정
# =========================
BASE_PATH = r"D:\smart_tunnel_V3\data\raw_frames"

IMAGE_PATH = os.path.join(BASE_PATH, "images")
LABEL_PATH = os.path.join(BASE_PATH, "labels")

OUTPUT_IMG = os.path.join(BASE_PATH, "clahe_images")
OUTPUT_LABEL = os.path.join(BASE_PATH, "clahe_labels")

os.makedirs(OUTPUT_IMG, exist_ok=True)
os.makedirs(OUTPUT_LABEL, exist_ok=True)

# =========================
# CLAHE 함수
# =========================
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# =========================
# 상황별 분류
# =========================
normal_files = []
congestion_files = []
accident_files = []

for file in os.listdir(IMAGE_PATH):
    if not file.endswith(".jpg"):
        continue

    name = file.lower()

    if "normal" in name:
        normal_files.append(file)
    elif "congestion" in name:
        congestion_files.append(file)
    elif "accident" in name:
        accident_files.append(file)

print(f"정상: {len(normal_files)}")
print(f"혼잡: {len(congestion_files)}")
print(f"사고: {len(accident_files)}")

# =========================
# 비율 설정
# =========================
TARGET = {
    "normal": 120,
    "congestion": 120,
    "accident": 60
}

# =========================
# 랜덤 선택
# =========================
selected_normal = random.sample(normal_files, min(TARGET["normal"], len(normal_files)))
selected_congestion = random.sample(congestion_files, min(TARGET["congestion"], len(congestion_files)))
selected_accident = random.sample(accident_files, min(TARGET["accident"], len(accident_files)))

selected_all = selected_normal + selected_congestion + selected_accident

print(f"총 선택: {len(selected_all)}")

# =========================
# CLAHE 적용 + 라벨 복사
# =========================
for img_name in selected_all:
    img_path = os.path.join(IMAGE_PATH, img_name)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(LABEL_PATH, label_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    # CLAHE 적용
    clahe_img = apply_clahe(img)

    # 새 이름
    new_img_name = os.path.splitext(img_name)[0] + "_clahe.jpg"
    new_label_name = os.path.splitext(img_name)[0] + "_clahe.txt"

    # 이미지 저장
    cv2.imwrite(os.path.join(OUTPUT_IMG, new_img_name), clahe_img)

    # 라벨 복사 (이름 변경)
    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(OUTPUT_LABEL, new_label_name))

print("✅ CLAHE 데이터 생성 완료")