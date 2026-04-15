# 1️ raw_frames + clahe 합치기
# 2️ dataset/images / dataset/labels 구조 만들기
# 3️ train / val 분리 (80:20)

import os
import shutil
import random

# =========================
# 경로 설정
# =========================
BASE_PATH = r"D:\smart_tunnel_V3\data"

RAW_IMG = os.path.join(BASE_PATH, "raw_frames", "images")
RAW_LABEL = os.path.join(BASE_PATH, "raw_frames", "labels")

CLAHE_IMG = os.path.join(BASE_PATH, "raw_frames", "clahe_images")
CLAHE_LABEL = os.path.join(BASE_PATH, "raw_frames", "clahe_labels")

DATASET = os.path.join(BASE_PATH, "dataset")

IMG_TRAIN = os.path.join(DATASET, "images", "train")
IMG_VAL = os.path.join(DATASET, "images", "val")

LABEL_TRAIN = os.path.join(DATASET, "labels", "train")
LABEL_VAL = os.path.join(DATASET, "labels", "val")

# 폴더 생성
for path in [IMG_TRAIN, IMG_VAL, LABEL_TRAIN, LABEL_VAL]:
    os.makedirs(path, exist_ok=True)

# =========================
# 전체 파일 수집
# =========================
all_images = []

for f in os.listdir(RAW_IMG):
    if f.endswith(".jpg"):
        all_images.append(("raw", f))

for f in os.listdir(CLAHE_IMG):
    if f.endswith(".jpg"):
        all_images.append(("clahe", f))

print(f"총 데이터 수: {len(all_images)}")

# =========================
# 랜덤 셔플
# =========================
random.shuffle(all_images)

# =========================
# split
# =========================
split_idx = int(len(all_images) * 0.8)

train_set = all_images[:split_idx]
val_set = all_images[split_idx:]

print(f"train: {len(train_set)}")
print(f"val: {len(val_set)}")

# =========================
# 복사 함수
# =========================
def copy_data(dataset, img_dst, label_dst):
    for dtype, fname in dataset:

        if dtype == "raw":
            img_src = os.path.join(RAW_IMG, fname)
            label_src = os.path.join(RAW_LABEL, fname.replace(".jpg", ".txt"))
        else:
            img_src = os.path.join(CLAHE_IMG, fname)
            label_src = os.path.join(CLAHE_LABEL, fname.replace(".jpg", ".txt"))

        shutil.copy(img_src, os.path.join(img_dst, fname))

        if os.path.exists(label_src):
            shutil.copy(label_src, os.path.join(label_dst, fname.replace(".jpg", ".txt")))

# =========================
# 실행
# =========================
copy_data(train_set, IMG_TRAIN, LABEL_TRAIN)
copy_data(val_set, IMG_VAL, LABEL_VAL)

print("✅ dataset 생성 완료")
