# 라벨 정리 스크립트    
# 1. 라벨 없는 이미지 → 빈 라벨 생성
# 2. 이미지 없는 라벨 삭제
# 3. 클래스 0 변환 + confidence 제거
# 4. 빈 라벨 검사
 

import os

# =========================
# 경로 설정
# =========================
BASE_PATH = r"D:\smart_tunnel_V3\data\raw_frames"
LABEL_PATH = os.path.join(BASE_PATH, "labels")
IMAGE_PATH = os.path.join(BASE_PATH, "images")

# =========================
# 1. 라벨 검사
# =========================
print("\n🔍 [1] 라벨 검사 시작")

image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith(".jpg")]
label_files = [f for f in os.listdir(LABEL_PATH) if f.endswith(".txt")]

image_set = set([os.path.splitext(f)[0] for f in image_files])
label_set = set([os.path.splitext(f)[0] for f in label_files])

# 라벨 없는 이미지
missing_labels = image_set - label_set

if len(missing_labels) > 0:
    print(f"❌ 라벨 없는 이미지: {len(missing_labels)}개")
    print(list(missing_labels)[:10])

# =========================
# 2. 라벨 없는 이미지 → 빈 라벨 생성
# =========================
print("\n📝 라벨 없는 이미지 → 빈 라벨 생성")

created_count = 0

for img_name in missing_labels:
    label_path = os.path.join(LABEL_PATH, img_name + ".txt")

    if not os.path.exists(label_path):
        with open(label_path, "w") as f:
            pass

        created_count += 1

print(f"🆕 생성된 빈 라벨: {created_count}개")

# 🔥 label_files 다시 불러오기 (중요)
label_files = [f for f in os.listdir(LABEL_PATH) if f.endswith(".txt")]
label_set = set([os.path.splitext(f)[0] for f in label_files])

# =========================
# 3. 이미지 없는 라벨 삭제
# =========================
print("\n🧹 이미지 없는 라벨 삭제")

missing_images = label_set - image_set
removed_count = 0

for label_name in missing_images:
    label_path = os.path.join(LABEL_PATH, label_name + ".txt")

    if os.path.exists(label_path):
        os.remove(label_path)
        removed_count += 1

print(f"🗑 삭제된 라벨: {removed_count}개")

print(f"✔ 이미지 수: {len(image_files)}")
print(f"✔ 라벨 수: {len(label_files)}")

# =========================
# 4. 클래스 0 변환 + confidence 제거
# =========================
print("\n🔄 [2] 클래스 변환 + confidence 제거 시작")

for label_file in label_files:
    label_path = os.path.join(LABEL_PATH, label_file)

    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        # 🔥 클래스 0 통일
        parts[0] = "0"

        # 🔥 confidence 제거 (앞 5개만)
        parts = parts[:5]

        new_lines.append(" ".join(parts))

    with open(label_path, "w") as f:
        f.write("\n".join(new_lines))

print("✅ 클래스 0 변환 + confidence 제거 완료")

# =========================
# 5. 빈 라벨 검사
# =========================
print("\n📦 [3] 빈 라벨 검사")

empty_count = 0

for label_file in label_files:
    label_path = os.path.join(LABEL_PATH, label_file)

    if os.path.getsize(label_path) == 0:
        empty_count += 1

print(f"⚠ 빈 라벨 파일: {empty_count}개")

print("\n🎯 전체 라벨 정리 완료!")