import os
import cv2
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random
import time  # 🔥 FPS 측정용

# =========================
# 1. 경로 설정
# =========================
BASE_PATH = r"D:\smart_tunnel_V3\data"

INPUT_PATH = os.path.join(BASE_PATH, "preprocess_frames")
OUTPUT_PATH = os.path.join(BASE_PATH, "preprocess_output")

# 🔥 기존 결과 삭제 (추천)
shutil.rmtree(OUTPUT_PATH, ignore_errors=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================
# 2. 전처리 함수
# =========================

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_gamma(img, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)


def apply_clahe_blur(img):
    img = apply_clahe(img)
    return cv2.GaussianBlur(img, (5,5), 0)


# =========================
# 3. 전처리 적용
# =========================
PREPROCESS = {
    "original": lambda x: x,
    "clahe": apply_clahe,
    "gamma": apply_gamma,
    "clahe_blur": apply_clahe_blur
}

print("🚀 전처리 시작")

for p_name, func in PREPROCESS.items():
    save_dir = os.path.join(OUTPUT_PATH, p_name)
    os.makedirs(save_dir, exist_ok=True)

    for root, _, files in os.walk(INPUT_PATH):
        for file in files:
            if not file.endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            processed = func(img)

            save_path = os.path.join(save_dir, file)
            cv2.imwrite(save_path, processed)

print("✅ 전처리 완료")

# =========================
# 4. 모델 평가
# =========================
MODELS = {
    "yolov8s": YOLO("yolov8s.pt"),
    "yolo11n": YOLO("yolo11n.pt")
}

results = []

print("🚀 모델 평가 시작")

for model_name, model in MODELS.items():
    for p_name in PREPROCESS.keys():

        img_dir = os.path.join(OUTPUT_PATH, p_name)
        img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

        total_det = 0
        total_conf = 0
        total_fps = 0  # 🔥 추가
        count = 0

        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)

            # 🔥 FPS 측정 시작
            start = time.time()

            result = model.predict(img_path, conf=0.25, save=False)

            end = time.time()
            fps = 1 / (end - start)  # 🔥 FPS 계산
            total_fps += fps

            boxes = result[0].boxes

            if boxes is not None:
                total_det += len(boxes)

                if len(boxes) > 0:
                    total_conf += boxes.conf.mean().item()

            count += 1

        avg_det = total_det / count
        avg_conf = total_conf / count
        avg_fps = total_fps / count  # 🔥 추가

        detection_rate = total_det / count if total_det > 0 else 0

        print(f"{model_name} | {p_name} → det:{avg_det:.2f}, conf:{avg_conf:.2f}, fps:{avg_fps:.2f}")

        results.append({
            "model": model_name,
            "preprocess": p_name,
            "avg_detection": avg_det,
            "avg_confidence": avg_conf,
            "detection_rate": detection_rate,
            "avg_fps": avg_fps  # 🔥 추가
        })

# =========================
# 5. 결과 저장
# =========================
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_PATH, "results.csv")
df.to_csv(csv_path, index=False)

print(f"📁 결과 저장: {csv_path}")

# =========================
# 6. 그래프 (막대)
# =========================
def draw_bar(y_col, title, ylabel, filename):
    plt.figure()
    x = np.arange(len(df["preprocess"].unique()))
    width = 0.35

    for i, model_name in enumerate(df["model"].unique()):
        subset = df[df["model"] == model_name]
        plt.bar(x + i*width, subset[y_col], width, label=model_name)

    plt.xticks(x + width/2, subset["preprocess"])
    plt.title(title)
    plt.xlabel("Preprocess")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(OUTPUT_PATH, filename))
    plt.show()


draw_bar("avg_detection", "Detection Count", "Detection", "detection_graph.png")
draw_bar("avg_confidence", "Confidence", "Confidence", "confidence_graph.png")
draw_bar("detection_rate", "Detection Rate", "Rate", "detection_rate_graph.png")
draw_bar("avg_fps", "FPS", "FPS", "fps_graph.png")  # 🔥 추가

print("📊 그래프 완료")

# =========================
# 7. 비교 이미지 생성
# =========================
COMPARE_PATH = os.path.join(OUTPUT_PATH, "compare_samples")
os.makedirs(COMPARE_PATH, exist_ok=True)

print("🖼️ 비교 이미지 생성 시작")

original_dir = os.path.join(OUTPUT_PATH, "original")
clahe_dir = os.path.join(OUTPUT_PATH, "clahe")
gamma_dir = os.path.join(OUTPUT_PATH, "gamma")
clahe_blur_dir = os.path.join(OUTPUT_PATH, "clahe_blur")

img_files = [f for f in os.listdir(original_dir) if f.endswith(".jpg")]

sample_files = random.sample(img_files, min(50, len(img_files)))

for idx, file in enumerate(sample_files):

    imgs = [
        cv2.imread(os.path.join(original_dir, file)),
        cv2.imread(os.path.join(clahe_dir, file)),
        cv2.imread(os.path.join(gamma_dir, file)),
        cv2.imread(os.path.join(clahe_blur_dir, file))
    ]

    if any(img is None for img in imgs):
        continue

    h, w = imgs[0].shape[:2]
    imgs = [cv2.resize(img, (w, h)) for img in imgs]

    labels = ["Original", "CLAHE", "Gamma", "CLAHE+Blur"]

    for i in range(4):
        cv2.putText(imgs[i], labels[i], (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    combined = np.hstack(imgs)

    save_path = os.path.join(COMPARE_PATH, f"compare_{idx:03d}.jpg")
    cv2.imwrite(save_path, combined)

print("✅ 비교 이미지 생성 완료")