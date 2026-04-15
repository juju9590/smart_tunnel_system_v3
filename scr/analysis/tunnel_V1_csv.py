# ==========================================
# 🚀 CSV 분석 자동화 (그래프 + 저장 + 요약)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================
# 📁 경로 설정 (여기만 수정)
# ==========================
NORMAL_PATH = "../../outputs/v1/tunnel_traffic_V5_20260409_193014_정상.csv"
CONGESTION_PATH = "../../outputs/v1/tunnel_traffic_V6_20260409_193827_혼잡.csv"
ACCIDENT_PATH = "../../outputs/v1/tunnel_traffic_V3_20260409_185806_사고.csv"

# ==========================
# 📁 출력 폴더 생성
# ==========================
OUTPUT_DIR = "../../outputs/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# 📊 데이터 로드
# ==========================
df_normal = pd.read_csv(NORMAL_PATH)
df_congestion = pd.read_csv(CONGESTION_PATH)
df_accident = pd.read_csv(ACCIDENT_PATH)

# ==========================
# 🎯 대표 차량 선택 함수
# ==========================
def select_vehicle(df):
    return df["tid"].value_counts().idxmax()

# 대표 차량 선택
tid_n = select_vehicle(df_normal)
tid_c = select_vehicle(df_congestion)
tid_a = select_vehicle(df_accident)

car_n = df_normal[df_normal["tid"] == tid_n]
car_c = df_congestion[df_congestion["tid"] == tid_c]
car_a = df_accident[df_accident["tid"] == tid_a]

# ==========================
# 📈 그래프 1: 속도 비교
# ==========================
plt.figure(figsize=(12,6))

plt.plot(car_n["frame_idx"], car_n["speed"], label="NORMAL")
plt.plot(car_c["frame_idx"], car_c["speed"], label="CONGESTION")
plt.plot(car_a["frame_idx"], car_a["speed"], label="ACCIDENT")

plt.title("Speed Comparison")
plt.xlabel("Frame")
plt.ylabel("Speed")
plt.legend()
plt.grid()

# 저장
plt.savefig(os.path.join(OUTPUT_DIR, "speed_comparison.png"))

plt.show()

# ==========================
# 📈 그래프 2: 거리 변화
# ==========================
plt.figure(figsize=(12,6))

plt.plot(car_n["frame_idx"], car_n["dist_diff"], label="NORMAL")
plt.plot(car_c["frame_idx"], car_c["dist_diff"], label="CONGESTION")
plt.plot(car_a["frame_idx"], car_a["dist_diff"], label="ACCIDENT")

plt.title("Distance Change Comparison")
plt.xlabel("Frame")
plt.ylabel("dist_diff")
plt.legend()
plt.grid()

# 저장
plt.savefig(os.path.join(OUTPUT_DIR, "dist_diff_comparison.png"))

plt.show()

# ==========================
# 📊 평균 비교 출력
# ==========================
print("===== 평균 비교 =====")

print("\n[ NORMAL ]")
print("speed:", df_normal["speed"].mean())
print("dist_diff:", df_normal["dist_diff"].mean())

print("\n[ CONGESTION ]")
print("speed:", df_congestion["speed"].mean())
print("dist_diff:", df_congestion["dist_diff"].mean())

print("\n[ ACCIDENT ]")
print("speed:", df_accident["speed"].mean())
print("dist_diff:", df_accident["dist_diff"].mean())

# ==========================
# 📁 요약 CSV 저장
# ==========================
summary = pd.DataFrame({
    "type": ["normal", "congestion", "accident"],
    "speed_mean": [
        df_normal["speed"].mean(),
        df_congestion["speed"].mean(),
        df_accident["speed"].mean()
    ],
    "dist_diff_mean": [
        df_normal["dist_diff"].mean(),
        df_congestion["dist_diff"].mean(),
        df_accident["dist_diff"].mean()
    ]
})

summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
summary.to_csv(summary_path, index=False)

print("\n📁 저장 완료:")
print("그래프 →", OUTPUT_DIR)
print("요약 CSV →", summary_path)