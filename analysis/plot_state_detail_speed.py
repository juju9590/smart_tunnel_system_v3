import os
import ast
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1) 경로 설정
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
PARENT_DIR = os.path.dirname(PROJECT_ROOT)

# CSV는 실제 저장된 곳에서 읽기
INPUT_DIR = os.path.join(PARENT_DIR, "smart_tunnel_V3_outputs", "analysis_v5_3")

# 그래프는 analysis 폴더에 저장
OUTPUT_DIR = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_outputs/analysis_v5_3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 실제 파일명
CSV_FILENAME = "state_detail_long_log_v5_3_2_20260417_174033.csv"
CSV_PATH = os.path.join(INPUT_DIR, CSV_FILENAME)

TARGET_TRACK_ID = 205

print("BASE_DIR:", BASE_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("PARENT_DIR:", PARENT_DIR)
print("INPUT_DIR:", INPUT_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("CSV_PATH:", CSV_PATH)
print("CSV exists:", os.path.exists(CSV_PATH))


# ==========================================
# 2) CSV 로드
# ==========================================
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

required_cols = [
    "frame_id",
    "track_id",
    "raw_speed",
    "corrected_speed",
    "ema_speed",
    "frame_avg_speed",
    "buffer_avg_speed",
    "final_speed",
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"필수 컬럼 없음: {col}")


# ==========================================
# 3) 특정 차량 ID 필터
# ==========================================
df_track = df[df["track_id"] == TARGET_TRACK_ID].copy()

if df_track.empty:
    raise ValueError(f"차량 ID {TARGET_TRACK_ID} 가 로그에 없습니다.")

df_track = df_track.sort_values("frame_id")


# ==========================================
# 4) 그래프 1: 차량 속도 변화
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(df_track["frame_id"], df_track["raw_speed"], label="raw_speed")
plt.plot(df_track["frame_id"], df_track["corrected_speed"], label="corrected_speed")
plt.plot(df_track["frame_id"], df_track["ema_speed"], label="ema_speed")
plt.xlabel("Frame ID")
plt.ylabel("Speed")
plt.title(f"Track ID {TARGET_TRACK_ID} Speed Change")
plt.legend()
plt.grid(True)

save_path_1 = os.path.join(
    OUTPUT_DIR,
    f"track_{TARGET_TRACK_ID}_speed_change.png"
)
plt.savefig(save_path_1, dpi=150, bbox_inches="tight")
plt.show()


# ==========================================
# 5) 그래프 2: 차량 속도 vs 평균 속도
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(df_track["frame_id"], df_track["ema_speed"], label=f"track_{TARGET_TRACK_ID}_ema_speed")
plt.plot(df_track["frame_id"], df_track["frame_avg_speed"], label="frame_avg_speed")
plt.plot(df_track["frame_id"], df_track["buffer_avg_speed"], label="buffer_avg_speed")
plt.plot(df_track["frame_id"], df_track["final_speed"], label="final_speed")
plt.xlabel("Frame ID")
plt.ylabel("Speed")
plt.title(f"Track ID {TARGET_TRACK_ID} vs Average Speeds")
plt.legend()
plt.grid(True)

save_path_2 = os.path.join(
    OUTPUT_DIR,
    f"track_{TARGET_TRACK_ID}_vs_avg_speed.png"
)
plt.savefig(save_path_2, dpi=150, bbox_inches="tight")
plt.show()

print("✅ 그래프 저장 완료")
print(" -", save_path_1)
print(" -", save_path_2)

# ==========================================
# 4) 그래프 1: 차량 속도(선) + 평균속도(막대)
# ==========================================
plt.figure(figsize=(12, 6))

# 평균속도는 막대로
plt.bar(
    df_track["frame_id"],
    df_track["frame_avg_speed"],
    width=1.0,
    alpha=0.35,
    label="frame_avg_speed"
)

# 차량 속도는 선으로
plt.plot(df_track["frame_id"], df_track["raw_speed"], label="raw_speed")
plt.plot(df_track["frame_id"], df_track["corrected_speed"], label="corrected_speed")
plt.plot(df_track["frame_id"], df_track["ema_speed"], label="ema_speed")

# 필요하면 final_speed도 같이
plt.plot(
    df_track["frame_id"],
    df_track["final_speed"],
    linestyle="--",
    label="final_speed"
)

plt.xlabel("Frame ID")
plt.ylabel("Speed")
plt.title(f"Track ID {TARGET_TRACK_ID} Speed Change + Frame Avg Speed")
plt.legend()
plt.grid(True)

save_path_1 = os.path.join(
    OUTPUT_DIR,
    f"track_{TARGET_TRACK_ID}_speed_with_frame_avg_bar.png"
)
plt.savefig(save_path_1, dpi=150, bbox_inches="tight")
plt.show()
