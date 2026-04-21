import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1) 경로 설정
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
PARENT_DIR = os.path.dirname(PROJECT_ROOT)

INPUT_DIR = os.path.join(PARENT_DIR, "smart_tunnel_V3_outputs", "analysis_v5_3")
OUTPUT_DIR = os.path.join(PARENT_DIR, "smart_tunnel_V3_outputs", "analysis_v5_3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 실제 파일명으로 바꿔줘
CSV_FILENAME = "state_detail_long_log_v5_3_2_20260417_175810.csv"
CSV_PATH = os.path.join(INPUT_DIR, CSV_FILENAME)

# 보고 싶은 차량 ID
TARGET_TRACK_ID = 205

print("BASE_DIR:", BASE_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("PARENT_DIR:", PARENT_DIR)
print("INPUT_DIR:", INPUT_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("CSV_PATH:", CSV_PATH)
print("CSV exists:", os.path.exists(CSV_PATH))

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV 파일 없음: {CSV_PATH}")

# ==========================================
# 2) CSV 로드
# ==========================================
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

required_cols = [
    "frame_id",
    "track_id",
    "dy",
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

# 숫자형 변환
numeric_cols = [
    "frame_id", "track_id", "dy", "raw_speed", "corrected_speed",
    "ema_speed", "frame_avg_speed", "buffer_avg_speed", "final_speed"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================================
# 3) 특정 차량 ID 필터
# ==========================================
df_track = df[df["track_id"] == TARGET_TRACK_ID].copy()

if df_track.empty:
    raise ValueError(f"차량 ID {TARGET_TRACK_ID} 가 로그에 없습니다.")

df_track = df_track.sort_values("frame_id").reset_index(drop=True)

# 선택 컬럼 확인
has_bbox_height = "bbox_height" in df_track.columns
has_bbox_area = "bbox_area" in df_track.columns

if has_bbox_height:
    df_track["bbox_height"] = pd.to_numeric(df_track["bbox_height"], errors="coerce")

if has_bbox_area:
    df_track["bbox_area"] = pd.to_numeric(df_track["bbox_area"], errors="coerce")

# ==========================================
# 4) 그래프 1: dy + raw/corrected/ema
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(df_track["frame_id"], df_track["dy"], label="dy")
plt.plot(df_track["frame_id"], df_track["raw_speed"], label="raw_speed")
plt.plot(df_track["frame_id"], df_track["corrected_speed"], label="corrected_speed")
plt.plot(df_track["frame_id"], df_track["ema_speed"], label="ema_speed")
plt.xlabel("Frame ID")
plt.ylabel("Value")
plt.title(f"Track ID {TARGET_TRACK_ID} : dy and speed change")
plt.legend()
plt.grid(True)

save_path_1 = os.path.join(
    OUTPUT_DIR,
    f"track_{TARGET_TRACK_ID}_dy_speed_change.png"
)
plt.savefig(save_path_1, dpi=150, bbox_inches="tight")
plt.show()

# ==========================================
# 5) 그래프 2: ema_speed vs 평균속도 + dy(막대)
# ==========================================
plt.figure(figsize=(12, 6))

# dy는 막대로
plt.bar(
    df_track["frame_id"],
    df_track["dy"],
    width=1.0,
    alpha=0.30,
    label="dy"
)

# 속도 선
plt.plot(df_track["frame_id"], df_track["ema_speed"], label="ema_speed")
plt.plot(df_track["frame_id"], df_track["frame_avg_speed"], label="frame_avg_speed")
plt.plot(df_track["frame_id"], df_track["buffer_avg_speed"], label="buffer_avg_speed")
plt.plot(df_track["frame_id"], df_track["final_speed"], label="final_speed")

plt.xlabel("Frame ID")
plt.ylabel("Value")
plt.title(f"Track ID {TARGET_TRACK_ID} : dy vs average speeds")
plt.legend()
plt.grid(True)

save_path_2 = os.path.join(
    OUTPUT_DIR,
    f"track_{TARGET_TRACK_ID}_dy_vs_avg_speed.png"
)
plt.savefig(save_path_2, dpi=150, bbox_inches="tight")
plt.show()

# ==========================================
# 6) 그래프 3: BBox 높이 또는 면적이 있으면 같이 표시
# ==========================================
if has_bbox_height or has_bbox_area:
    plt.figure(figsize=(12, 6))

    plt.plot(df_track["frame_id"], df_track["dy"], label="dy")
    plt.plot(df_track["frame_id"], df_track["ema_speed"], label="ema_speed")

    if has_bbox_height:
        plt.plot(df_track["frame_id"], df_track["bbox_height"], label="bbox_height")

    if has_bbox_area:
        plt.plot(df_track["frame_id"], df_track["bbox_area"], label="bbox_area")

    plt.xlabel("Frame ID")
    plt.ylabel("Value")
    plt.title(f"Track ID {TARGET_TRACK_ID} : BBox and dy relation")
    plt.legend()
    plt.grid(True)

    save_path_3 = os.path.join(
        OUTPUT_DIR,
        f"track_{TARGET_TRACK_ID}_bbox_dy_relation.png"
    )
    plt.savefig(save_path_3, dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("ℹ️ bbox_height / bbox_area 컬럼이 없어서 BBox 그래프는 건너뜀")
    save_path_3 = None

# ==========================================
# 7) 그래프 4: dy 절대값 vs raw_speed 확인
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(df_track["frame_id"], df_track["dy"].abs(), label="abs(dy)")
plt.plot(df_track["frame_id"], df_track["raw_speed"], label="raw_speed")
plt.xlabel("Frame ID")
plt.ylabel("Value")
plt.title(f"Track ID {TARGET_TRACK_ID} : abs(dy) vs raw_speed")
plt.legend()
plt.grid(True)

save_path_4 = os.path.join(
    OUTPUT_DIR,
    f"track_{TARGET_TRACK_ID}_absdy_vs_rawspeed.png"
)
plt.savefig(save_path_4, dpi=150, bbox_inches="tight")
plt.show()

print("✅ 그래프 저장 완료")
print(" -", save_path_1)
print(" -", save_path_2)
if save_path_3 is not None:
    print(" -", save_path_3)
print(" -", save_path_4)