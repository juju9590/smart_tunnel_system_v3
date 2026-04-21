import os
import pandas as pd

# ==========================================
# 1) 경로 설정
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
PARENT_DIR = os.path.dirname(PROJECT_ROOT)

INPUT_DIR = os.path.join(PARENT_DIR, "smart_tunnel_V3_outputs", "analysis_v5_4")
OUTPUT_DIR = os.path.join(PARENT_DIR, "smart_tunnel_V3_outputs", "analysis_v5_4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2) 분석할 파일 / 상황 매핑
#    파일명은 실제 값으로 바꿔줘
# ==========================================
FILES = [
    {
        "label": "NORMAL",
        "filename": "state_detail_long_log_v5_4_20260417_191229.csv",
    },
    {
        "label": "CONGESTION",
        "filename": "state_detail_long_log_v5_4_20260417_191458.csv",
    },
    {
        "label": "JAM",
        "filename": "state_detail_long_log_v5_4_20260417_191118.csv",
    },
]

# 너무 짧게 잡힌 track 제외
MIN_FRAMES_PER_TRACK = 3

# 이상치 제거 여부
USE_OUTLIER_FILTER = False
LOW_Q = 0.05
HIGH_Q = 0.95

# ==========================================
# 3) 파일별 로드 + 기본 전처리
# ==========================================
all_rows = []
summary_rows = []
track_summary_rows = []

for item in FILES:
    label = item["label"]
    csv_path = os.path.join(INPUT_DIR, item["filename"])

    print(f"\n📂 로드: {csv_path}")
    print("   exists:", os.path.exists(csv_path))

    if not os.path.exists(csv_path):
        print("   -> 파일 없음, 건너뜀")
        continue

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = ["frame_id", "track_id", "norm_speed"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{item['filename']} 에 필수 컬럼 없음: {col}")

    # 숫자형 변환
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")
    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")
    df["norm_speed"] = pd.to_numeric(df["norm_speed"], errors="coerce")

    # 빈 track_id, 빈 norm_speed 제거
    df = df.dropna(subset=["frame_id", "track_id", "norm_speed"]).copy()
    df["track_id"] = df["track_id"].astype(int)
    df["label"] = label
    df["source_file"] = item["filename"]

    # track 길이 계산
    track_counts = df.groupby("track_id").size().reset_index(name="track_len")
    df = df.merge(track_counts, on="track_id", how="left")

    print("   원본 row 수:", len(df))

    track_counts = df.groupby("track_id").size().reset_index(name="track_len")
    print("   track 수:", len(track_counts))
    print("   track 길이 상위 10개:")
    print(track_counts.sort_values("track_len", ascending=False).head(10))

    # 짧은 track 제거
    df = df[df["track_len"] >= MIN_FRAMES_PER_TRACK].copy()

    print("   MIN_FRAMES 적용 후 row 수:", len(df))

    # 이상치 제거
    if USE_OUTLIER_FILTER and not df.empty:
        q_low = df["norm_speed"].quantile(LOW_Q)
        q_high = df["norm_speed"].quantile(HIGH_Q)
        df = df[(df["norm_speed"] >= q_low) & (df["norm_speed"] <= q_high)].copy()

    if df.empty:
        print("   -> 전처리 후 데이터 없음")
        continue

    print("   이상치 제거 후 row 수:", len(df))

    all_rows.append(df)



    # --------------------------------------
    # 파일별 전체 요약
    # --------------------------------------
    summary_rows.append({
        "label": label,
        "source_file": item["filename"],
        "n_rows": len(df),
        "n_tracks": df["track_id"].nunique(),
        "mean_norm_speed": round(df["norm_speed"].mean(), 4),
        "std_norm_speed": round(df["norm_speed"].std(), 4),
        "median_norm_speed": round(df["norm_speed"].median(), 4),
        "p10": round(df["norm_speed"].quantile(0.10), 4),
        "p25": round(df["norm_speed"].quantile(0.25), 4),
        "p50": round(df["norm_speed"].quantile(0.50), 4),
        "p75": round(df["norm_speed"].quantile(0.75), 4),
        "p90": round(df["norm_speed"].quantile(0.90), 4),
        "min_norm_speed": round(df["norm_speed"].min(), 4),
        "max_norm_speed": round(df["norm_speed"].max(), 4),
    })

    # --------------------------------------
    # track별 평균 요약
    # --------------------------------------
    track_df = (
        df.groupby("track_id")["norm_speed"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
        .rename(columns={
            "count": "track_len",
            "mean": "track_mean_norm_speed",
            "std": "track_std_norm_speed",
            "median": "track_median_norm_speed",
        })
    )
    track_df["label"] = label
    track_df["source_file"] = item["filename"]
    track_summary_rows.append(track_df)

# ==========================================
# 4) 결과 저장
# ==========================================
if not all_rows:
    raise ValueError("분석 가능한 CSV가 없습니다.")

all_df = pd.concat(all_rows, ignore_index=True)
summary_df = pd.DataFrame(summary_rows)
track_summary_df = pd.concat(track_summary_rows, ignore_index=True)

# 상황별 "track 평균의 평균"도 계산
track_level_summary_df = (
    track_summary_df.groupby("label")
    .agg(
        n_tracks=("track_id", "count"),
        mean_of_track_means=("track_mean_norm_speed", "mean"),
        std_of_track_means=("track_mean_norm_speed", "std"),
        median_of_track_means=("track_mean_norm_speed", "median"),
        p25_track_mean=("track_mean_norm_speed", lambda x: x.quantile(0.25)),
        p75_track_mean=("track_mean_norm_speed", lambda x: x.quantile(0.75)),
    )
    .reset_index()
)

# 반올림
for col in [
    "mean_of_track_means",
    "std_of_track_means",
    "median_of_track_means",
    "p25_track_mean",
    "p75_track_mean",
]:
    track_level_summary_df[col] = track_level_summary_df[col].round(4)

summary_path = os.path.join(OUTPUT_DIR, "norm_speed_summary_by_file.csv")
track_summary_path = os.path.join(OUTPUT_DIR, "norm_speed_summary_by_track.csv")
track_level_summary_path = os.path.join(OUTPUT_DIR, "norm_speed_track_level_summary.csv")

summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
track_summary_df.to_csv(track_summary_path, index=False, encoding="utf-8-sig")
track_level_summary_df.to_csv(track_level_summary_path, index=False, encoding="utf-8-sig")

print("\n✅ 저장 완료")
print(" -", summary_path)
print(" -", track_summary_path)
print(" -", track_level_summary_path)

print("\n[파일별 norm_speed 요약]")
print(summary_df)

print("\n[상황별 track 평균 요약]")
print(track_level_summary_df)