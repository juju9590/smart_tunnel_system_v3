# ==========================================
# 파일명: plot_accident_debug_V5_5_2.py
# 설명:
# SMART TUNNEL V5_5_2 사고 디버그 CSV 시각화 코드
#
# 기능
# 1) accident_debug_v5_5_2 CSV 로드
# 2) 프레임 단위로 집계
# 3) 사고 후보 / 반복횟수 / 최종 lock 흐름 시각화
# 4) 전체 구간 + 관심 구간(예: 150~400) 그래프 저장
#
# 출력
# - accident_plot_full_v5_5_2_*.png
# - accident_plot_focus_v5_5_2_*.png
# - accident_summary_v5_5_2_*.csv
# - accident_frame_summary_v5_5_2_*.csv
# ==========================================

import os
import glob
import traceback
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


print("🚀 plot_accident_debug_V5_5_2 시작")

# =========================================================
# 1) 경로 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "smart_tunnel_V3_outputs")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "analysis_v5_5_2")

# ---------------------------------------------------------
# 분석할 CSV 직접 지정 가능
# ---------------------------------------------------------
CSV_PATH = None

# 예시:
# CSV_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_outputs/analysis_v5_5_2/accident_debug_v5_5_2_20260421_180000.csv"

# 관심 구간
FOCUS_START = 150
FOCUS_END = 400

# =========================================================
# 2) 유틸
# =========================================================
def find_latest_csv(output_dir):
    pattern = os.path.join(output_dir, "accident_debug_v5_5_2_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def to_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "nan": False,
            "": False
        })
        .fillna(False)
    )

def safe_numeric(series, default=0):
    return pd.to_numeric(series, errors="coerce").fillna(default)

def aggregate_frame_level(df):
    bool_cols = [
        "frame_accident_prediction",
        "frame_accident",
        "accident_locked",
        "same_lane",
        "dist_drop",
        "gap_up",
        "vertical",
        "vertical_or_lane",
        "lane_break",
        "lane_break_acc",
        "impact_persist",
        "persist_evidence",
        "stop_evidence",
        "lane_break_evidence",
        "smoke_fire",
        "abnormal",
        "abnormal_stop",
        "abnormal_pose",
        "rear_core",
        "gap_weak",
        "gap_strong",
        "post_evidence",
        "weak_pair_candidate",
        "strong_pair_candidate",
        "pair_accident_candidate",
        "pair_repeat_candidate",
        "pair_consecutive_candidate",
        "repeat_strong_candidate",
        "early_repeat_candidate",
        "early_repeat_valid",
        "pair_high_score",
    ]

    for col in bool_cols:
        if col in df.columns:
            df[col] = to_bool(df[col])
        else:
            df[col] = False

    num_cols = [
        "frame_id",
        "vehicle_count",
        "lane_count",
        "recent_prediction_count",
        "accident_start_frame",
        "dist",
        "gap",
        "repeat_count_window",
        "early_repeat_count_window",
        "pair_score",
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col], default=0)
        else:
            df[col] = 0

    frame_df = df.groupby("frame_id").agg(
        vehicle_count=("vehicle_count", "max"),
        lane_count=("lane_count", "max"),

        frame_accident_prediction=("frame_accident_prediction", "max"),
        frame_accident=("frame_accident", "max"),
        accident_locked=("accident_locked", "max"),
        recent_prediction_count=("recent_prediction_count", "max"),
        accident_start_frame=("accident_start_frame", "max"),

        pair_rows=("frame_id", "size"),

        strong_pair_count=("strong_pair_candidate", "sum"),
        pair_accident_count=("pair_accident_candidate", "sum"),
        repeat_pair_count=("pair_repeat_candidate", "sum"),
        repeat_strong_count=("repeat_strong_candidate", "sum"),
        early_repeat_count=("early_repeat_candidate", "sum"),
        early_repeat_valid_count=("early_repeat_valid", "sum"),
        pair_high_score_count=("pair_high_score", "sum"),

        max_repeat_count_window=("repeat_count_window", "max"),
        max_early_repeat_count_window=("early_repeat_count_window", "max"),
        max_pair_score=("pair_score", "max"),

        max_gap=("gap", "max"),
        min_dist=("dist", "min"),
    ).reset_index()

    return frame_df

def summarize_events(frame_df):
    pred_frames = frame_df.loc[frame_df["frame_accident_prediction"] == True, "frame_id"].tolist()
    locked_frames = frame_df.loc[frame_df["accident_locked"] == True, "frame_id"].tolist()
    strong_frames = frame_df.loc[frame_df["strong_pair_count"] > 0, "frame_id"].tolist()
    repeat_frames = frame_df.loc[frame_df["repeat_pair_count"] > 0, "frame_id"].tolist()
    early_valid_frames = frame_df.loc[frame_df["early_repeat_valid_count"] > 0, "frame_id"].tolist()
    high_score_frames = frame_df.loc[frame_df["pair_high_score_count"] > 0, "frame_id"].tolist()

    summary = {
        "total_frames": int(frame_df["frame_id"].max() + 1) if len(frame_df) > 0 else 0,
        "prediction_frame_count": len(pred_frames),
        "prediction_frames": ",".join(map(str, pred_frames)),
        "locked_frame_count": len(locked_frames),
        "locked_start_frame": locked_frames[0] if len(locked_frames) > 0 else None,
        "strong_pair_frame_count": len(strong_frames),
        "strong_pair_frames": ",".join(map(str, strong_frames)),
        "repeat_pair_frame_count": len(repeat_frames),
        "repeat_pair_frames": ",".join(map(str, repeat_frames)),
        "early_repeat_valid_frame_count": len(early_valid_frames),
        "early_repeat_valid_frames": ",".join(map(str, early_valid_frames)),
        "pair_high_score_frame_count": len(high_score_frames),
        "pair_high_score_frames": ",".join(map(str, high_score_frames)),
        "max_recent_prediction_count": int(frame_df["recent_prediction_count"].max()) if len(frame_df) > 0 else 0,
        "max_repeat_count_window": int(frame_df["max_repeat_count_window"].max()) if len(frame_df) > 0 else 0,
        "max_early_repeat_count_window": int(frame_df["max_early_repeat_count_window"].max()) if len(frame_df) > 0 else 0,
        "max_pair_score": int(frame_df["max_pair_score"].max()) if len(frame_df) > 0 else 0,
        "focus_start": FOCUS_START,
        "focus_end": FOCUS_END,
    }

    return pd.DataFrame([summary])

def save_full_plot(frame_df, out_path, title_suffix=""):
    plt.figure(figsize=(16, 12))

    plt.subplot(5, 1, 1)
    plt.plot(frame_df["frame_id"], frame_df["vehicle_count"])
    plt.ylabel("Vehicles")
    plt.title(f"Accident Debug Full Timeline {title_suffix}")

    plt.subplot(5, 1, 2)
    plt.plot(frame_df["frame_id"], frame_df["strong_pair_count"], label="strong_pair_count")
    plt.plot(frame_df["frame_id"], frame_df["repeat_pair_count"], label="repeat_pair_count")
    plt.plot(frame_df["frame_id"], frame_df["early_repeat_valid_count"], label="early_repeat_valid_count")
    plt.plot(frame_df["frame_id"], frame_df["pair_high_score_count"], label="pair_high_score_count")
    plt.ylabel("Pair Counts")
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(frame_df["frame_id"], frame_df["recent_prediction_count"], label="recent_prediction_count")
    plt.plot(frame_df["frame_id"], frame_df["max_repeat_count_window"], label="max_repeat_count_window")
    plt.plot(frame_df["frame_id"], frame_df["max_early_repeat_count_window"], label="max_early_repeat_count_window")
    plt.plot(frame_df["frame_id"], frame_df["max_pair_score"], label="max_pair_score")
    plt.ylabel("Counts / Score")
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(frame_df["frame_id"], frame_df["max_gap"], label="max_gap")
    plt.plot(frame_df["frame_id"], frame_df["min_dist"], label="min_dist")
    plt.ylabel("Gap / Dist")
    plt.legend()

    plt.subplot(5, 1, 5)
    plt.plot(frame_df["frame_id"], frame_df["frame_accident_prediction"].astype(int), label="frame_accident_prediction")
    plt.plot(frame_df["frame_id"], frame_df["frame_accident"].astype(int), label="frame_accident")
    plt.plot(frame_df["frame_id"], frame_df["accident_locked"].astype(int), label="accident_locked")
    plt.ylabel("Accident Flags")
    plt.xlabel("Frame ID")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def save_focus_plot(frame_df, out_path, start_frame, end_frame, title_suffix=""):
    focus_df = frame_df[(frame_df["frame_id"] >= start_frame) & (frame_df["frame_id"] <= end_frame)].copy()

    plt.figure(figsize=(16, 12))

    plt.subplot(5, 1, 1)
    plt.plot(focus_df["frame_id"], focus_df["vehicle_count"])
    plt.ylabel("Vehicles")
    plt.title(f"Accident Debug Focus {start_frame}~{end_frame} {title_suffix}")

    plt.subplot(5, 1, 2)
    plt.plot(focus_df["frame_id"], focus_df["strong_pair_count"], label="strong_pair_count")
    plt.plot(focus_df["frame_id"], focus_df["repeat_pair_count"], label="repeat_pair_count")
    plt.plot(focus_df["frame_id"], focus_df["early_repeat_valid_count"], label="early_repeat_valid_count")
    plt.plot(focus_df["frame_id"], focus_df["pair_high_score_count"], label="pair_high_score_count")
    plt.ylabel("Pair Counts")
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(focus_df["frame_id"], focus_df["recent_prediction_count"], label="recent_prediction_count")
    plt.plot(focus_df["frame_id"], focus_df["max_repeat_count_window"], label="max_repeat_count_window")
    plt.plot(focus_df["frame_id"], focus_df["max_early_repeat_count_window"], label="max_early_repeat_count_window")
    plt.plot(focus_df["frame_id"], focus_df["max_pair_score"], label="max_pair_score")
    plt.ylabel("Counts / Score")
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(focus_df["frame_id"], focus_df["max_gap"], label="max_gap")
    plt.plot(focus_df["frame_id"], focus_df["min_dist"], label="min_dist")
    plt.ylabel("Gap / Dist")
    plt.legend()

    plt.subplot(5, 1, 5)
    plt.plot(focus_df["frame_id"], focus_df["frame_accident_prediction"].astype(int), label="frame_accident_prediction")
    plt.plot(focus_df["frame_id"], focus_df["frame_accident"].astype(int), label="frame_accident")
    plt.plot(focus_df["frame_id"], focus_df["accident_locked"].astype(int), label="accident_locked")
    plt.ylabel("Accident Flags")
    plt.xlabel("Frame ID")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def print_focus_summary(frame_df, start_frame, end_frame):
    focus_df = frame_df[(frame_df["frame_id"] >= start_frame) & (frame_df["frame_id"] <= end_frame)].copy()

    pred_frames = focus_df.loc[focus_df["frame_accident_prediction"] == True, "frame_id"].tolist()
    strong_frames = focus_df.loc[focus_df["strong_pair_count"] > 0, "frame_id"].tolist()
    repeat_frames = focus_df.loc[focus_df["repeat_pair_count"] > 0, "frame_id"].tolist()
    early_valid_frames = focus_df.loc[focus_df["early_repeat_valid_count"] > 0, "frame_id"].tolist()
    high_score_frames = focus_df.loc[focus_df["pair_high_score_count"] > 0, "frame_id"].tolist()

    print("\n================ FOCUS SUMMARY ================")
    print(f"구간: {start_frame} ~ {end_frame}")
    print(f"prediction frames          : {pred_frames}")
    print(f"strong pair frames         : {strong_frames}")
    print(f"repeat pair frames         : {repeat_frames}")
    print(f"early repeat valid frames  : {early_valid_frames}")
    print(f"high score frames          : {high_score_frames}")
    print(f"max recent_prediction_count: {int(focus_df['recent_prediction_count'].max()) if len(focus_df) > 0 else 0}")
    print(f"max repeat_count_window    : {int(focus_df['max_repeat_count_window'].max()) if len(focus_df) > 0 else 0}")
    print(f"max early_repeat_window    : {int(focus_df['max_early_repeat_count_window'].max()) if len(focus_df) > 0 else 0}")
    print(f"max pair_score             : {int(focus_df['max_pair_score'].max()) if len(focus_df) > 0 else 0}")
    print("==============================================\n")

# =========================================================
# 3) 메인
# =========================================================
def main():
    global CSV_PATH

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if CSV_PATH is None:
        CSV_PATH = find_latest_csv(OUTPUT_DIR)

    if CSV_PATH is None or not os.path.exists(CSV_PATH):
        print("❌ 분석할 CSV를 찾지 못함")
        return

    print("📂 CSV 로드:", CSV_PATH)
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    if "frame_id" not in df.columns:
        print("❌ frame_id 컬럼이 없음")
        print("컬럼명:", list(df.columns))
        return

    frame_df = aggregate_frame_level(df)

    base_name = os.path.splitext(os.path.basename(CSV_PATH))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    full_png = os.path.join(OUTPUT_DIR, f"accident_plot_full_v5_5_2_{timestamp}.png")
    focus_png = os.path.join(OUTPUT_DIR, f"accident_plot_focus_v5_5_2_{timestamp}.png")
    summary_csv = os.path.join(OUTPUT_DIR, f"accident_summary_v5_5_2_{timestamp}.csv")
    frame_summary_csv = os.path.join(OUTPUT_DIR, f"accident_frame_summary_v5_5_2_{timestamp}.csv")

    save_full_plot(frame_df, full_png, title_suffix=f"({base_name})")
    save_focus_plot(frame_df, focus_png, FOCUS_START, FOCUS_END, title_suffix=f"({base_name})")

    summary_df = summarize_events(frame_df)
    summary_df["source_csv"] = CSV_PATH
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    frame_df.to_csv(frame_summary_csv, index=False, encoding="utf-8-sig")

    print_focus_summary(frame_df, FOCUS_START, FOCUS_END)

    print("✅ 저장 완료")
    print("전체 그래프:", full_png)
    print("집중 그래프:", focus_png)
    print("요약 CSV:", summary_csv)
    print("프레임 집계 CSV:", frame_summary_csv)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ 실행 중 오류")
        print(e)
        traceback.print_exc()