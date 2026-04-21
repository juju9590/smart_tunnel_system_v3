# ==========================================
# 파일명: plot_accident_debug_V5_5.py
# 위치: analysis/
# 설명:
# - accident_debug_v5_5 CSV를 읽어서
# - 프레임 단위 그래프 / 파일 비교 그래프를 저장
# - V5.5 컬럼 구조 대응 버전
# ==========================================

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 0. 경로 설정
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 외부 출력 루트
OUTPUT_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "smart_tunnel_V3_outputs")

# CSV가 저장된 폴더
CSV_DIR = os.path.join(OUTPUT_ROOT, "analysis_v5_5")

# 그래프 저장 폴더
GRAPH_DIR = os.path.join(OUTPUT_ROOT, "analysis_v5_5_graph")
os.makedirs(GRAPH_DIR, exist_ok=True)


# =========================================================
# 1. bool 변환 유틸
# =========================================================
def to_bool_series(series):
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "nan": False,
            "none": False,
            "": False
        })
        .fillna(False)
    )


def safe_col(df, col):
    if col in df.columns:
        return df[col]
    return pd.Series([0] * len(df), index=df.index)


# =========================================================
# 2. CSV 읽기 + 전처리
# =========================================================
def load_debug_csv(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    bool_cols = [
        # frame-level
        "frame_accident",
        "frame_accident_prediction",
        "accident_locked",

        # lane / geometry
        "same_lane",
        "dist_drop",
        "gap_up",
        "vertical",
        "vertical_or_lane",
        "lane_break",
        "lane_break_acc",

        # evidence
        "impact_persist",
        "persist_evidence",
        "stop_evidence",
        "lane_break_evidence",
        "smoke_fire",
        "abnormal",
        "abnormal_stop",
        "abnormal_pose",

        # pair logic
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
        "pair_high_score",

        # lane template
        "template_confirmed",
    ]

    num_cols = [
        "frame_id",
        "frame_acc_ratio",
        "recent_prediction_count",
        "accident_start_frame",

        "pair_id1",
        "pair_id2",
        "lane1",
        "lane2",

        "dist",
        "gap",
        "repeat_count_window",
        "pair_score",

        "lane_count",
    ]

    for col in bool_cols:
        if col in df.columns:
            df[col] = to_bool_series(df[col])

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["source_file"] = os.path.basename(csv_path)
    return df


# =========================================================
# 3. 프레임 단위 집계
# =========================================================
def aggregate_by_frame(df):
    agg_dict = {
        # frame-level
        "frame_accident": "max",
        "frame_acc_ratio": "max",
        "frame_accident_prediction": "max",
        "recent_prediction_count": "max",
        "accident_locked": "max",
        "accident_start_frame": "max",

        # pair count
        "pair_id1": "count",

        # pair candidate / repeat
        "pair_accident_candidate": "sum",
        "strong_pair_candidate": "sum",
        "weak_pair_candidate": "sum",
        "pair_repeat_candidate": "sum",
        "pair_consecutive_candidate": "sum",
        "repeat_strong_candidate": "sum",
        "pair_high_score": "sum",

        # evidence / core
        "rear_core": "sum",
        "post_evidence": "sum",
        "gap_weak": "sum",
        "gap_strong": "sum",

        # scores / repeat
        "pair_score": ["max", "mean"],
        "repeat_count_window": "max",

        # lane
        "lane_count": "max",
        "template_confirmed": "max",
    }

    usable = {k: v for k, v in agg_dict.items() if k in df.columns}
    frame_df = df.groupby("frame_id").agg(usable)

    # MultiIndex 컬럼 평탄화
    flat_cols = []
    for c in frame_df.columns:
        if isinstance(c, tuple):
            flat_cols.append("_".join([str(x) for x in c if x]))
        else:
            flat_cols.append(c)
    frame_df.columns = flat_cols

    rename_map = {
        "pair_id1_count": "pair_count",

        "pair_accident_candidate_sum": "pair_accident_candidate_count",
        "strong_pair_candidate_sum": "strong_pair_candidate_count",
        "weak_pair_candidate_sum": "weak_pair_candidate_count",
        "pair_repeat_candidate_sum": "pair_repeat_candidate_count",
        "pair_consecutive_candidate_sum": "pair_consecutive_candidate_count",
        "repeat_strong_candidate_sum": "repeat_strong_candidate_count",
        "pair_high_score_sum": "pair_high_score_count",

        "rear_core_sum": "rear_core_count",
        "post_evidence_sum": "post_evidence_count",
        "gap_weak_sum": "gap_weak_count",
        "gap_strong_sum": "gap_strong_count",

        "pair_score_max": "pair_score_max",
        "pair_score_mean": "pair_score_mean",
        "repeat_count_window_max": "repeat_count_window_max",

        "frame_accident_max": "frame_accident",
        "frame_acc_ratio_max": "frame_acc_ratio",
        "frame_accident_prediction_max": "frame_accident_prediction",
        "recent_prediction_count_max": "recent_prediction_count",
        "accident_locked_max": "accident_locked",
        "accident_start_frame_max": "accident_start_frame",

        "lane_count_max": "lane_count",
        "template_confirmed_max": "template_confirmed",
    }

    frame_df = frame_df.rename(columns=rename_map).reset_index()
    return frame_df


# =========================================================
# 4. 파일별 요약 통계
# =========================================================
def summarize_file(frame_df, raw_df, source_name):
    summary = {
        "source_file": source_name,
        "total_frames": int(frame_df["frame_id"].nunique()),
        "total_rows": int(len(raw_df)),
        "total_pairs": int(safe_col(frame_df, "pair_count").sum()),

        # frame-level
        "frames_with_frame_accident_prediction": int(safe_col(frame_df, "frame_accident_prediction").sum()),
        "frames_with_accident_locked": int(safe_col(frame_df, "accident_locked").sum()),
        "max_recent_prediction_count": float(safe_col(frame_df, "recent_prediction_count").max()),
        "first_accident_start_frame": float(safe_col(frame_df, "accident_start_frame").replace(0, pd.NA).dropna().min()) if len(safe_col(frame_df, "accident_start_frame").replace(0, pd.NA).dropna()) > 0 else 0,

        # pair-level
        "total_pair_accident_candidate": int(safe_col(frame_df, "pair_accident_candidate_count").sum()),
        "total_strong_pair_candidate": int(safe_col(frame_df, "strong_pair_candidate_count").sum()),
        "total_repeat_strong": int(safe_col(frame_df, "repeat_strong_candidate_count").sum()),
        "total_pair_high_score": int(safe_col(frame_df, "pair_high_score_count").sum()),

        "total_rear_core": int(safe_col(frame_df, "rear_core_count").sum()),
        "total_post_evidence": int(safe_col(frame_df, "post_evidence_count").sum()),
        "total_gap_weak": int(safe_col(frame_df, "gap_weak_count").sum()),
        "total_gap_strong": int(safe_col(frame_df, "gap_strong_count").sum()),

        "max_pair_score": float(safe_col(frame_df, "pair_score_max").max()),
        "mean_pair_score": float(safe_col(frame_df, "pair_score_mean").mean()),
        "max_repeat_count_window": float(safe_col(frame_df, "repeat_count_window_max").max()),

        "max_lane_count": float(safe_col(frame_df, "lane_count").max()),
        "template_confirmed_frames": int(safe_col(frame_df, "template_confirmed").sum()),
    }
    return summary


# =========================================================
# 5. 개별 파일 그래프 저장
# =========================================================
def plot_per_file(frame_df, source_name, save_dir):
    base_name = os.path.splitext(source_name)[0]

    # -----------------------------
    # 1) frame accident prediction / lock
    # -----------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "frame_accident_prediction"), label="frame_accident_prediction")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "recent_prediction_count"), label="recent_prediction_count")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "accident_locked"), label="accident_locked")
    plt.xlabel("frame_id")
    plt.ylabel("value")
    plt.title(f"[{base_name}] frame accident prediction / lock")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_frame_accident_prediction_lock.png"), dpi=150)
    plt.close()

    # -----------------------------
    # 2) pair candidate / high score
    # -----------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "pair_count"), label="pair_count")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "pair_accident_candidate_count"), label="pair_accident_candidate_count")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "pair_high_score_count"), label="pair_high_score_count")
    plt.xlabel("frame_id")
    plt.ylabel("count")
    plt.title(f"[{base_name}] pair candidate / high score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_pair_candidate_high_score.png"), dpi=150)
    plt.close()

    # -----------------------------
    # 3) condition counts
    # -----------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "rear_core_count"), label="rear_core_count")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "post_evidence_count"), label="post_evidence_count")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "strong_pair_candidate_count"), label="strong_pair_candidate_count")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "repeat_strong_candidate_count"), label="repeat_strong_candidate_count")
    plt.xlabel("frame_id")
    plt.ylabel("count")
    plt.title(f"[{base_name}] condition counts per frame")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_frame_condition_counts.png"), dpi=150)
    plt.close()

    # -----------------------------
    # 4) gap / score trend
    # -----------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "gap_weak_count"), label="gap_weak_count")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "gap_strong_count"), label="gap_strong_count")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "pair_score_max"), label="pair_score_max")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "pair_score_mean"), label="pair_score_mean")
    plt.xlabel("frame_id")
    plt.ylabel("value")
    plt.title(f"[{base_name}] gap / pair score trend")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_gap_pair_score_trend.png"), dpi=150)
    plt.close()

    # -----------------------------
    # 5) repeat candidate status
    # -----------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "pair_repeat_candidate_count"), label="pair_repeat_candidate_count")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "pair_consecutive_candidate_count"), label="pair_consecutive_candidate_count")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "repeat_count_window_max"), label="repeat_count_window_max")
    plt.xlabel("frame_id")
    plt.ylabel("value")
    plt.title(f"[{base_name}] repeat candidate status")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_repeat_candidate_status.png"), dpi=150)
    plt.close()

    # -----------------------------
    # 6) lane template status
    # -----------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "lane_count"), label="lane_count")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "template_confirmed"), label="template_confirmed")
    plt.xlabel("frame_id")
    plt.ylabel("value")
    plt.title(f"[{base_name}] lane template status")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_lane_template_status.png"), dpi=150)
    plt.close()


# =========================================================
# 6. 파일 간 비교 그래프
# =========================================================
def plot_comparison(summary_df, save_dir):
    if summary_df.empty:
        return

    # frame prediction / lock
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df["source_file"], summary_df["frames_with_frame_accident_prediction"], label="frames_with_prediction")
    plt.bar(summary_df["source_file"], summary_df["frames_with_accident_locked"], label="frames_with_accident_locked")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("frames")
    plt.title("CSV comparison: frame prediction / accident lock")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_frame_prediction_lock.png"), dpi=150)
    plt.close()

    # recent prediction count
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df["source_file"], summary_df["max_recent_prediction_count"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("count")
    plt.title("CSV comparison: max recent_prediction_count")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_max_recent_prediction_count.png"), dpi=150)
    plt.close()

    # pair candidate / strong / repeat
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df["source_file"], summary_df["total_pair_accident_candidate"], label="pair_accident_candidate")
    plt.bar(summary_df["source_file"], summary_df["total_strong_pair_candidate"], label="strong_pair_candidate")
    plt.bar(summary_df["source_file"], summary_df["total_repeat_strong"], label="repeat_strong")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("count")
    plt.title("CSV comparison: pair accident / strong / repeat")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_pair_accident_strong_repeat.png"), dpi=150)
    plt.close()

    # rear / post / gap strong
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df["source_file"], summary_df["total_rear_core"], label="rear_core")
    plt.bar(summary_df["source_file"], summary_df["total_post_evidence"], label="post_evidence")
    plt.bar(summary_df["source_file"], summary_df["total_gap_strong"], label="gap_strong")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("count")
    plt.title("CSV comparison: rear / post / gap strong")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_rear_post_gapstrong.png"), dpi=150)
    plt.close()

    # pair score / repeat
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df["source_file"], summary_df["max_pair_score"], label="max_pair_score")
    plt.bar(summary_df["source_file"], summary_df["max_repeat_count_window"], label="max_repeat_count_window")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("value")
    plt.title("CSV comparison: pair score / repeat window")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_pairscore_repeatwindow.png"), dpi=150)
    plt.close()

    # lane template
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df["source_file"], summary_df["max_lane_count"], label="max_lane_count")
    plt.bar(summary_df["source_file"], summary_df["template_confirmed_frames"], label="template_confirmed_frames")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("value")
    plt.title("CSV comparison: lane template status")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_lane_template_status.png"), dpi=150)
    plt.close()


# =========================================================
# 7. 실행
# =========================================================
def main():
    print("🚀 accident debug csv graph start")
    print("CSV_DIR:", CSV_DIR)
    print("GRAPH_DIR:", GRAPH_DIR)

    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "accident_debug_v5_5_*.csv")))

    if not csv_files:
        print("❌ CSV 파일이 없습니다.")
        return

    print(f"찾은 CSV 수: {len(csv_files)}")

    all_summary = []

    for csv_path in csv_files:
        print("처리 중:", csv_path)

        raw_df = load_debug_csv(csv_path)
        frame_df = aggregate_by_frame(raw_df)

        source_name = os.path.basename(csv_path)
        save_dir = os.path.join(GRAPH_DIR, os.path.splitext(source_name)[0])
        os.makedirs(save_dir, exist_ok=True)

        # 집계 CSV 저장
        frame_df.to_csv(
            os.path.join(save_dir, f"{os.path.splitext(source_name)[0]}_frame_summary.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        # 개별 파일 그래프 저장
        plot_per_file(frame_df, source_name, save_dir)

        # 파일별 요약
        summary = summarize_file(frame_df, raw_df, source_name)
        all_summary.append(summary)

    summary_df = pd.DataFrame(all_summary)
    summary_csv_path = os.path.join(GRAPH_DIR, "accident_debug_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    # 비교 그래프 저장
    plot_comparison(summary_df, GRAPH_DIR)

    print("✅ 완료")
    print("요약 CSV:", summary_csv_path)
    print("그래프 폴더:", GRAPH_DIR)


if __name__ == "__main__":
    main()