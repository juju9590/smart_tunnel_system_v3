# ==========================================
# 파일명: plot_accident_debug_V5_4_3.py
# 위치: analysis/
# 설명:
# - accident_debug_v5_4_2 / v5_4_3 CSV를 읽어서
# - 프레임 단위 그래프 / 파일 비교 그래프를 저장
# - V5.4.3 컬럼 구조 대응 버전
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
CSV_DIR = os.path.join(OUTPUT_ROOT, "analysis_v5_4_3")

# 그래프 저장 폴더
GRAPH_DIR = os.path.join(OUTPUT_ROOT, "analysis_v5_4_3_graph")
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
        "frame_accident",
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
        "post_evidence",

        "weak_candidate",
        "strong_candidate",
        "accident_candidate",

        "pair_repeat_candidate",
        "pair_consecutive_candidate",
        "repeat_strong_candidate",

        "confirmed",
        "template_confirmed",
    ]

    num_cols = [
        "frame_id",
        "frame_acc_ratio",
        "pair_id1",
        "pair_id2",
        "lane1",
        "lane2",
        "dist",
        "gap",
        "repeat_count_window",
        "accident_score",
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
        "frame_accident": "max",
        "frame_acc_ratio": "max",
        "pair_id1": "count",

        "accident_candidate": "sum",
        "confirmed": "sum",

        "rear_core": "sum",
        "post_evidence": "sum",
        "weak_candidate": "sum",
        "strong_candidate": "sum",
        "pair_repeat_candidate": "sum",
        "pair_consecutive_candidate": "sum",
        "repeat_strong_candidate": "sum",

        "accident_score": ["max", "mean"],
        "repeat_count_window": "max",

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
        "accident_candidate_sum": "candidate_count",
        "confirmed_sum": "confirmed_count",

        "rear_core_sum": "rear_core_count",
        "post_evidence_sum": "post_evidence_count",
        "weak_candidate_sum": "weak_candidate_count",
        "strong_candidate_sum": "strong_candidate_count",
        "pair_repeat_candidate_sum": "pair_repeat_candidate_count",
        "pair_consecutive_candidate_sum": "pair_consecutive_candidate_count",
        "repeat_strong_candidate_sum": "repeat_strong_candidate_count",

        "accident_score_max": "score_max",
        "accident_score_mean": "score_mean",
        "repeat_count_window_max": "repeat_count_window_max",

        "frame_accident_max": "frame_accident",
        "frame_acc_ratio_max": "frame_acc_ratio",
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
        "total_candidates": int(safe_col(frame_df, "candidate_count").sum()),
        "total_confirmed": int(safe_col(frame_df, "confirmed_count").sum()),

        "frames_with_candidate": int((safe_col(frame_df, "candidate_count") > 0).sum()),
        "frames_with_confirmed": int((safe_col(frame_df, "confirmed_count") > 0).sum()),

        "total_rear_core": int(safe_col(frame_df, "rear_core_count").sum()),
        "total_post_evidence": int(safe_col(frame_df, "post_evidence_count").sum()),
        "total_strong_candidate": int(safe_col(frame_df, "strong_candidate_count").sum()),
        "total_repeat_strong": int(safe_col(frame_df, "repeat_strong_candidate_count").sum()),

        "max_score": float(safe_col(frame_df, "score_max").max()),
        "mean_score": float(safe_col(frame_df, "score_mean").mean()),

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
    # 1) 후보 / 확정 / 페어 수
    # -----------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "pair_count"), label="pair_count")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "candidate_count"), label="candidate_count")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "confirmed_count"), label="confirmed_count")
    plt.xlabel("frame_id")
    plt.ylabel("count")
    plt.title(f"[{base_name}] pair / candidate / confirmed per frame")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_frame_pair_candidate_confirmed.png"), dpi=150)
    plt.close()

    # -----------------------------
    # 2) 조건별 카운트
    # -----------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "rear_core_count"), label="rear_core_count")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "post_evidence_count"), label="post_evidence_count")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "strong_candidate_count"), label="strong_candidate_count")
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
    # 3) 점수 추이
    # -----------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "score_max"), label="score_max")
    plt.plot(frame_df["frame_id"], safe_col(frame_df, "score_mean"), label="score_mean")
    plt.xlabel("frame_id")
    plt.ylabel("score")
    plt.title(f"[{base_name}] accident score trend")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_frame_score_trend.png"), dpi=150)
    plt.close()

    # -----------------------------
    # 4) 반복성 상태
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
    # 5) 차선 상태
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

    # 총 후보 / 총 확정
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df["source_file"], summary_df["total_candidates"], label="total_candidates")
    plt.bar(summary_df["source_file"], summary_df["total_confirmed"], label="total_confirmed")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("count")
    plt.title("CSV comparison: total candidates / confirmed")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_total_candidates_confirmed.png"), dpi=150)
    plt.close()

    # 후보 프레임 / 확정 프레임
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df["source_file"], summary_df["frames_with_candidate"], label="frames_with_candidate")
    plt.bar(summary_df["source_file"], summary_df["frames_with_confirmed"], label="frames_with_confirmed")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("frames")
    plt.title("CSV comparison: frames with candidate / confirmed")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_frames_with_candidate_confirmed.png"), dpi=150)
    plt.close()

    # 최대 점수 비교
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df["source_file"], summary_df["max_score"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("score")
    plt.title("CSV comparison: max accident score")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_max_score.png"), dpi=150)
    plt.close()

    # rear_core / strong / repeat strong
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df["source_file"], summary_df["total_rear_core"], label="total_rear_core")
    plt.bar(summary_df["source_file"], summary_df["total_strong_candidate"], label="total_strong_candidate")
    plt.bar(summary_df["source_file"], summary_df["total_repeat_strong"], label="total_repeat_strong")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("count")
    plt.title("CSV comparison: rear_core / strong / repeat_strong")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_rear_strong_repeat.png"), dpi=150)
    plt.close()

    # repeat window 비교
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df["source_file"], summary_df["max_repeat_count_window"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("value")
    plt.title("CSV comparison: max repeat_count_window")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_max_repeat_window.png"), dpi=150)
    plt.close()

    # 차선 템플릿 상태
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

    # V5.4.3 / V5.4.2 둘 다 읽을 수 있게
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "accident_debug_v5_4_*.csv")))

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