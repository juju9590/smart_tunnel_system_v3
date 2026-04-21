# ==========================================
# 파일명: eval_state_logger.py
# 위치: evaluation/eval_state_logger.py
# 설명:
# - 파이프라인 로그 CSV를 읽는다.
# - state 컬럼(dict 문자열)을 펼친다.
# - 영상별 GT CSV와 frame_id 기준으로 매칭한다.
# - 프레임별 상태 평가 로그 저장
# - 상태 평가 요약 저장
#
# 출력:
# - evaluation/outputs_summaries/state_eval_log_<video_name>.csv
# - evaluation/outputs_summaries/state_summary_<video_name>.csv
# ==========================================

import os
import pandas as pd
from datetime import datetime

from eval_utils import (
    load_csv,
    parse_state_column,
    standardize_state_pred_columns,
    standardize_state_gt_columns,
    normalize_state_name,
    prepare_pred_frame_column,
    merge_on_frame,
    build_confusion_counts,
    value_counts_to_summary_rows,
    save_csv,
    save_summary_and_confusion,
    ensure_dir,
)


# ==========================================
# [1] 경로 자동 설정
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EVAL_ROOT = r"D:\Finalpj_tunnel_V3\smart_tunnel_V3_eval"
GT_DIR = os.path.join(EVAL_ROOT, "ground_truth")
OUTPUT_DIR = os.path.join(EVAL_ROOT, "outputs")
SUMMARY_DIR = os.path.join(EVAL_ROOT, "outputs_summaries")

ensure_dir(GT_DIR)
ensure_dir(OUTPUT_DIR)
ensure_dir(SUMMARY_DIR)

# ==========================================
# [2] 사용자 설정
# ==========================================
PIPELINE_LOG_CSV = r"D:\Finalpj_tunnel_V3\smart_tunnel_V3_outputs\pipeline_v5_4\log_v5_4_20260418_000058.csv"

print("PIPELINE_LOG_CSV =", PIPELINE_LOG_CSV)
print("repr =", repr(PIPELINE_LOG_CSV))
print("exists =", os.path.exists(PIPELINE_LOG_CSV))
print("abspath =", os.path.abspath(PIPELINE_LOG_CSV))

# GT_CSV = os.path.join(GT_DIR, "state_gt_test_normal_2.csv")
# GT_CSV = os.path.join(GT_DIR, "state_gt_test_accident_1-1.csv")
GT_CSV = os.path.join(GT_DIR, "state_gt_test_congestion_2-1.csv")

gt_name = os.path.splitext(os.path.basename(GT_CSV))[0]
video_tag = gt_name.replace("state_gt_", "")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"state_eval_log_v5_4_{video_tag}_{ts}.csv")
SUMMARY_CSV = os.path.join(SUMMARY_DIR, f"state_summary_v5_4_{video_tag}_{ts}.csv")

# ==========================================
# [3] 평가 함수
# ==========================================
def evaluate_state_log(pipeline_log_csv, gt_csv, output_csv, summary_csv):
    if not os.path.exists(pipeline_log_csv):
        print("❌ 파이프라인 로그 CSV가 없습니다.")
        print("경로:", pipeline_log_csv)
        return

    if not os.path.exists(gt_csv):
        print("❌ GT CSV가 없습니다.")
        print("경로:", gt_csv)
        return

    print("📂 pipeline 로그 로드:", pipeline_log_csv)
    pred_df = load_csv(pipeline_log_csv)

    print("📂 GT 로드:", gt_csv)
    gt_df = load_csv(gt_csv)

    # --------------------------------------
    # 1) frame 컬럼 통일
    # --------------------------------------
    pred_df = prepare_pred_frame_column(pred_df)
    gt_df = standardize_state_gt_columns(gt_df)

    # --------------------------------------
    # 2) state 컬럼 펼치기
    # --------------------------------------
    pred_df = parse_state_column(pred_df, state_col="state")

    # --------------------------------------
    # 3) state 관련 예측 컬럼명 통일
    # --------------------------------------
    pred_df = standardize_state_pred_columns(pred_df)

    # --------------------------------------
    # 4) 상태명 정규화
    # --------------------------------------
    gt_df["gt_state"] = gt_df["gt_state"].apply(normalize_state_name)
    pred_df["pred_candidate_state"] = pred_df["pred_candidate_state"].apply(normalize_state_name)
    pred_df["pred_final_state"] = pred_df["pred_final_state"].apply(normalize_state_name)

    # --------------------------------------
    # 5) frame_id 기준 merge
    # --------------------------------------
    eval_df = merge_on_frame(pred_df, gt_df[["frame_id", "gt_state"]], how="inner")

    if len(eval_df) == 0:
        print("❌ frame_id 기준으로 매칭된 데이터가 없습니다.")
        return

    # --------------------------------------
    # 6) 평가 컬럼 생성
    # --------------------------------------
    eval_df["candidate_match"] = eval_df["pred_candidate_state"] == eval_df["gt_state"]
    eval_df["final_match"] = eval_df["pred_final_state"] == eval_df["gt_state"]

    def judge_error(row):
        if row["final_match"]:
            return "TP"
        return f"MISS_{row['gt_state']}_AS_{row['pred_final_state']}"

    eval_df["final_judge"] = eval_df.apply(judge_error, axis=1)

    # --------------------------------------
    # 7) 컬럼 순서 정리
    # --------------------------------------
    front_cols = [
        "frame_id",
        "gt_state",
        "pred_candidate_state",
        "pred_final_state",
        "candidate_match",
        "final_match",
        "final_judge",
        "frame_avg_speed",
        "buffer_avg_speed",
        "final_speed",
        "empty_frame",
        "hold_count",
        "buffer_size",
    ]

    existing_front_cols = [c for c in front_cols if c in eval_df.columns]
    remaining_cols = [c for c in eval_df.columns if c not in existing_front_cols]
    eval_df = eval_df[existing_front_cols + remaining_cols]

    # --------------------------------------
    # 8) 상세 로그 저장
    # --------------------------------------
    save_csv(eval_df, output_csv)
    print("✅ 상태 평가 로그 저장:", output_csv)

    # --------------------------------------
    # 9) summary 생성
    # --------------------------------------
    total_frames = len(eval_df)
    candidate_acc = round(eval_df["candidate_match"].mean() * 100, 2)
    final_acc = round(eval_df["final_match"].mean() * 100, 2)

    summary_rows = [
        {"metric": "total_frames", "value": total_frames},
        {"metric": "candidate_accuracy_percent", "value": candidate_acc},
        {"metric": "final_accuracy_percent", "value": final_acc},
    ]

    summary_rows += value_counts_to_summary_rows(eval_df["gt_state"], "gt_state")
    summary_rows += value_counts_to_summary_rows(eval_df["pred_final_state"], "pred_state")

    # final_speed 요약도 같이 넣기
    if "final_speed" in eval_df.columns:
        summary_rows += [
            {"metric": "final_speed_mean", "value": round(eval_df["final_speed"].mean(), 4)},
            {"metric": "final_speed_std", "value": round(eval_df["final_speed"].std(), 4)},
            {"metric": "final_speed_min", "value": round(eval_df["final_speed"].min(), 4)},
            {"metric": "final_speed_25%", "value": round(eval_df["final_speed"].quantile(0.25), 4)},
            {"metric": "final_speed_50%", "value": round(eval_df["final_speed"].quantile(0.50), 4)},
            {"metric": "final_speed_75%", "value": round(eval_df["final_speed"].quantile(0.75), 4)},
            {"metric": "final_speed_max", "value": round(eval_df["final_speed"].max(), 4)},
        ]

    summary_df = pd.DataFrame(summary_rows)

    confusion_df = build_confusion_counts(
        eval_df,
        gt_col="gt_state",
        pred_col="pred_final_state",
        labels=["NORMAL", "CONGESTION", "JAM"]
    )

    save_summary_and_confusion(summary_df, confusion_df, summary_csv)
    print("✅ 상태 평가 요약 저장:", summary_csv)

    # --------------------------------------
    # 10) 콘솔 요약
    # --------------------------------------
    print("\n[요약]")
    print("총 프레임 수:", total_frames)
    print("Candidate Accuracy(%):", candidate_acc)
    print("Final Accuracy(%):", final_acc)

    if "final_speed" in eval_df.columns:
        print("Final Speed Mean:", round(eval_df["final_speed"].mean(), 4))
        print("Final Speed Median:", round(eval_df["final_speed"].median(), 4))

    print("\n[GT 분포]")
    print(eval_df["gt_state"].value_counts().to_dict())

    print("\n[예측 분포]")
    print(eval_df["pred_final_state"].value_counts().to_dict())


# ==========================================
# [4] 실행
# ==========================================
if __name__ == "__main__":
    evaluate_state_log(
        pipeline_log_csv=PIPELINE_LOG_CSV,
        gt_csv=GT_CSV,
        output_csv=OUTPUT_CSV,
        summary_csv=SUMMARY_CSV,
    )