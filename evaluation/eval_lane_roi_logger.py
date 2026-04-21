# ==========================================
# 파일명: eval_lane_roi_logger.py
# 위치: evaluation/eval_lane_roi_logger.py
# 설명:
# - pipeline 로그 CSV를 읽는다.
# - analysis 컬럼(dict 문자열)을 펼친다.
# - lane_gt.csv, roi_gt.csv 와 frame_id 기준으로 매칭한다.
# - 프레임별 차선/ROI 평가 로그 저장
# - 차선/ROI 요약 저장
#
# 출력:
# - smart_tunnel_V3_eval/outputs/lane_roi_eval_log_<video_tag>.csv
# - smart_tunnel_V3_eval/outputs_summaries/lane_roi_summary_<video_tag>.csv
# ==========================================

import os
import pandas as pd
from datetime import datetime

from eval_utils import (
    load_csv,
    expand_dict_column,
    prepare_pred_frame_column,
    prepare_gt_frame_column,
    find_first_existing_column,
    merge_on_frame,
    build_basic_accuracy_summary,
    save_csv,
    save_summary_and_confusion,
    ensure_dir,
    rename_if_exists,
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

# 영상별 GT 파일명 맞춰서 변경해서 사용
LANE_GT_CSV = os.path.join(GT_DIR, "lane_gt_test_congestion_2-1.csv")
ROI_GT_CSV = os.path.join(GT_DIR, "roi_gt_test_congestion_2-1.csv")

lane_gt_name = os.path.splitext(os.path.basename(LANE_GT_CSV))[0]
video_tag = lane_gt_name.replace("lane_gt_", "")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"lane_roi_eval_log_{video_tag}_{ts}.csv")
SUMMARY_CSV = os.path.join(SUMMARY_DIR, f"lane_roi_summary_{video_tag}_{ts}.csv")


# ==========================================
# [3] analysis 예측 컬럼 정리
# ==========================================
def standardize_analysis_pred_columns(df):
    print("정리 전 컬럼명:", df.columns.tolist())

    """
    analysis 컬럼 펼친 후 / 또는 개별 컬럼 로그일 때
    필요한 예측 컬럼명을 공통 이름으로 정리
    """

    # 1차: 존재하는 컬럼명을 공통 이름으로 바로 rename
    rename_map = {
        # lane
        "analysis_lane_count": "pred_lane_count",
        "lane_count": "pred_lane_count",

        # roi
        "analysis_roi_raw_y1": "pred_roi_y1",
        "analysis_roi_raw_y2": "pred_roi_y2",
        "analysis_raw_y1": "pred_roi_y1",
        "analysis_raw_y2": "pred_roi_y2",
        "roi_raw_y1": "pred_roi_y1",
        "roi_raw_y2": "pred_roi_y2",
        "raw_y1": "pred_roi_y1",
        "raw_y2": "pred_roi_y2",
        "roi_y1": "pred_roi_y1",
        "roi_y2": "pred_roi_y2",

        # 기타
        "analysis_roi_span": "pred_roi_span",
        "roi_span": "pred_roi_span",
        "analysis_roi_used_fallback": "pred_roi_used_fallback",
        "roi_used_fallback": "pred_roi_used_fallback",
        "roi_fixed": "pred_roi_fixed",
        "analysis_template_phase": "template_phase",
        "template_phase": "template_phase",
        "analysis_template_confirmed": "template_confirmed",
        "template_confirmed": "template_confirmed",
    }

    # 실제 존재하는 컬럼만 rename
    for old_col, new_col in rename_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})

    # 2차: 혹시 아직 못 찾았으면 후보군에서 탐색
    if "pred_lane_count" not in df.columns:
        alt_lane_col = find_first_existing_column(
            df,
            ["pred_lane_count", "lane_count", "analysis_lane_count"]
        )
        if alt_lane_col is not None:
            df = df.rename(columns={alt_lane_col: "pred_lane_count"})

    if "pred_roi_y1" not in df.columns:
        alt_y1 = find_first_existing_column(
            df,
            ["pred_roi_y1", "roi_y1", "roi_raw_y1", "raw_y1", "analysis_roi_raw_y1", "analysis_raw_y1"]
        )
        if alt_y1 is not None:
            df = df.rename(columns={alt_y1: "pred_roi_y1"})

    if "pred_roi_y2" not in df.columns:
        alt_y2 = find_first_existing_column(
            df,
            ["pred_roi_y2", "roi_y2", "roi_raw_y2", "raw_y2", "analysis_roi_raw_y2", "analysis_raw_y2"]
        )
        if alt_y2 is not None:
            df = df.rename(columns={alt_y2: "pred_roi_y2"})

    print("정리 후 컬럼명:", df.columns.tolist())
    return df



def standardize_lane_gt_columns(gt_df):
    """
    lane GT 컬럼 정리
    최종:
    - frame_id
    - gt_lane_count
    """
    gt_df = prepare_gt_frame_column(gt_df)

    lane_col = find_first_existing_column(
        gt_df,
        ["gt_lane_count", "lane_count", "lane", "gt_lane"]
    )

    if lane_col is None:
        raise ValueError("lane GT 컬럼(gt_lane_count/lane_count/lane/gt_lane)을 찾을 수 없습니다.")

    gt_df = gt_df.rename(columns={lane_col: "gt_lane_count"})
    return gt_df


def standardize_roi_gt_columns(gt_df):
    """
    roi GT 컬럼 정리
    최종:
    - frame_id
    - gt_roi_y1
    - gt_roi_y2
    """
    gt_df = prepare_gt_frame_column(gt_df)

    y1_col = find_first_existing_column(
        gt_df,
        ["gt_roi_y1", "roi_y1", "y1"]
    )
    y2_col = find_first_existing_column(
        gt_df,
        ["gt_roi_y2", "roi_y2", "y2"]
    )

    if y1_col is None or y2_col is None:
        raise ValueError("roi GT 컬럼(gt_roi_y1/roi_y1/y1, gt_roi_y2/roi_y2/y2)을 찾을 수 없습니다.")

    gt_df = gt_df.rename(columns={
        y1_col: "gt_roi_y1",
        y2_col: "gt_roi_y2",
    })
    return gt_df


# ==========================================
# [4] 평가 함수
# ==========================================
def evaluate_lane_roi_log(pipeline_log_csv, lane_gt_csv, roi_gt_csv, output_csv, summary_csv):
    if not os.path.exists(pipeline_log_csv):
        print("❌ 파이프라인 로그 CSV가 없습니다.")
        print("경로:", pipeline_log_csv)
        return

    if not os.path.exists(lane_gt_csv):
        print("❌ lane GT CSV가 없습니다.")
        print("경로:", lane_gt_csv)
        return

    if not os.path.exists(roi_gt_csv):
        print("❌ roi GT CSV가 없습니다.")
        print("경로:", roi_gt_csv)
        return

    print("📂 pipeline 로그 로드:", pipeline_log_csv)
    pred_df = load_csv(pipeline_log_csv)
    print("로그 컬럼명:", pred_df.columns.tolist())

    print("📂 lane GT 로드:", lane_gt_csv)
    lane_gt_df = load_csv(lane_gt_csv)

    print("📂 roi GT 로드:", roi_gt_csv)
    roi_gt_df = load_csv(roi_gt_csv)

    # --------------------------------------
    # 1) frame 컬럼 통일
    # --------------------------------------
    pred_df = prepare_pred_frame_column(pred_df)
    lane_gt_df = standardize_lane_gt_columns(lane_gt_df)
    roi_gt_df = standardize_roi_gt_columns(roi_gt_df)

    # --------------------------------------
    # 2) analysis 컬럼 펼치기
    # --------------------------------------
    # analysis가 있으면 펼치고,
    # 없으면 이미 컬럼으로 저장된 것으로 보고 그대로 진행
    if "analysis" in pred_df.columns:
        pred_df = expand_dict_column(pred_df, "analysis")
    else:
        print("ℹ️ 'analysis' 컬럼이 없어 펼치기 생략 - 개별 컬럼 로그로 간주합니다.")

    # --------------------------------------
    # 3) 예측 컬럼명 통일
    # --------------------------------------
    pred_df = standardize_analysis_pred_columns(pred_df)

    # 필수 컬럼 확인
    if "pred_lane_count" not in pred_df.columns:
        raise ValueError("예측 lane_count 컬럼을 찾을 수 없습니다.")

    if "pred_roi_y1" not in pred_df.columns or "pred_roi_y2" not in pred_df.columns:
        raise ValueError("예측 ROI y1/y2 컬럼을 찾을 수 없습니다.")

    # --------------------------------------
    # 4) lane / roi 각각 frame_id 기준 merge
    # --------------------------------------
    lane_eval_df = merge_on_frame(pred_df, lane_gt_df[["frame_id", "gt_lane_count"]], how="inner")
    lane_roi_eval_df = merge_on_frame(lane_eval_df, roi_gt_df[["frame_id", "gt_roi_y1", "gt_roi_y2"]], how="inner")

    if len(lane_roi_eval_df) == 0:
        print("❌ frame_id 기준으로 매칭된 데이터가 없습니다.")
        return

    # --------------------------------------
    # 5) lane 평가
    # --------------------------------------
    lane_roi_eval_df["lane_match"] = lane_roi_eval_df["pred_lane_count"] == lane_roi_eval_df["gt_lane_count"]

    # --------------------------------------
    # 6) roi 평가
    # --------------------------------------
    lane_roi_eval_df["roi_y1_error"] = (lane_roi_eval_df["pred_roi_y1"] - lane_roi_eval_df["gt_roi_y1"]).abs()
    lane_roi_eval_df["roi_y2_error"] = (lane_roi_eval_df["pred_roi_y2"] - lane_roi_eval_df["gt_roi_y2"]).abs()
    lane_roi_eval_df["roi_mean_error"] = (
        lane_roi_eval_df["roi_y1_error"] + lane_roi_eval_df["roi_y2_error"]
    ) / 2.0

    # 허용 오차 기준
    ROI_TOLERANCE = 20
    lane_roi_eval_df["roi_match"] = (
        (lane_roi_eval_df["roi_y1_error"] <= ROI_TOLERANCE) &
        (lane_roi_eval_df["roi_y2_error"] <= ROI_TOLERANCE)
    )

    # --------------------------------------
    # 7) 컬럼 순서 정리
    # --------------------------------------
    front_cols = [
        "frame_id",

        "gt_lane_count",
        "pred_lane_count",
        "lane_match",

        "gt_roi_y1",
        "pred_roi_y1",
        "roi_y1_error",

        "gt_roi_y2",
        "pred_roi_y2",
        "roi_y2_error",

        "roi_mean_error",
        "roi_match",

        "pred_roi_span",
        "pred_roi_used_fallback",
        "template_phase",
        "template_confirmed",
    ]

    existing_front_cols = [c for c in front_cols if c in lane_roi_eval_df.columns]
    remaining_cols = [c for c in lane_roi_eval_df.columns if c not in existing_front_cols]
    lane_roi_eval_df = lane_roi_eval_df[existing_front_cols + remaining_cols]

    # --------------------------------------
    # 8) 상세 로그 저장
    # --------------------------------------
    save_csv(lane_roi_eval_df, output_csv)
    print("✅ lane/roi 평가 로그 저장:", output_csv)

    # --------------------------------------
    # 9) summary 생성
    # --------------------------------------
    lane_summary_df = build_basic_accuracy_summary(
        lane_roi_eval_df,
        gt_col="gt_lane_count",
        pred_col="pred_lane_count",
        metric_name="lane_accuracy_percent"
    )

    roi_match_percent = round(lane_roi_eval_df["roi_match"].mean() * 100, 2)
    roi_y1_mae = round(lane_roi_eval_df["roi_y1_error"].mean(), 4)
    roi_y2_mae = round(lane_roi_eval_df["roi_y2_error"].mean(), 4)
    roi_mean_mae = round(lane_roi_eval_df["roi_mean_error"].mean(), 4)

    extra_rows = [
        {"metric": "total_frames", "value": len(lane_roi_eval_df)},
        {"metric": "roi_match_percent", "value": roi_match_percent},
        {"metric": "roi_y1_mae", "value": roi_y1_mae},
        {"metric": "roi_y2_mae", "value": roi_y2_mae},
        {"metric": "roi_mean_mae", "value": roi_mean_mae},
    ]

    summary_df = pd.concat([
        pd.DataFrame(extra_rows),
        lane_summary_df
    ], ignore_index=True)

    save_summary_and_confusion(summary_df, confusion_df=None, output_path=summary_csv)
    print("✅ lane/roi 평가 요약 저장:", summary_csv)

    # --------------------------------------
    # 10) 콘솔 요약
    # --------------------------------------
    lane_acc = round(lane_roi_eval_df["lane_match"].mean() * 100, 2)

    print("\n[요약]")
    print("총 프레임 수:", len(lane_roi_eval_df))
    print("Lane Accuracy(%):", lane_acc)
    print("ROI Match(%):", roi_match_percent)
    print("ROI y1 MAE:", roi_y1_mae)
    print("ROI y2 MAE:", roi_y2_mae)
    print("ROI Mean MAE:", roi_mean_mae)

    print("\n[Lane GT 분포]")
    print(lane_roi_eval_df["gt_lane_count"].value_counts().to_dict())

    print("\n[Lane 예측 분포]")
    print(lane_roi_eval_df["pred_lane_count"].value_counts().to_dict())


# ==========================================
# [5] 실행
# ==========================================
if __name__ == "__main__":
    evaluate_lane_roi_log(
        pipeline_log_csv=PIPELINE_LOG_CSV,
        lane_gt_csv=LANE_GT_CSV,
        roi_gt_csv=ROI_GT_CSV,
        output_csv=OUTPUT_CSV,
        summary_csv=SUMMARY_CSV,
    )