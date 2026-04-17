# ==========================================
# 파일명: eval_accident_logger.py
# 위치: evaluation/eval_accident_logger.py
# 설명:
# - 파이프라인 로그 CSV를 읽는다.
# - accident 컬럼(dict 문자열)을 펼친다.
# - 영상별 GT CSV와 frame_id 기준으로 매칭한다.
# - 프레임별 사고 평가 로그 저장
# - 사고 평가 요약 저장
#
# 출력:
# - evaluation/outputs_summaries/accident_eval_log_<video_name>.csv
# - evaluation/outputs_summaries/accident_summary_<video_name>.csv
# ==========================================

import os
import pandas as pd

from eval_utils import (
    load_csv,
    expand_dict_column,
    prepare_pred_frame_column,
    standardize_accident_gt_columns,
    normalize_accident_label,
    merge_on_frame,
    build_confusion_counts,
    value_counts_to_summary_rows,
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
PIPELINE_LOG_CSV = r"D:\Finalpj_tunnel_V3\smart_tunnel_V3_outputs\pipeline_v5_3\log_v5_3_20260417_105214.csv"
GT_CSV = os.path.join(GT_DIR, "state_gt_test_accident_1-1.csv")

gt_name = os.path.splitext(os.path.basename(GT_CSV))[0]
video_tag = gt_name.replace("accident_gt_", "")

OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"accident_eval_log_{video_tag}.csv")
SUMMARY_CSV = os.path.join(SUMMARY_DIR, f"accident_summary_{video_tag}.csv")


# ==========================================
# [3] accident 예측 컬럼 정리
# ==========================================
def standardize_accident_pred_columns(df):
    """
    accident 컬럼 펼친 후 예측 컬럼명을 공통 이름으로 정리
    최종 주요 컬럼:
    - pred_accident
    - pred_accident_label
    """
    rename_map = {
        "accident_accident": "pred_accident",
        "accident_debug_accident": "pred_accident_debug",
        "accident_state": "pred_accident_label",
    }

    df = rename_if_exists(df, rename_map)

    # 가장 우선적으로 pred_accident 사용
    if "pred_accident" in df.columns:
        df["pred_accident_label"] = df["pred_accident"]
    elif "pred_accident_label" in df.columns:
        pass
    elif "pred_accident_debug" in df.columns:
        df["pred_accident_label"] = df["pred_accident_debug"]
    else:
        raise ValueError("사고 예측 컬럼(accident_accident 등)을 찾을 수 없습니다.")

    return df


# ==========================================
# [4] 평가 함수
# ==========================================
def evaluate_accident_log(pipeline_log_csv, gt_csv, output_csv, summary_csv):
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
    gt_df = standardize_accident_gt_columns(gt_df)

    # --------------------------------------
    # 2) accident 컬럼 펼치기
    # --------------------------------------
    if "accident" not in pred_df.columns:
        raise ValueError("'accident' 컬럼이 로그에 없습니다.")

    pred_df = expand_dict_column(pred_df, "accident")

    # --------------------------------------
    # 3) 사고 관련 예측 컬럼명 통일
    # --------------------------------------
    pred_df = standardize_accident_pred_columns(pred_df)

    # --------------------------------------
    # 4) 라벨 정규화
    # --------------------------------------
    gt_df["gt_accident"] = gt_df["gt_accident"].apply(normalize_accident_label)
    pred_df["pred_accident_label"] = pred_df["pred_accident_label"].apply(normalize_accident_label)

    # --------------------------------------
    # 5) frame_id 기준 merge
    # --------------------------------------
    eval_df = merge_on_frame(pred_df, gt_df[["frame_id", "gt_accident"]], how="inner")

    if len(eval_df) == 0:
        print("❌ frame_id 기준으로 매칭된 데이터가 없습니다.")
        return

    # --------------------------------------
    # 6) 평가 컬럼 생성
    # --------------------------------------
    eval_df["accident_match"] = eval_df["pred_accident_label"] == eval_df["gt_accident"]

    def judge_error(row):
        if row["accident_match"]:
            return "TP"
        return f"MISS_{row['gt_accident']}_AS_{row['pred_accident_label']}"

    eval_df["accident_judge"] = eval_df.apply(judge_error, axis=1)

    # 바이너리 분류 관점 추가
    eval_df["is_gt_accident"] = eval_df["gt_accident"] == "ACCIDENT"
    eval_df["is_pred_accident"] = eval_df["pred_accident_label"] == "ACCIDENT"

    # --------------------------------------
    # 7) 컬럼 순서 정리
    # --------------------------------------
    front_cols = [
        "frame_id",
        "gt_accident",
        "pred_accident_label",
        "accident_match",
        "accident_judge",
        "is_gt_accident",
        "is_pred_accident",
    ]

    existing_front_cols = [c for c in front_cols if c in eval_df.columns]
    remaining_cols = [c for c in eval_df.columns if c not in existing_front_cols]
    eval_df = eval_df[existing_front_cols + remaining_cols]

    # --------------------------------------
    # 8) 상세 로그 저장
    # --------------------------------------
    save_csv(eval_df, output_csv)
    print("✅ 사고 평가 로그 저장:", output_csv)

    # --------------------------------------
    # 9) summary 생성
    # --------------------------------------
    total_frames = len(eval_df)
    acc = round(eval_df["accident_match"].mean() * 100, 2)

    # binary metrics
    tp = len(eval_df[(eval_df["is_gt_accident"] == True) & (eval_df["is_pred_accident"] == True)])
    tn = len(eval_df[(eval_df["is_gt_accident"] == False) & (eval_df["is_pred_accident"] == False)])
    fp = len(eval_df[(eval_df["is_gt_accident"] == False) & (eval_df["is_pred_accident"] == True)])
    fn = len(eval_df[(eval_df["is_gt_accident"] == True) & (eval_df["is_pred_accident"] == False)])

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    summary_rows = [
        {"metric": "total_frames", "value": total_frames},
        {"metric": "accident_accuracy_percent", "value": acc},
        {"metric": "tp", "value": tp},
        {"metric": "tn", "value": tn},
        {"metric": "fp", "value": fp},
        {"metric": "fn", "value": fn},
        {"metric": "precision", "value": precision},
        {"metric": "recall", "value": recall},
        {"metric": "f1_score", "value": f1},
    ]

    summary_rows += value_counts_to_summary_rows(eval_df["gt_accident"], "gt_accident")
    summary_rows += value_counts_to_summary_rows(eval_df["pred_accident_label"], "pred_accident")

    summary_df = pd.DataFrame(summary_rows)

    confusion_df = build_confusion_counts(
        eval_df,
        gt_col="gt_accident",
        pred_col="pred_accident_label",
        labels=["NON_ACCIDENT", "ACCIDENT"]
    )

    save_summary_and_confusion(summary_df, confusion_df, summary_csv)
    print("✅ 사고 평가 요약 저장:", summary_csv)

    # --------------------------------------
    # 10) 콘솔 요약
    # --------------------------------------
    print("\n[요약]")
    print("총 프레임 수:", total_frames)
    print("Accuracy(%):", acc)
    print("TP / TN / FP / FN:", tp, tn, fp, fn)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    print("\n[GT 분포]")
    print(eval_df["gt_accident"].value_counts().to_dict())

    print("\n[예측 분포]")
    print(eval_df["pred_accident_label"].value_counts().to_dict())


# ==========================================
# [5] 실행
# ==========================================
if __name__ == "__main__":
    evaluate_accident_log(
        pipeline_log_csv=PIPELINE_LOG_CSV,
        gt_csv=GT_CSV,
        output_csv=OUTPUT_CSV,
        summary_csv=SUMMARY_CSV,
    )