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
# [현재 버전 대응]
# - V5.5 사고 컬럼 우선 지원:
#   * accident_accident_locked
#   * accident_accident
#   * accident_frame_accident_prediction
# - 최종 평가 라벨은 위 우선순위로 결정
#
# 출력:
# - evaluation/outputs/accident_eval_log_<video_name>.csv
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

TUNNEL_DIR = r"D:\스마트터널시스템_개인\smart_tunnel_V3_개인_web\backend_flask\modules\tunnel"

GT_DIR = os.path.join(TUNNEL_DIR, "runtime_data", "eval_gt")
OUTPUT_DIR = os.path.join(TUNNEL_DIR, "runtime_data", "eval_results")
SUMMARY_DIR = os.path.join(TUNNEL_DIR, "runtime_data", "eval_summaries")

ensure_dir(GT_DIR)
ensure_dir(OUTPUT_DIR)
ensure_dir(SUMMARY_DIR)

# ==========================================
# [2] 사용자 설정
# ==========================================
PIPELINE_LOG_CSV = os.path.join(
    TUNNEL_DIR,
    "runtime_data",
    "eval_outputs",
    "pipeline_v6_1_0428",
    "accident_tunnel_samae_log_20260428_151520.csv"  # 로그 파일명 붙여넣기
)

# GT_CSV = os.path.join(GT_DIR, "gt_accident_gubong.csv") #구봉터널 정답
# GT_CSV = os.path.join(GT_DIR, "gt_accident_sangju.csv") #상주터널 정답
GT_CSV = os.path.join(GT_DIR, "gt_accident_samae.csv") #사매터널 정답
# GT_CSV = os.path.join(GT_DIR, "gt_congestion_jam_5min.csv") #혼잡+정체 정답


gt_name = os.path.splitext(os.path.basename(GT_CSV))[0]
video_tag = (
    gt_name
    .replace("accident_gt_", "")
    .replace("state_gt_", "")
    .replace("gt_", "")
)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"accident_eval_log_{video_tag}.csv")
SUMMARY_CSV = os.path.join(SUMMARY_DIR, f"accident_summary_{video_tag}.csv")


# ==========================================
# [3] bool / label 정규화 유틸
# ==========================================
def to_bool_safe(v):
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in ["true", "1", "yes", "y", "accident"]


def bool_to_accident_label(v):
    return "ACCIDENT" if to_bool_safe(v) else "NON_ACCIDENT"


# ==========================================
# [4] accident 예측 컬럼 정리
# ==========================================
def standardize_accident_pred_columns(df):
    """
    accident 컬럼 펼친 후 예측 컬럼명을 공통 이름으로 정리

    현재 버전 우선순위:
    1) accident_accident_locked
    2) accident_accident
    3) accident_frame_accident_prediction

    최종 주요 컬럼:
    - pred_accident_source
    - pred_accident_label
    """
    rename_map = {
        # 예전 버전 호환
        "accident_accident": "pred_accident_bool",
        "accident_debug_accident": "pred_accident_debug",
        "accident_state": "pred_accident_label_old",

        # 현재 버전(V5.5) 호환
        "accident_accident_locked": "pred_accident_locked",
        "accident_frame_accident_prediction": "pred_frame_accident_prediction",
        "accident_recent_prediction_count": "pred_recent_prediction_count",
        "accident_acc_ratio": "pred_acc_ratio",
    }

    df = rename_if_exists(df, rename_map)

    # 최종 평가용 기준 선택
    if "pred_accident_locked" in df.columns:
        df["pred_accident_source"] = "accident_locked"
        df["pred_accident_label"] = df["pred_accident_locked"].apply(bool_to_accident_label)

    elif "pred_accident_bool" in df.columns:
        df["pred_accident_source"] = "accident"
        df["pred_accident_label"] = df["pred_accident_bool"].apply(bool_to_accident_label)

    elif "pred_frame_accident_prediction" in df.columns:
        df["pred_accident_source"] = "frame_accident_prediction"
        df["pred_accident_label"] = df["pred_frame_accident_prediction"].apply(bool_to_accident_label)

    elif "pred_accident_label_old" in df.columns:
        df["pred_accident_source"] = "legacy_label"
        df["pred_accident_label"] = df["pred_accident_label_old"].apply(normalize_accident_label)

    elif "pred_accident_debug" in df.columns:
        df["pred_accident_source"] = "legacy_debug"
        df["pred_accident_label"] = df["pred_accident_debug"].apply(normalize_accident_label)

    else:
        raise ValueError(
            "사고 예측 컬럼을 찾을 수 없습니다. "
            "accident_accident_locked / accident_accident / "
            "accident_frame_accident_prediction 중 하나가 필요합니다."
        )

    return df


# ==========================================
# [5] 평가 함수
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
        gt_label = row["gt_accident"]
        pred_label = row["pred_accident_label"]

        if gt_label == "ACCIDENT" and pred_label == "ACCIDENT":
            return "TP"
        if gt_label == "NON_ACCIDENT" and pred_label == "NON_ACCIDENT":
            return "TN"
        if gt_label == "NON_ACCIDENT" and pred_label == "ACCIDENT":
            return "FP"
        if gt_label == "ACCIDENT" and pred_label == "NON_ACCIDENT":
            return "FN"
        return f"MISS_{gt_label}_AS_{pred_label}"

    eval_df["accident_judge"] = eval_df.apply(judge_error, axis=1)

    # 바이너리 분류 관점
    eval_df["is_gt_accident"] = eval_df["gt_accident"] == "ACCIDENT"
    eval_df["is_pred_accident"] = eval_df["pred_accident_label"] == "ACCIDENT"

    # --------------------------------------
    # 7) 컬럼 순서 정리
    # --------------------------------------
    front_cols = [
        "frame_id",
        "gt_accident",
        "pred_accident_label",
        "pred_accident_source",
        "accident_match",
        "accident_judge",
        "is_gt_accident",
        "is_pred_accident",
    ]

    extra_cols_preferred = [
        "pred_accident_locked",
        "pred_accident_bool",
        "pred_frame_accident_prediction",
        "pred_recent_prediction_count",
        "pred_acc_ratio",
    ]

    existing_front_cols = [c for c in front_cols if c in eval_df.columns]
    existing_extra_cols = [c for c in extra_cols_preferred if c in eval_df.columns]
    remaining_cols = [
        c for c in eval_df.columns
        if c not in existing_front_cols + existing_extra_cols
    ]

    eval_df = eval_df[existing_front_cols + existing_extra_cols + remaining_cols]

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

    if "pred_recent_prediction_count" in eval_df.columns:
        summary_rows.append({
            "metric": "max_recent_prediction_count",
            "value": float(pd.to_numeric(eval_df["pred_recent_prediction_count"], errors="coerce").max())
        })

    if "pred_accident_source" in eval_df.columns:
        pred_source = eval_df["pred_accident_source"].iloc[0]
        summary_rows.append({
            "metric": "prediction_source_used",
            "value": pred_source
        })

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

    if "pred_recent_prediction_count" in eval_df.columns:
        print("Max recent_prediction_count:",
              pd.to_numeric(eval_df["pred_recent_prediction_count"], errors="coerce").max())

    if "pred_accident_source" in eval_df.columns:
        print("Prediction source used:", eval_df["pred_accident_source"].iloc[0])

    print("\n[GT 분포]")
    print(eval_df["gt_accident"].value_counts().to_dict())

    print("\n[예측 분포]")
    print(eval_df["pred_accident_label"].value_counts().to_dict())


# ==========================================
# [6] 실행
# ==========================================
if __name__ == "__main__":
    evaluate_accident_log(
        pipeline_log_csv=PIPELINE_LOG_CSV,
        gt_csv=GT_CSV,
        output_csv=OUTPUT_CSV,
        summary_csv=SUMMARY_CSV,
    )