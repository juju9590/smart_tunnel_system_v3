# ==========================================
# 파일명: eval_utils.py
# 위치: scr/evaluation/eval_utils.py
# 설명:
# 성능평가용 공통 함수 모음
# - CSV 로드
# - dict 문자열 파싱
# - dict 펼치기
# - frame 컬럼 탐색
# - 상태명 정규화
# - 평가 결과 저장 보조
# ==========================================

import os
import ast
import json
import pandas as pd


# ==========================================
# 기본 유틸
# ==========================================
def ensure_dir(path):
    """
    디렉토리가 없으면 생성
    path가 파일 경로여도 상위 폴더 기준으로 생성 가능
    """
    if path is None or str(path).strip() == "":
        return

    # 파일 경로면 상위 폴더 사용
    if os.path.splitext(path)[1] != "":
        folder = os.path.dirname(path)
    else:
        folder = path

    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def load_csv(csv_path, encoding="utf-8-sig"):
    """
    CSV 로드
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_path}")

    return pd.read_csv(csv_path, encoding=encoding)


# ==========================================
# 문자열 파싱 / dict 펼치기
# ==========================================
def safe_parse(value):
    """
    문자열 형태의 dict / list / bool / number 를 안전하게 파싱
    실패하면 원본 그대로 반환
    """
    if pd.isna(value):
        return value

    if isinstance(value, (dict, list, int, float, bool)):
        return value

    if not isinstance(value, str):
        return value

    text = value.strip()

    if text == "":
        return value

    # python literal 우선
    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    # json 시도
    try:
        return json.loads(text)
    except Exception:
        pass

    return value


def is_scalar(x):
    """
    DataFrame 컬럼에 바로 넣기 쉬운 단일값인지 확인
    - list / dict / tuple / set 는 scalar 아님
    - pandas/numpy 배열류도 scalar 아님
    - None / NaN 은 scalar 취급
    """
    if x is None:
        return True

    # dict/list/tuple/set 은 scalar 아님
    if isinstance(x, (dict, list, tuple, set)):
        return False

    # 문자열/숫자/불리언은 scalar
    if isinstance(x, (str, int, float, bool)):
        return True

    # pandas/numpy scalar 여부 확인
    try:
        return pd.api.types.is_scalar(x)
    except Exception:
        return False


def flatten_dict(data, parent_key=""):
    """
    중첩 dict를 1차원 dict로 펼침
    예:
        {"debug": {"final_speed": 2.3}}
    ->
        {"debug_final_speed": 2.3}
    """
    items = {}

    for k, v in data.items():
        new_key = f"{parent_key}_{k}" if parent_key else str(k)

        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v

    return items


def expand_dict_column(df, col_name):
    """
    특정 컬럼의 dict 문자열을 파싱 후 컬럼으로 펼침
    """
    if col_name not in df.columns:
        raise ValueError(f"'{col_name}' 컬럼이 없습니다.")

    expanded_rows = []

    for _, row in df.iterrows():
        parsed = safe_parse(row[col_name])

        one = {}
        if isinstance(parsed, dict):
            flat = flatten_dict(parsed, parent_key=col_name)

            for k, v in flat.items():
                if is_scalar(v):
                    one[k] = v
                else:
                    one[k] = str(v)
        else:
            # dict가 아니면 원본 그대로 보존
            one[col_name] = parsed if is_scalar(parsed) else str(parsed)

        expanded_rows.append(one)

    expanded_df = pd.DataFrame(expanded_rows)

    df = df.drop(columns=[col_name])
    df = pd.concat([df.reset_index(drop=True), expanded_df.reset_index(drop=True)], axis=1)

    return df


def auto_expand_possible_dict_columns(df, sample_size=10):
    """
    전체 컬럼 중 dict 문자열처럼 보이는 컬럼 자동 확장
    """
    original_cols = list(df.columns)

    for col in original_cols:
        if col not in df.columns:
            continue

        non_null = df[col].dropna().astype(str)
        if len(non_null) == 0:
            continue

        samples = non_null.head(sample_size).tolist()
        maybe_dict = False

        for s in samples:
            s = s.strip()
            if s.startswith("{") and s.endswith("}"):
                maybe_dict = True
                break

        if maybe_dict:
            print(f"📌 dict 컬럼 확장: {col}")
            df = expand_dict_column(df, col)

    return df


# ==========================================
# 컬럼 / 값 정리
# ==========================================
def find_frame_column(df):
    """
    frame 컬럼 자동 탐색
    허용 후보:
    frame, frame_id, Frame, FRAME
    """
    candidates = ["frame", "frame_id", "Frame", "FRAME"]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError("프레임 컬럼(frame/frame_id)을 찾을 수 없습니다.")


def normalize_state_name(x):
    """
    상태명 통일
    최종 사용 상태:
    NORMAL / CONGESTION / JAM
    """
    if pd.isna(x):
        return None

    text = str(x).strip().upper()

    mapping = {
        "NORMAL": "NORMAL",
        "N": "NORMAL",

        "CONGESTION": "CONGESTION",
        "C": "CONGESTION",
        "CROWD": "CONGESTION",

        "JAM": "JAM",
        "J": "JAM",
        "STOP": "JAM",
        "STANDSTILL": "JAM",
    }

    return mapping.get(text, text)


def normalize_accident_label(x):
    """
    사고 라벨 통일
    최종 사용 라벨:
    ACCIDENT / NON_ACCIDENT
    """
    if pd.isna(x):
        return None

    text = str(x).strip().upper()

    mapping = {
        "ACCIDENT": "ACCIDENT",
        "TRUE": "ACCIDENT",
        "1": "ACCIDENT",
        "Y": "ACCIDENT",
        "YES": "ACCIDENT",

        "NON_ACCIDENT": "NON_ACCIDENT",
        "NORMAL": "NON_ACCIDENT",
        "FALSE": "NON_ACCIDENT",
        "0": "NON_ACCIDENT",
        "N": "NON_ACCIDENT",
        "NO": "NON_ACCIDENT",
    }

    return mapping.get(text, text)


def find_first_existing_column(df, candidates):
    """
    후보 컬럼명 중 먼저 존재하는 컬럼 반환
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def rename_if_exists(df, rename_map):
    """
    존재하는 컬럼만 골라 rename
    """
    valid_map = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=valid_map)


# ==========================================
# GT / 예측 merge 보조
# ==========================================
def prepare_gt_frame_column(gt_df):
    """
    GT DataFrame에서 frame 컬럼을 frame_id로 통일
    """
    gt_frame_col = find_frame_column(gt_df)
    gt_df = gt_df.rename(columns={gt_frame_col: "frame_id"})
    return gt_df


def prepare_pred_frame_column(pred_df):
    """
    예측 DataFrame에서 frame 컬럼을 frame_id로 통일
    """
    pred_frame_col = find_frame_column(pred_df)
    pred_df = pred_df.rename(columns={pred_frame_col: "frame_id"})
    return pred_df


def merge_on_frame(pred_df, gt_df, how="inner"):
    """
    frame_id 기준 병합
    """
    if "frame_id" not in pred_df.columns:
        raise ValueError("pred_df 에 frame_id 컬럼이 없습니다.")

    if "frame_id" not in gt_df.columns:
        raise ValueError("gt_df 에 frame_id 컬럼이 없습니다.")

    merged = pd.merge(pred_df, gt_df, on="frame_id", how=how)
    return merged


# ==========================================
# confusion / summary 보조
# ==========================================
def build_confusion_counts(df, gt_col, pred_col, labels):
    """
    confusion matrix 카운트용 DataFrame 생성
    """
    rows = []

    for gt in labels:
        row = {gt_col: gt}
        for pred in labels:
            count = len(df[(df[gt_col] == gt) & (df[pred_col] == pred)])
            row[f"pred_{pred}"] = count
        rows.append(row)

    return pd.DataFrame(rows)


def build_basic_accuracy_summary(df, gt_col, pred_col, metric_name="accuracy_percent"):
    """
    단순 정확도 요약 DataFrame 생성
    """
    if len(df) == 0:
        acc = 0.0
    else:
        acc = round((df[gt_col] == df[pred_col]).mean() * 100, 2)

    summary_df = pd.DataFrame([
        {"metric": "total_rows", "value": len(df)},
        {"metric": metric_name, "value": acc},
    ])
    return summary_df


def value_counts_to_summary_rows(series, prefix):
    """
    value_counts 결과를 summary 행 리스트로 변환
    """
    counts = series.value_counts(dropna=False).to_dict()
    rows = []

    for k, v in counts.items():
        key = "None" if pd.isna(k) else str(k)
        rows.append({
            "metric": f"{prefix}_{key}_count",
            "value": v
        })

    return rows


# ==========================================
# 저장 보조
# ==========================================
def save_csv(df, output_path, encoding="utf-8-sig"):
    """
    DataFrame CSV 저장
    """
    ensure_dir(output_path)
    df.to_csv(output_path, index=False, encoding=encoding)


def save_summary_and_confusion(summary_df, confusion_df, output_path, encoding="utf-8-sig"):
    """
    summary + confusion 을 한 CSV 파일에 순서대로 저장
    """
    ensure_dir(output_path)

    with open(output_path, "w", encoding=encoding, newline="") as f:
        summary_df.to_csv(f, index=False)
        if confusion_df is not None and len(confusion_df) > 0:
            f.write("\n")
            confusion_df.to_csv(f, index=False)


# ==========================================
# state 전용 보조
# ==========================================
def parse_state_column(df, state_col="state"):
    """
    state 컬럼(dict 문자열)을 펼침
    기대 예:
        {"state": "NORMAL", "debug": {...}}
    """
    return expand_dict_column(df, state_col)


def standardize_state_pred_columns(df):
    """
    state 예측 컬럼명을 공통 이름으로 정리
    결과 컬럼 예:
    - pred_state
    - pred_candidate_state
    - pred_final_state
    - frame_avg_speed
    - buffer_avg_speed
    - final_speed
    - empty_frame
    - hold_count
    - buffer_size
    """
    rename_map = {
        "state_state": "pred_state",
        "state_debug_candidate_state": "pred_candidate_state",
        "state_debug_final_state": "pred_final_state",
        "state_debug_frame_avg_speed": "frame_avg_speed",
        "state_debug_buffer_avg_speed": "buffer_avg_speed",
        "state_debug_final_speed": "final_speed",
        "state_debug_empty_frame": "empty_frame",
        "state_debug_hold_count": "hold_count",
        "state_debug_buffer_size": "buffer_size",
    }

    df = rename_if_exists(df, rename_map)

    if "pred_final_state" not in df.columns:
        if "pred_state" in df.columns:
            df["pred_final_state"] = df["pred_state"]
        else:
            raise ValueError("예측 상태 컬럼(pred_final_state / pred_state)을 찾을 수 없습니다.")

    if "pred_candidate_state" not in df.columns:
        df["pred_candidate_state"] = df["pred_final_state"]

    return df


def standardize_state_gt_columns(gt_df):
    """
    state GT 컬럼명을 공통 이름으로 정리
    최종:
    - frame_id
    - gt_state
    """
    gt_df = prepare_gt_frame_column(gt_df)

    gt_state_col = find_first_existing_column(
        gt_df,
        ["gt_state", "state", "label", "gt_label"]
    )

    if gt_state_col is None:
        raise ValueError("GT 상태 컬럼(gt_state/state/label/gt_label)을 찾을 수 없습니다.")

    gt_df = gt_df.rename(columns={gt_state_col: "gt_state"})
    return gt_df


# ==========================================
# accident 전용 보조
# ==========================================
def standardize_accident_gt_columns(gt_df):
    """
    accident GT 컬럼명 공통화
    최종:
    - frame_id
    - gt_accident
    """
    gt_df = prepare_gt_frame_column(gt_df)

    gt_accident_col = find_first_existing_column(
        gt_df,
        ["gt_accident", "accident", "label", "gt_label"]
    )

    if gt_accident_col is None:
        raise ValueError("GT 사고 컬럼(gt_accident/accident/label/gt_label)을 찾을 수 없습니다.")

    gt_df = gt_df.rename(columns={gt_accident_col: "gt_accident"})
    return gt_df