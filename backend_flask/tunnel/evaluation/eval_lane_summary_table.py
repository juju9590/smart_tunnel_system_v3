# ==========================================
# 파일명: eval_lane_summary_table.py
# 위치: backend_flask/tunnel/evaluation/eval_lane_summary_table.py
# 설명:
# - lane_eval_*.csv와 lane_gt_final.csv를 읽어
# - PPT/최종보고서용 가로 비교 표를 콘솔/CSV/Excel로 출력한다.
# - AI 파이프라인 로직은 수정하지 않는 평가 전용 스크립트.
# - pandas 없이 표준 csv 모듈만으로 동작한다.
# ==========================================

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "runtime_data"
CSV_LOG_DIR = RUNTIME_DIR / "csv_logs"
EVAL_GT_DIR = RUNTIME_DIR / "eval_gt"
EVAL_RESULTS_DIR = RUNTIME_DIR / "eval_results"
EVAL_SUMMARIES_DIR = RUNTIME_DIR / "eval_summaries"
DEFAULT_GT_PATH = EVAL_GT_DIR / "lane_gt_final.csv"
MISSING = "확인 불가"

TARGET_VIDEOS = [
    {
        "key": "gubong_accident",
        "name": "구봉 사고영상",
        "filename": "accident_tunnel_gubong.mp4",
        "expected_lane_count": 2,
        "expected_target_lane_count": 2,
        "expected_template_confirmed": True,
    },
    {
        "key": "sangju_accident",
        "name": "상주 사고영상",
        "filename": "accident_tunnel_sangju.mp4",
        "expected_lane_count": 2,
        "expected_target_lane_count": 2,
        "expected_template_confirmed": True,
    },
    {
        "key": "congestion_jam",
        "name": "혼잡_잼 영상",
        "filename": "congestion_jam_5min.mp4",
        "expected_lane_count": 2,
        "expected_target_lane_count": 2,
        "expected_template_confirmed": True,
    },
]


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_latest(pattern):
    files = sorted(CSV_LOG_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def find_gt_file():
    if DEFAULT_GT_PATH.exists():
        return DEFAULT_GT_PATH
    return None


def first_key(row, candidates):
    for key in candidates:
        if key in row:
            return key
    return None


def first_key_in_rows(rows, candidates):
    if not rows:
        return None
    keys = set()
    for row in rows[:20]:
        keys.update(row.keys())
    for key in candidates:
        if key in keys:
            return key
    return None


def to_int(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value).strip()))
    except Exception:
        return None


def to_bool(value):
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in ("true", "1", "yes", "y", "o", "loaded", "load", "사용", "완료")


def format_rate(value):
    if value is None:
        return MISSING
    return f"{float(value):.1f}"


def format_dist(counter):
    if counter is None:
        return MISSING
    return str({int(k): int(v) for k, v in sorted(counter.items())})


def filter_for_video(rows, target):
    if not rows:
        return []

    filename_stem = Path(target["filename"]).stem.lower()
    needles = [
        target["filename"].lower(),
        filename_stem,
        target["name"].lower(),
        target["key"].lower(),
    ]
    candidates = [
        "demo_video_filename",
        "source_name",
        "filename",
        "video_filename",
        "video_name",
        "source_filename",
        "cctv_name",
        "name",
    ]

    for col in candidates:
        if col not in rows[0]:
            continue

        matched = []
        for row in rows:
            value = str(row.get(col, "")).lower()
            if any(needle in value for needle in needles):
                matched.append(row)

        if matched:
            return matched

    # 영상 식별 컬럼이 없고 단일 영상 로그라면 전체를 사용한다.
    return list(rows)


def get_gt_segment(gt_rows, target):
    video_gt = filter_for_video(gt_rows, target)
    return video_gt[0] if video_gt else {}


def expected_from_gt(gt_row, target, key):
    value = gt_row.get(key) if gt_row else None
    if value is None or str(value).strip() == "":
        return target.get(key)
    return value


def apply_gt_segment(pred_rows, gt_row):
    rows = list(pred_rows)
    if not gt_row:
        return rows

    frame_col = first_key_in_rows(rows, ["frame_id", "frame", "Frame", "FRAME"])
    start_frame = to_int(gt_row.get("start_frame"))
    end_frame = to_int(gt_row.get("end_frame"))

    if not frame_col or start_frame is None or end_frame is None:
        return rows

    filtered = []
    for row in rows:
        frame = to_int(row.get(frame_col))
        if frame is None:
            continue
        if start_frame <= frame <= end_frame:
            one = dict(row)
            one["frame_id"] = frame
            filtered.append(one)

    return filtered


def int_values(rows, col):
    values = []
    for row in rows:
        value = to_int(row.get(col))
        if value is not None:
            values.append(value)
    return values


def distribution(rows, col):
    values = int_values(rows, col)
    if not values:
        return None
    return Counter(values)


def has_reestimate(rows, gt_row=None):
    status_col = first_key_in_rows(rows, ["lane_reestimate_status", "reestimate_status"])
    if status_col:
        for row in rows:
            value = str(row.get(status_col, "")).strip().lower()
            if value in ("reestimating", "reestimated") or "reestimate" in value or "재추정" in value:
                return "O"

    if gt_row:
        note = str(gt_row.get("note", ""))
        if "재추정" in note or "reestimate" in note.lower():
            return "O"

    note_col = first_key_in_rows(rows, ["note", "notes", "event", "events", "message"])
    if note_col:
        for row in rows:
            value = str(row.get(note_col, ""))
            if "재추정" in value or "reestimate" in value.lower():
                return "O"

    return "X"


def has_lane_memory_loaded(rows):
    col = first_key_in_rows(rows, ["lane_memory_loaded", "memory_loaded", "lane_template_loaded"])
    if not col:
        return MISSING
    return "O" if any(to_bool(row.get(col)) for row in rows) else "X"


def final_judgement(lane_accuracy, target_match_rate):
    if lane_accuracy is None or target_match_rate is None:
        return MISSING
    score = min(lane_accuracy, target_match_rate)
    if score >= 90:
        return "성공"
    if score >= 70:
        return "부분성공"
    return "개선필요"


def evaluate_one(pred_rows, gt_rows, target):
    video_pred = filter_for_video(pred_rows, target)
    gt_row = get_gt_segment(gt_rows, target)
    eval_rows = apply_gt_segment(video_pred, gt_row)
    total_frames = len(eval_rows)

    lane_col = first_key_in_rows(
        eval_rows,
        ["lane_count", "pred_lane_count", "lane_count_estimated", "estimated_lane_count"],
    )
    template_col = first_key_in_rows(eval_rows, ["template_confirmed", "lane_template_confirmed"])
    gt_lane_col = first_key_in_rows(
        eval_rows,
        ["gt_lane_count", "expected_lane_count", "lane_count_gt", "lane_count_label"],
    )

    expected_lane = int(expected_from_gt(gt_row, target, "expected_lane_count"))
    expected_target = int(expected_from_gt(gt_row, target, "expected_target_lane_count"))
    expected_template = to_bool(expected_from_gt(gt_row, target, "expected_template_confirmed"))

    if lane_col and total_frames > 0:
        lane_values = int_values(eval_rows, lane_col)
        pred_dist = Counter(lane_values) if lane_values else None
        lane_accuracy = (
            round(sum(1 for value in lane_values if value == expected_lane) / len(lane_values) * 100, 1)
            if lane_values
            else None
        )
        target_match_rate = (
            round(sum(1 for value in lane_values if value == expected_target) / len(lane_values) * 100, 1)
            if lane_values
            else None
        )
    else:
        pred_dist = None
        lane_accuracy = None
        target_match_rate = None

    if total_frames > 0:
        gt_dist = Counter({expected_lane: total_frames})
    elif gt_lane_col:
        gt_dist = distribution(eval_rows, gt_lane_col)
    else:
        gt_dist = None

    if template_col and total_frames > 0:
        confirmed_count = sum(
            1
            for row in eval_rows
            if to_bool(row.get(template_col)) == expected_template
        )
        template_rate = round(confirmed_count / total_frames * 100, 1)
    else:
        template_rate = None

    return {
        "목표 차선 수": expected_lane,
        "총 평가 프레임 수": total_frames if total_frames > 0 else MISSING,
        "Lane Accuracy(%)": format_rate(lane_accuracy),
        "GT 분포": format_dist(gt_dist),
        "예측 분포": format_dist(pred_dist),
        "Target Match Rate(%)": format_rate(target_match_rate),
        "Template Confirm Rate(%)": format_rate(template_rate),
        "재추정 사용 여부": has_reestimate(eval_rows, gt_row) if total_frames > 0 else MISSING,
        "Lane Memory 로드": has_lane_memory_loaded(eval_rows) if total_frames > 0 else MISSING,
        "최종 판정": final_judgement(lane_accuracy, target_match_rate),
    }


def build_wide_rows(pred_rows, gt_rows):
    metrics = [
        "목표 차선 수",
        "총 평가 프레임 수",
        "Lane Accuracy(%)",
        "GT 분포",
        "예측 분포",
        "Target Match Rate(%)",
        "Template Confirm Rate(%)",
        "재추정 사용 여부",
        "Lane Memory 로드",
        "최종 판정",
    ]
    results = {target["name"]: evaluate_one(pred_rows, gt_rows, target) for target in TARGET_VIDEOS}

    rows = []
    for metric in metrics:
        row = {"구분": "Lane", "항목": metric}
        for target in TARGET_VIDEOS:
            row[target["name"]] = results[target["name"]][metric]
        rows.append(row)
    return rows


def print_markdown_table(rows, headers):
    print("\n차선추정 & 차선관리 성능 평가")
    widths = {
        header: max(len(str(header)), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }
    header_line = "| " + " | ".join(str(h).ljust(widths[h]) for h in headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    print(header_line)
    print(sep_line)
    for row in rows:
        print("| " + " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers) + " |")


def save_csv(rows, headers, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"lane_eval_summary_{timestamp}.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ CSV 저장: {path}")
    return path


def save_excel_optional(rows, headers, output_dir):
    try:
        from openpyxl import Workbook
    except Exception as exc:
        print(f"⚠️ Excel 저장 생략: openpyxl 사용 불가 ({exc})")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"lane_eval_summary_{timestamp}.xlsx"

    wb = Workbook()
    ws = wb.active
    ws.title = "Lane Summary"
    ws.append(headers)
    for row in rows:
        ws.append([row.get(header, "") for header in headers])
    wb.save(path)
    print(f"✅ Excel 저장: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="차선추정 & 차선관리 성능 평가 가로 비교 표 생성")
    parser.add_argument(
        "--lane-eval",
        default=None,
        help="lane_eval_YYYYMMDD_HHMMSS.csv 경로. 생략 시 runtime_data 아래 최신 lane_eval_*.csv 사용",
    )
    parser.add_argument(
        "--lane-gt",
        default=None,
        help="lane_gt_final.csv 경로. 생략 시 runtime_data 아래 lane_gt_final.csv 자동 탐색",
    )
    parser.add_argument(
        "--csv-output-dir",
        default=str(EVAL_RESULTS_DIR),
        help="평가 결과 CSV 저장 폴더",
    )
    parser.add_argument(
        "--excel-output-dir",
        default=str(EVAL_SUMMARIES_DIR),
        help="PPT/보고서용 Excel 요약표 저장 폴더",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lane_eval_path = Path(args.lane_eval) if args.lane_eval else find_latest("lane_eval_20260501_133407.csv")
    lane_gt_path = Path(args.lane_gt) if args.lane_gt else find_gt_file()

    if lane_eval_path is None or not lane_eval_path.exists():
        print("❌ lane_eval_YYYYMMDD_HHMMSS.csv를 찾지 못했습니다.")
        print("   --lane-eval 경로를 지정하거나 runtime_data 아래에 lane_eval_*.csv를 두세요.")
        return

    print("📂 Lane eval CSV:", lane_eval_path)
    pred_rows = read_csv_rows(lane_eval_path)

    gt_rows = []
    if lane_gt_path and lane_gt_path.exists():
        print("📂 Lane GT CSV:", lane_gt_path)
        gt_rows = read_csv_rows(lane_gt_path)
    else:
        print("⚠️ lane_gt_final.csv를 찾지 못했습니다. expected_lane_count 기준으로 GT 분포를 보완합니다.")

    headers = ["구분", "항목"] + [target["name"] for target in TARGET_VIDEOS]
    rows = build_wide_rows(pred_rows, gt_rows)

    print_markdown_table(rows, headers)
    save_csv(rows, headers, Path(args.csv_output_dir))
    save_excel_optional(rows, headers, Path(args.excel_output_dir))


if __name__ == "__main__":
    main()
