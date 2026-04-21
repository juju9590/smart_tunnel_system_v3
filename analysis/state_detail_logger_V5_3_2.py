# ==========================================
# 파일명: state_detail_logger_V5_3_2.py
# 설명:
# - tunnel_main_V5_3.py는 건드리지 않고
# - 상태 로직 상세 로그만 별도 CSV로 저장하는 전용 실행 파일
# - 저장 항목:
#   frame_id
#   vehicle_ids
#   vehicle_speeds_raw
#   vehicle_speeds_corrected
#   vehicle_speeds_ema
#   frame_avg_speed
#   buffer_avg_speed
#   final_speed
#   state
#   accident
# ==========================================

import os
import sys
import csv
import ast
import traceback
from datetime import datetime

import cv2
from ultralytics import YOLO

# =========================================================
# pipeline_V5_3(state_logic) 경로 추가
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

PIPELINE_DIR = os.path.join(PROJECT_ROOT, "scr", "pipeline_V5_3(state_logic)")
if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

from pipeline_core_V5_3 import PipelineCore

print("🚀 STATE DETAIL LOGGER V5_3_2 시작")

print("BASE_DIR:", BASE_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("PIPELINE_DIR:", PIPELINE_DIR)
print("PIPELINE_DIR exists:", os.path.exists(PIPELINE_DIR))


# ---------------------------------------------------------
# 테스트 영상 선택
# ---------------------------------------------------------
# VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_congestion_2-1.mp4"
# VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_normal_2.mp4"
VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_accident_1-1.mp4"

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")

# ---------------------------------------------------------
# 출력 경로
# ---------------------------------------------------------
OUTPUT_DIR = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_outputs/pipeline_v5_3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DETAIL_LOG_PATH = os.path.join(OUTPUT_DIR, f"state_detail_log_v5_3_2_{timestamp}.csv")

print("BASE_DIR:", BASE_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("VIDEO_PATH:", VIDEO_PATH)
print("영상 존재 여부:", os.path.exists(VIDEO_PATH))
print("MODEL_PATH:", MODEL_PATH)
print("모델 존재 여부:", os.path.exists(MODEL_PATH))
print("DETAIL_LOG_PATH:", DETAIL_LOG_PATH)


# =========================================================
# 2) 문자열 안정 변환
# =========================================================
def safe_to_str(value):
    """
    dict/list를 CSV에 안전하게 문자열로 저장
    """
    try:
        return str(value)
    except Exception:
        return ""


# =========================================================
# 3) 상세 CSV 헤더
# =========================================================
def write_detail_log_header(writer):
    writer.writerow([
        "frame_id",
        "vehicle_count",
        "lane_count",
        "roi_y1",
        "roi_y2",
        "roi_span",
        "roi_fixed",
        "template_phase",
        "template_confirmed",

        "vehicle_ids",
        "vehicle_speeds_raw",
        "vehicle_speeds_corrected",
        "vehicle_speeds_ema",

        "frame_avg_speed",
        "buffer_avg_speed",
        "final_speed",

        "state",
        "accident",
    ])


# =========================================================
# 4) 상세 CSV 한 줄 기록
# =========================================================
def write_detail_log_row(writer, frame_id, result):
    merged = result.get("analysis", {})
    state_result = result.get("state", {})
    accident_result = result.get("accident", {})

    # state_result 구조:
    # {"state": "NORMAL/JAM/CONGESTION", "debug": {...}}
    if isinstance(state_result, dict):
        state_text = state_result.get("state", "UNKNOWN")
        state_debug = state_result.get("debug", {})
    else:
        state_text = str(state_result)
        state_debug = {}

    # accident_result 구조 안전 처리
    if isinstance(accident_result, dict):
        accident_text = safe_to_str(accident_result)
    else:
        accident_text = str(accident_result)

    writer.writerow([
        frame_id,
        merged.get("vehicle_count", 0),
        merged.get("lane_count", 0),
        merged.get("roi_y1", 0),
        merged.get("roi_y2", 0),
        merged.get("roi_span", 0),
        merged.get("roi_fixed", False),
        merged.get("template_phase", ""),
        merged.get("template_confirmed", False),

        safe_to_str(state_debug.get("vehicle_ids", [])),
        safe_to_str(state_debug.get("vehicle_speeds_raw", {})),
        safe_to_str(state_debug.get("vehicle_speeds_corrected", {})),
        safe_to_str(state_debug.get("vehicle_speeds_ema", {})),

        state_debug.get("frame_avg_speed", 0.0),
        state_debug.get("buffer_avg_speed", 0.0),
        state_debug.get("final_speed", 0.0),

        state_text,
        accident_text,
    ])


# =========================================================
# 5) 메인 실행
# =========================================================
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ 영상 열기 실패")
        return

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    model = YOLO(MODEL_PATH)
    pipeline = PipelineCore(
        frame_height=height,
        lane_output_dir=None
    )

    print("✅ 상세 로그 추출 시작")
    print("   └ 저장 로그:", DETAIL_LOG_PATH)

    with open(DETAIL_LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        write_detail_log_header(writer)

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            yolo_results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=0.25,
                verbose=False
            )

            tracks = []

            if yolo_results and len(yolo_results) > 0:
                r = yolo_results[0]

                if r.boxes is not None and r.boxes.id is not None:
                    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                    ids = r.boxes.id.cpu().numpy().astype(int)

                    for box, tid in zip(boxes_xyxy, ids):
                        x1, y1, x2, y2 = box.tolist()
                        tracks.append({
                            "id": int(tid),
                            "bbox": (int(x1), int(y1), int(x2), int(y2))
                        })

            result = pipeline.process(frame_id, tracks)
            write_detail_log_row(writer, frame_id, result)

            if frame_id % 100 == 0:
                print(f"   └ frame {frame_id} 처리 완료")

    cap.release()

    print("✅ 상세 로그 저장 완료")
    print("   └ 로그:", DETAIL_LOG_PATH)
    print("   └ 폴더:", OUTPUT_DIR)


# =========================================================
# 6) 엔트리포인트
# =========================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ 실행 중 오류:", e)
        traceback.print_exc()