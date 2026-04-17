# ==========================================
# 파일명: state_detail_long_logger_V5_4.py
# 설명:
# - tunnel_main_V5_3.py는 건드리지 않음
# - 차량 1대당 1줄(long format) 상세 로그 저장
# - V5_4 상태로직용 분석 로그
# - 목적:
#   차량 ID별 dy / bbox_height / 정규화 속도 분석
#   임계값 / 보정계수 / EMA 튜닝
# ==========================================

import os
import sys
import csv
import traceback
from datetime import datetime

import cv2
from ultralytics import YOLO

# =========================================================
# 경로 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
PARENT_DIR = os.path.dirname(PROJECT_ROOT)

PIPELINE_DIR = os.path.join(PROJECT_ROOT, "scr", "pipeline_V5_4(상태로직_합격)")
if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

from pipeline_core_V5_3 import PipelineCore

print("🚀 STATE DETAIL LONG LOGGER V5_4 시작")
print("BASE_DIR:", BASE_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("PARENT_DIR:", PARENT_DIR)
print("PIPELINE_DIR:", PIPELINE_DIR)
print("PIPELINE_DIR exists:", os.path.exists(PIPELINE_DIR))

# ---------------------------------------------------------
# 테스트 영상 선택
# ---------------------------------------------------------
VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_congestion_2-1.mp4"
# VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_normal_2.mp4"
# VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_accident_1-1.mp4"

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")

# ---------------------------------------------------------
# 출력 경로
# ---------------------------------------------------------
OUTPUT_DIR = os.path.join(PARENT_DIR, "smart_tunnel_V3_outputs", "analysis_v5_4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DETAIL_LONG_LOG_PATH = os.path.join(
    OUTPUT_DIR,
    f"state_detail_long_log_v5_4_{timestamp}.csv"
)

print("VIDEO_PATH:", VIDEO_PATH)
print("영상 존재 여부:", os.path.exists(VIDEO_PATH))
print("MODEL_PATH:", MODEL_PATH)
print("모델 존재 여부:", os.path.exists(MODEL_PATH))
print("DETAIL_LONG_LOG_PATH:", DETAIL_LONG_LOG_PATH)


# =========================================================
# CSV 헤더
# =========================================================
def write_detail_long_header(writer):
    writer.writerow([
        "frame_id",
        "track_id",

        "dy",
        "bbox_height",
        "raw_speed",
        "norm_speed",
        "corrected_speed",
        "ema_speed",

        "frame_avg_speed",
        "buffer_avg_speed",
        "final_speed",
        "state_speed",

        "vehicle_count",
        "lane_count",
        "roi_y1",
        "roi_y2",
        "roi_span",
        "roi_fixed",
        "template_phase",
        "template_confirmed",

        "state",
        "accident",
    ])


# =========================================================
# 프레임 결과를 차량 1대당 1줄로 기록
# =========================================================
def write_detail_long_rows(writer, frame_id, result):
    merged = result.get("analysis", {})
    state_result = result.get("state", {})
    accident_result = result.get("accident", {})

    # ---------------------------------------------
    # state_result 안전 처리
    # ---------------------------------------------
    if isinstance(state_result, dict):
        state_text = state_result.get("state", "UNKNOWN")
        state_debug = state_result.get("debug", {})
    else:
        state_text = str(state_result)
        state_debug = {}

    accident_text = str(accident_result)

    # ---------------------------------------------
    # 디버그 정보 추출
    # ---------------------------------------------
    vehicle_ids = state_debug.get("vehicle_ids", [])
    dy_dict = state_debug.get("vehicle_dy", {})
    bbox_height_dict = state_debug.get("vehicle_bbox_height", {})
    raw_dict = state_debug.get("vehicle_speeds_raw", {})
    norm_dict = state_debug.get("vehicle_speeds_norm", {})
    corrected_dict = state_debug.get("vehicle_speeds_corrected", {})
    ema_dict = state_debug.get("vehicle_speeds_ema", {})

    frame_avg_speed = state_debug.get("frame_avg_speed", 0.0)
    buffer_avg_speed = state_debug.get("buffer_avg_speed", 0.0)
    final_speed = state_debug.get("final_speed", 0.0)
    state_speed = state_debug.get("state_speed", 0.0)

    # ---------------------------------------------
    # 프레임 공통 정보
    # ---------------------------------------------
    vehicle_count = merged.get("vehicle_count", 0)
    lane_count = merged.get("lane_count", 0)
    roi_y1 = merged.get("roi_y1", 0)
    roi_y2 = merged.get("roi_y2", 0)
    roi_span = merged.get("roi_span", 0)
    roi_fixed = merged.get("roi_fixed", False)
    template_phase = merged.get("template_phase", "")
    template_confirmed = merged.get("template_confirmed", False)

    # ---------------------------------------------
    # 차량이 없는 프레임도 기록
    # ---------------------------------------------
    if not vehicle_ids:
        writer.writerow([
            frame_id,
            "",

            "",   # dy
            "",   # bbox_height
            "",   # raw_speed
            "",   # norm_speed
            "",   # corrected_speed
            "",   # ema_speed

            frame_avg_speed,
            buffer_avg_speed,
            final_speed,
            state_speed,

            vehicle_count,
            lane_count,
            roi_y1,
            roi_y2,
            roi_span,
            roi_fixed,
            template_phase,
            template_confirmed,

            state_text,
            accident_text,
        ])
        return

    # ---------------------------------------------
    # 차량 1대당 1줄 기록
    # ---------------------------------------------
    for tid in vehicle_ids:
        dy = dy_dict.get(tid, "")
        bbox_height = bbox_height_dict.get(tid, "")
        raw_speed = raw_dict.get(tid, "")
        norm_speed = norm_dict.get(tid, "")
        corrected_speed = corrected_dict.get(tid, "")
        ema_speed = ema_dict.get(tid, "")

        writer.writerow([
            frame_id,
            tid,

            dy,
            bbox_height,
            raw_speed,
            norm_speed,
            corrected_speed,
            ema_speed,

            frame_avg_speed,
            buffer_avg_speed,
            final_speed,
            state_speed,

            vehicle_count,
            lane_count,
            roi_y1,
            roi_y2,
            roi_span,
            roi_fixed,
            template_phase,
            template_confirmed,

            state_text,
            accident_text,
        ])


# =========================================================
# 메인 실행
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

    print("✅ long format 상세 로그 추출 시작")
    print("   └ 저장 로그:", DETAIL_LONG_LOG_PATH)

    with open(DETAIL_LONG_LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        write_detail_long_header(writer)

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
            write_detail_long_rows(writer, frame_id, result)

            if frame_id % 100 == 0:
                print(f"   └ frame {frame_id} 처리 완료")

    cap.release()

    print("✅ long format 상세 로그 저장 완료")
    print("   └ 로그:", DETAIL_LONG_LOG_PATH)
    print("   └ 폴더:", OUTPUT_DIR)


# =========================================================
# 엔트리포인트
# =========================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ 실행 중 오류:", e)
        traceback.print_exc()