# ==========================================
# 파일명: tunnel_main_V5_3.py
# 설명:
# SMART TUNNEL V5_3 실행 파일
#
# 역할
# 1) 테스트 영상을 읽는다
# 2) YOLO + ByteTrack으로 차량을 추적한다
# 3) PipelineCore(V5_3)에 tracks를 넘겨서
#    - Adaptive ROI
#    - Track Analyzer
#    - Lane Template
#    - Traffic State
#    - Accident Detector
#    를 순서대로 수행한다
# 4) 결과를 화면에 시각화하고
# 5) 결과 영상 / CSV 로그를 저장한다
#
# [V5_3 핵심]
# - ROI 자동설정(초기 100프레임 bootstrap 후 고정)
# - ROI 안에서 차선 bootstrap
# - 1차 군집 + 2차 군집 기반 대표 차선 추정
# - 상태 / 사고 로직과 공통 분석 결과 통합
# ==========================================

import os
import csv
import cv2
import traceback
from datetime import datetime
from ultralytics import YOLO

from pipeline_core_V5_3 import PipelineCore


print("🚀 SMART TUNNEL V5_3 시작")

# =========================================================
# 1) 기본 경로 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))

# ---------------------------------------------------------
# 테스트 영상 선택
# 필요한 영상만 주석 해제해서 사용
# ---------------------------------------------------------
VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_congestion_2-1.mp4"
# VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_normal_2.mp4"
# VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_accident_1-1.mp4"

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")

# ---------------------------------------------------------
# 출력 경로
# ---------------------------------------------------------
OUTPUT_DIR = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_outputs/pipeline_v5_3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, f"v5_3_{timestamp}.mp4")
LOG_PATH = os.path.join(OUTPUT_DIR, f"log_v5_3_{timestamp}.csv")

print("BASE_DIR:", BASE_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("VIDEO_PATH:", VIDEO_PATH)
print("👉 EXISTS:", os.path.exists(VIDEO_PATH))
print("영상 존재 여부:", os.path.exists(VIDEO_PATH))
print("MODEL_PATH:", MODEL_PATH)
print("모델 존재 여부:", os.path.exists(MODEL_PATH))
print("VIDEO_OUT_PATH:", VIDEO_OUT_PATH)
print("LOG_PATH:", LOG_PATH)


# =========================================================
# 2) 시각화 함수
# =========================================================
def draw_centerlines(frame, centerlines, roi_y1, roi_y2):
    """
    대표 차선(centerlines)을 화면에 그린다.

    centerlines 구조 예시:
    [
        {
            "lane_id": 0,
            "rep_model": {"type":"linear", "coef":[a,b]},
            ...
        },
        ...
    ]
    """
    for lane in centerlines:
        lane_id = lane["lane_id"]
        model = lane["rep_model"]

        pts = []
        for y in range(int(roi_y1), int(roi_y2), 20):
            # 현재 V5_3에서는 선형(linear) 모델을 주로 사용
            if model["type"] == "linear":
                a, b = model["coef"]
                x = a * y + b
            else:
                # 혹시 확장 버전에서 quadratic이 들어오더라도 안전하게 처리
                a, b, c = model["coef"]
                x = a * (y ** 2) + b * y + c

            pts.append((int(x), int(y)))

        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (255, 255, 0), 2)

        if pts:
            cv2.putText(
                frame,
                f"LANE {lane_id}",
                pts[len(pts) // 2],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )


def draw_tracks(frame, tracks, merged_analysis):
    """
    차량 bbox / 속도 / raw lane / lane 결과를 그린다.
    """
    boxes = merged_analysis.get("boxes", {})
    speeds = merged_analysis.get("speeds", {})
    lane_map = merged_analysis.get("lane_map", {})
    raw_lane_map = merged_analysis.get("raw_lane_map", {})

    for t in tracks:
        tid = t["id"]
        if tid not in boxes:
            continue

        x1, y1, x2, y2 = boxes[tid]
        speed = speeds.get(tid, 0.0)
        raw_lane = raw_lane_map.get(tid, None)
        lane = lane_map.get(tid, None)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        text = f"ID:{tid} V:{speed:.1f} raw:{raw_lane} lane:{lane}"
        cv2.putText(
            frame,
            text,
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


def draw_roi_lines(frame, merged_analysis):
    """
    현재 ROI를 화면에 표시
    - final ROI: 파란색
    """
    h, w = frame.shape[:2]
    roi_y1 = int(merged_analysis.get("roi_y1", int(h * 0.2)))
    roi_y2 = int(merged_analysis.get("roi_y2", int(h * 0.8)))

    cv2.line(frame, (0, roi_y1), (w, roi_y1), (255, 0, 0), 2)
    cv2.line(frame, (0, roi_y2), (w, roi_y2), (255, 0, 0), 2)


def draw_summary(frame, result):
    """
    화면 왼쪽 상단에 핵심 요약 정보 출력
    """
    merged = result["analysis"]
    state_result = result["state"]
    accident_result = result["accident"]

    avg_speed = merged.get("avg_speed", 0.0)
    vehicle_count = merged.get("vehicle_count", 0)
    lane_count = merged.get("lane_count", 0)

    roi_fixed = merged.get("roi_fixed", False)
    roi_span = merged.get("roi_span", 0)
    roi_sample_count = merged.get("roi_sample_count", 0)

    template_phase = merged.get("template_phase", "-")
    template_confirmed = merged.get("template_confirmed", False)

    state_text = str(state_result)
    accident_text = str(accident_result)

    y = 30
    lines = [
        f"Vehicles: {vehicle_count}",
        f"Avg Speed: {avg_speed:.2f}",
        f"Lane Count: {lane_count}",
        f"ROI Fixed: {roi_fixed}",
        f"ROI Span: {roi_span}",
        f"ROI Samples: {roi_sample_count}",
        f"Template Phase: {template_phase}",
        f"Template Confirmed: {template_confirmed}",
        f"State: {state_text}",
        f"Accident: {accident_text}",
    ]

    for line in lines:
        cv2.putText(
            frame,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        y += 24


# =========================================================
# 3) CSV 로그 함수
# =========================================================
def write_log_header(writer):
    """
    CSV 헤더 작성
    """
    writer.writerow([
        "frame_id",
        "vehicle_count",
        "avg_speed",
        "lane_count",
        "roi_y1",
        "roi_y2",
        "roi_span",
        "roi_fixed",
        "roi_sample_count",
        "template_phase",
        "template_confirmed",
        "state",
        "accident",
    ])


def write_log_row(writer, frame_id, result):
    """
    한 프레임 결과를 CSV에 기록
    """
    merged = result["analysis"]

    writer.writerow([
        frame_id,
        merged.get("vehicle_count", 0),
        merged.get("avg_speed", 0.0),
        merged.get("lane_count", 0),
        merged.get("roi_y1", 0),
        merged.get("roi_y2", 0),
        merged.get("roi_span", 0),
        merged.get("roi_fixed", False),
        merged.get("roi_sample_count", 0),
        merged.get("template_phase", ""),
        merged.get("template_confirmed", False),
        str(result["state"]),
        str(result["accident"]),
    ])


# =========================================================
# 4) 메인 실행 함수
# =========================================================
def main():
    # -----------------------------------------------------
    # 영상 열기
    # -----------------------------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ 영상 열기 실패")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    # -----------------------------------------------------
    # 저장용 VideoWriter
    # -----------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps, (width, height))

    # -----------------------------------------------------
    # 모델 / 파이프라인 생성
    # -----------------------------------------------------
    model = YOLO(MODEL_PATH)

    pipeline = PipelineCore(
        frame_height=height,
        lane_output_dir=OUTPUT_DIR
    )

    print("✅ 분석 시작")
    print("   └ 저장 영상:", VIDEO_OUT_PATH)
    print("   └ 저장 로그:", LOG_PATH)

    with open(LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        write_log_header(writer)

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # -------------------------------------------------
            # YOLO + ByteTrack
            # classes=[0] 은 현재 학습 모델 기준 단일 차량 클래스 사용 가정
            # -------------------------------------------------
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

            # -------------------------------------------------
            # 파이프라인 처리
            # -------------------------------------------------
            result = pipeline.process(frame_id, tracks)
            merged = result["analysis"]

            # -------------------------------------------------
            # 시각화
            # -------------------------------------------------
            draw_roi_lines(frame, merged)
            draw_centerlines(
                frame,
                merged.get("centerlines", []),
                merged.get("roi_y1", int(height * 0.2)),
                merged.get("roi_y2", int(height * 0.8))
            )
            draw_tracks(frame, tracks, merged)
            draw_summary(frame, result)

            # -------------------------------------------------
            # 저장 / 로그
            # -------------------------------------------------
            out.write(frame)
            write_log_row(writer, frame_id, result)

            # -------------------------------------------------
            # 화면 출력
            # ESC 누르면 종료
            # -------------------------------------------------
            cv2.imshow("SMART TUNNEL V5_3", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("✅ 종료 완료")
    print("   └ 영상:", VIDEO_OUT_PATH)
    print("   └ 로그:", LOG_PATH)
    print("   └ 폴더:", OUTPUT_DIR)


# =========================================================
# 5) 엔트리포인트
# =========================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ 실행 중 오류:", e)
        traceback.print_exc()