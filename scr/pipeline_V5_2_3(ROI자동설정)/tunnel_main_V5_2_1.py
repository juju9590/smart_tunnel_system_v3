# ==========================================
# 파일명: tunnel_main_V5_2_1.py
# 설명:
# V5_2_1 실행 파일
# - YOLO + ByteTrack
# - PipelineCore V5_2_1 연결
# - 결과 영상 / CSV 로그 저장
#
# 주의:
# - traffic_state_V5_1, traffic_accident_V5_2_3의 입력 형식은
#   기존 merged_analysis dict를 받는다고 가정
# ==========================================

import os
import csv
import cv2
import traceback
from datetime import datetime
from ultralytics import YOLO

from pipeline_core_V5_2_2 import PipelineCore


print("🚀 SMART TUNNEL V5_2_1 시작")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))

# ------------------------------------------
# 경로 설정
# ------------------------------------------
# VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_congestion_2-1.mp4"
# VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_normal_2.mp4"
VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_accident_1-1.mp4"

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")

OUTPUT_DIR = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_outputs/pipeline_v5_2_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, f"v5_2_1_{timestamp}.mp4")
LOG_PATH = os.path.join(OUTPUT_DIR, f"log_v5_2_1_{timestamp}.csv")

print("BASE_DIR:", BASE_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("VIDEO_PATH:", VIDEO_PATH)
print("👉 EXISTS:", os.path.exists(VIDEO_PATH))
print("영상 존재 여부:", os.path.exists(VIDEO_PATH))
print("MODEL_PATH:", MODEL_PATH)
print("모델 존재 여부:", os.path.exists(MODEL_PATH))
print("VIDEO_OUT_PATH:", VIDEO_OUT_PATH)
print("LOG_PATH:", LOG_PATH)


def draw_centerlines(frame, centerlines, roi_y1, roi_y2):
    """
    대표 차선(centerlines) 시각화
    """
    for lane in centerlines:
        lane_id = lane["lane_id"]
        model = lane["rep_model"]

        pts = []
        for y in range(int(roi_y1), int(roi_y2), 20):
            if model["type"] == "linear":
                a, b = model["coef"]
                x = a * y + b
            else:
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


def draw_summary(frame, result):
    merged = result["analysis"]
    state_result = result["state"]
    accident_result = result["accident"]

    avg_speed = merged.get("avg_speed", 0.0)
    vehicle_count = merged.get("vehicle_count", 0)
    lane_count = merged.get("lane_count", 0)
    template_phase = merged.get("template_phase", "-")
    template_confirmed = merged.get("template_confirmed", False)
    freeze_until = merged.get("freeze_until", -1)
    template_update = merged.get("template_update", False)
    change_score = merged.get("change_score", 0.0)

    state_text = str(state_result)
    accident_text = str(accident_result)

    y = 30
    lines = [
        f"Vehicles: {vehicle_count}",
        f"Avg Speed: {avg_speed:.2f}",
        f"Lane Count: {lane_count}",
        f"Template Phase: {template_phase}",
        f"Template Confirmed: {template_confirmed}",
        f"Freeze Until: {freeze_until}",
        f"Template Update: {template_update}",
        f"Change Score: {change_score:.2f}",
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


def write_log_header(writer):
    writer.writerow([
        "frame_id",
        "vehicle_count",
        "avg_speed",
        "lane_count",
        "template_phase",
        "template_confirmed",
        "freeze_until",
        "template_update",
        "change_score",
        "state",
        "accident",
    ])


def write_log_row(writer, frame_id, result):
    merged = result["analysis"]

    writer.writerow([
        frame_id,
        merged.get("vehicle_count", 0),
        merged.get("avg_speed", 0.0),
        merged.get("lane_count", 0),
        merged.get("template_phase", ""),
        merged.get("template_confirmed", False),
        merged.get("freeze_until", -1),
        merged.get("template_update", False),
        merged.get("change_score", 0.0),
        str(result["state"]),
        str(result["accident"]),
    ])


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ 영상 열기 실패")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps, (width, height))

    model = YOLO(MODEL_PATH)
    pipeline = PipelineCore()

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

            # YOLO + ByteTrack
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

            # 파이프라인 처리
            result = pipeline.process(frame_id, tracks)
            merged = result["analysis"]

            # 시각화
            roi_y1 = merged.get("roi_y1", int(height * 0.3))
            roi_y2 = merged.get("roi_y2", int(height * 0.8))

            cv2.line(frame, (0, int(roi_y1)), (width, int(roi_y1)), (255, 0, 0), 1)
            cv2.line(frame, (0, int(roi_y2)), (width, int(roi_y2)), (255, 0, 0), 1)

            draw_centerlines(frame, merged.get("centerlines", []), roi_y1, roi_y2)
            draw_tracks(frame, tracks, merged)
            draw_summary(frame, result)

            out.write(frame)
            write_log_row(writer, frame_id, result)

            cv2.imshow("SMART TUNNEL V5_2_1", frame)
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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ 실행 중 오류:", e)
        traceback.print_exc()