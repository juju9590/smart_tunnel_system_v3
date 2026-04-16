# ==========================================
# 파일명: tunnel_main_V5_2.py
# 설명:
# V5_2 메인 실행 파일
# - 영상 입력
# - YOLO + ByteTrack 추적
# - PipelineCore(V5_2) 호출
# - 결과 시각화
# - 결과 영상 저장
# - 로그 CSV 저장
# ==========================================

import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO

from pipeline_core_V5_2 import PipelineCore

print("🚀 SMART TUNNEL V5_2 시작")


# ==========================
# 현재 파일 기준 절대경로
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# ==========================
# 외부 경로 (대용량 파일 분리)
# ==========================
DATA_ROOT = "E:/Finalpj_tunnel_V3/smart_tunnel_V3_data"
OUTPUT_ROOT = "E:/Finalpj_tunnel_V3/smart_tunnel_V3_outputs"

# ==========================
# 입력 영상 경로
# ==========================
# VIDEO_PATH = os.path.join(DATA_ROOT, "raw_video", "test_video", "test_normal_1.mp4")
# VIDEO_PATH = os.path.join(DATA_ROOT, "raw_video", "test_video", "test_normal_2.mp4")

VIDEO_PATH = os.path.join(DATA_ROOT, "raw_video", "test_video", "test_congestion_2-1.mp4")
# VIDEO_PATH = os.path.join(DATA_ROOT, "raw_video", "test_video", "test_congestion_2-2.mp4")

# VIDEO_PATH = os.path.join(DATA_ROOT, "raw_video", "test_video", "test_accident_1-1.mp4")
# VIDEO_PATH = os.path.join(DATA_ROOT, "raw_video", "test_video", "test_accident_2.mp4")
# VIDEO_PATH = os.path.join(DATA_ROOT, "raw_video", "test_video", "test_accident_3.mp4")

# ==========================
# 모델 경로
# ==========================
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best.pt"
)

# ==========================
# 출력 경로
# ==========================
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "pipeline_v5_2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, f"v5_2_{timestamp}.mp4")
LOG_PATH = os.path.join(OUTPUT_DIR, f"log_v5_2_{timestamp}.csv")

print("BASE_DIR:", BASE_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("VIDEO_PATH:", VIDEO_PATH)
print("👉 EXISTS:", os.path.exists(VIDEO_PATH))
print("영상 존재 여부:", os.path.exists(VIDEO_PATH))
print("MODEL_PATH:", MODEL_PATH)
print("모델 존재 여부:", os.path.exists(MODEL_PATH))
print("VIDEO_OUT_PATH:", VIDEO_OUT_PATH)
print("LOG_PATH:", LOG_PATH)


def draw_track_boxes(frame, tracks, lane_map, state, accident):
    """
    차량 박스 / ID / 차선 표시
    """
    for t in tracks:
        tid = t["id"]
        x1, y1, x2, y2 = t["bbox"]

        lane = lane_map.get(tid, None)
        lane_text = "?" if lane is None else str(lane)

        if accident:
            color = (0, 0, 255)
        elif state == "JAM":
            color = (255, 0, 0)
        elif state == "CONGESTION":
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            f"ID:{tid} L:{lane_text}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2
        )


def draw_top_ui(frame, result):
    """
    상단 상태 정보 표시
    """
    state = result.get("state", "NORMAL")
    avg_speed = float(result.get("avg_speed", 0.0))
    accident = bool(result.get("accident", False))
    acc_ratio = float(result.get("acc_ratio", 0.0))
    lane_count = int(result.get("lane_count", 0))
    vehicle_count = int(result.get("vehicle_count", 0))

    h, w, _ = frame.shape

    cv2.putText(
        frame, f"STATE: {state}", (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3
    )

    cv2.putText(
        frame, f"AVG_SPEED: {avg_speed:.2f}", (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2
    )

    cv2.putText(
        frame, f"VEHICLES: {vehicle_count}", (30, 135),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
    )

    cv2.putText(
        frame, f"LANE_COUNT: {lane_count}", (30, 175),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 0), 2
    )

    cv2.putText(
        frame, f"ACC_RATIO: {acc_ratio:.2f}", (30, 215),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 0), 2
    )

    if accident:
        cv2.putText(
            frame,
            "!!! ACCIDENT !!!",
            (w // 3, h // 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            4
        )


def create_writer(cap, video_out_path):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        video_out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )
    return out, fps, width, height


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ 영상 열기 실패")
        return

    out, fps, width, height = create_writer(cap, VIDEO_OUT_PATH)

    if not out.isOpened():
        print("❌ VideoWriter 생성 실패")
        cap.release()
        return

    if not os.path.exists(MODEL_PATH):
        print("❌ 모델 파일 없음:", MODEL_PATH)
        cap.release()
        out.release()
        return

    model = YOLO(MODEL_PATH)
    pipeline = PipelineCore()

    log_file = open(LOG_PATH, "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(log_file)

    writer.writerow([
        "frame",
        "vehicle_count",
        "avg_speed",
        "state",
        "accident",
        "acc_ratio",
        "lane_count",
        "lane_map",
        "accident_pairs"
    ])

    frame_id = 0

    print("✅ 분석 시작")
    print("   └ 저장 영상:", VIDEO_OUT_PATH)
    print("   └ 저장 로그:", LOG_PATH)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ℹ️ 영상 끝")
                break

            frame_id += 1

            results = model.track(
                frame,
                persist=True,
                conf=0.25,
                iou=0.5,
                tracker="bytetrack.yaml",
                verbose=False
            )

            tracks = []

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()

                for box, tid in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    tid = int(tid)

                    tracks.append({
                        "id": tid,
                        "bbox": (x1, y1, x2, y2)
                    })

            result = pipeline.process(frame_id, tracks)

            state = result.get("state", "NORMAL")
            accident = result.get("accident", False)
            lane_map = result.get("lane_map", {})
            accident_debug = result.get("accident_debug", {})
            accident_pairs = accident_debug.get("pairs", [])

            draw_track_boxes(frame, tracks, lane_map, state, accident)
            draw_top_ui(frame, result)

            out.write(frame)

            writer.writerow([
                frame_id,
                result.get("vehicle_count", 0),
                round(float(result.get("avg_speed", 0.0)), 2),
                result.get("state", "NORMAL"),
                int(bool(result.get("accident", False))),
                round(float(result.get("acc_ratio", 0.0)), 4),
                int(result.get("lane_count", 0)),
                str(result.get("lane_map", {})),
                str(accident_pairs)
            ])

            cv2.imshow("SMART TUNNEL V5_2", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                print("🛑 ESC 종료")
                break

    except Exception as e:
        print("❌ 실행 중 오류:", e)

    finally:
        log_file.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print("✅ 종료 완료")
        print("   └ 영상:", VIDEO_OUT_PATH)
        print("   └ 로그:", LOG_PATH)
        print("   └ 폴더:", OUTPUT_DIR)


if __name__ == "__main__":
    main()