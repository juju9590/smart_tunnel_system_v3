# ==========================================
# tunnel_main_V5_FINAL.py
# 🔥 최종 완성 버전 (안정화 완료)
# ==========================================

import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from pipeline_core_V5 import PipelineCore

print("🚀 프로그램 시작")

# ==========================
# 현재 파일 기준 절대경로
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "raw_video", "test_video", "test_normal_1.mp4")
# VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "raw_video", "test_video", "test_normal_2.mp4")

# VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "raw_video", "test_video", "test_congestion_2-1.mp4")
# VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "raw_video", "test_video", "test_congestion_2-2.mp4")

# VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "raw_video", "test_video", "test_accident_1-1.mp4")
# VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "raw_video", "test_video", "test_accident_2.mp4")
# VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "raw_video", "test_video", "test_accident_3.mp4")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "pipeline_v5_1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, f"result_{timestamp}.mp4")
LOG_PATH = os.path.join(OUTPUT_DIR, f"log_{timestamp}.csv")

print("📁 현재 파일 위치:", BASE_DIR)
print("📁 프로젝트 루트:", PROJECT_ROOT)
print("🎥 VIDEO_PATH:", VIDEO_PATH)
print("🎥 영상 존재 여부:", os.path.exists(VIDEO_PATH))




print("🚀 프로그램 시작")

import os
print("📁 현재 경로:", os.getcwd())
print("🎥 영상 존재 여부:", os.path.exists(VIDEO_PATH))

OUTPUT_DIR = "../../outputs/pipeline_v5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, f"result_{timestamp}.mp4")
LOG_PATH = os.path.join(OUTPUT_DIR, f"log_{timestamp}.csv")


# ==========================
# 차선 간단 추정
# ==========================
def get_lane(cx, frame_width):
    return 1 if cx < frame_width // 2 else 2


def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ 영상 열기 실패")
        return

    # -----------------------------
    # 🎥 영상 저장 설정
    # -----------------------------
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 20

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        VIDEO_OUT_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    # -----------------------------
    # 모델 로드 (🔥 네 학습 모델)
    # -----------------------------
    model = YOLO("../../scripts/runs/train/tunnel_final/weights/best.pt")

    pipeline = PipelineCore()

    frame_id = 0

    # -----------------------------
    # 로그 파일
    # -----------------------------
    log_file = open(LOG_PATH, "w", newline="")
    writer = csv.writer(log_file)

    writer.writerow([
        "frame", "vehicle_count",
        "avg_speed", "state",
        "accident", "acc_ratio"
    ])

    try:
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            h, w, _ = frame.shape

            # -----------------------------
            # YOLO + ByteTrack
            # -----------------------------
            results = model.track(
                frame,
                persist=False,
                conf=0.25,
                iou=0.5,
                tracker="bytetrack.yaml"
            )

            tracks = []

            # 🔥 안전 처리 (중요)
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

            # -----------------------------
            # 파이프라인 실행
            # -----------------------------
            result = pipeline.process(frame_id, tracks)

            state = result["state"]
            accident = result["accident"]
            avg_speed = result["avg_speed"]
            acc_ratio = result["acc_ratio"]

            # -----------------------------
            # 🎯 시각화
            # -----------------------------
            for t in tracks:

                tid = t["id"]
                x1, y1, x2, y2 = t["bbox"]

                cx = int((x1 + x2) / 2)
                lane = get_lane(cx, w)

                # 🔥 상태 우선순위
                if accident:
                    color = (0, 0, 255)       # 빨강
                elif state == "JAM":
                    color = (255, 0, 0)       # 파랑
                elif state == "CONGESTION":
                    color = (0, 255, 255)     # 노랑
                else:
                    color = (0, 255, 0)       # 초록

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    f"ID:{tid} L:{lane}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )

            # -----------------------------
            # 상태 UI
            # -----------------------------
            cv2.putText(frame, f"STATE: {state}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            cv2.putText(frame, f"AVG: {avg_speed:.2f}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(frame, f"ACC_RATIO: {acc_ratio}", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

            if accident:
                cv2.putText(frame, "!!! ACCIDENT !!!",
                            (w//3, h//3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 0, 255),
                            4)

            # -----------------------------
            # 영상 저장
            # -----------------------------
            out.write(frame)

            # -----------------------------
            # 로그 기록
            # -----------------------------
            writer.writerow([
                frame_id,
                len(tracks),
                round(avg_speed, 2),
                state,
                accident,
                acc_ratio
            ])

            cv2.imshow("SMART TUNNEL V5", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        # -----------------------------
        # 안전 종료 (🔥 중요)
        # -----------------------------
        log_file.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()