# ==========================================
# tunnel_main_V4_FINAL.py
# V4 완전 정리 (영상저장 포함)
# ==========================================

import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from pipeline_core_V4 import PipelineCore

# ==========================
# 설정
# ==========================
VIDEO_PATH = "../../data/raw_video/test_video/test_accident_1.mp4"

OUTPUT_DIR = "../../outputs/pipeline_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

video_name = os.path.basename(VIDEO_PATH).lower()

if "accident" in video_name:
    state_name = "accident"
elif "congestion" in video_name:
    state_name = "congestion"
elif "normal" in video_name:
    state_name = "normal"
else:
    state_name = "unknown"

VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, f"{state_name}_{timestamp}.mp4")
LOG_PATH = os.path.join(OUTPUT_DIR, f"{state_name}_{timestamp}.csv")


def get_lane(cx, frame_width):
    return 1 if cx < frame_width // 2 else 2


def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ 영상 열기 실패")
        return

    # -----------------------------
    # 🎥 영상 저장 초기화 (핵심)
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

    model = YOLO("../../scripts/runs/train/tunnel_final/weights/best.pt")

    pipeline = PipelineCore()

    frame_id = 0

    log_file = open(LOG_PATH, "w", newline="")
    writer = csv.writer(log_file)

    writer.writerow([
        "frame","vehicle_count","avg_speed",
        "state","accident","vehicle_ids"
    ])

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
            persist=True,
            conf=0.25,
            iou=0.5,
            tracker="bytetrack.yaml"
        )

        tracks = []
        vehicle_ids = []

        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue

            for box, tid in zip(r.boxes.xyxy, r.boxes.id):

                x1, y1, x2, y2 = map(int, box)
                tid = int(tid)

                cx = int((x1 + x2) / 2)

                tracks.append({
                    "id": tid,
                    "bbox": (x1, y1, x2, y2)
                })

                vehicle_ids.append(tid)

        # -----------------------------
        # 파이프라인 실행
        # -----------------------------
        result = pipeline.process(frame_id, tracks)

        state = result["state"]
        accident = result["accident"]
        avg_speed = result["avg_speed"]

        # -----------------------------
        # 시각화
        # -----------------------------
        for t in tracks:
            tid = t["id"]
            x1, y1, x2, y2 = t["bbox"]

            speed = 0
            if tid in pipeline.state_model.track_history:
                track = pipeline.state_model.track_history[tid]
                if len(track) >= 2:
                    speed = abs(track[-1][1] - track[-2][1])

            lane = get_lane(int((x1+x2)/2), w)

            color = (0,255,0)

            if accident:
                color = (0,0,255)
            elif state == "CONGESTION":
                color = (0,255,255)
            elif state == "JAM":
                color = (255,0,0)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            cv2.putText(frame,
                f"ID:{tid} S:{int(speed)} L:{lane}",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,0),
                2
            )

        # 상태 표시
        cv2.putText(frame, f"STATE: {state}", (30,50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        cv2.putText(frame, f"AVG: {avg_speed:.2f}", (30,100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        if accident:
            cv2.putText(frame, "!!! ACCIDENT !!!", (300,200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)

        # -----------------------------
        # 🎥 영상 저장 (핵심)
        # -----------------------------
        out.write(frame)

        # 로그
        writer.writerow([
            frame_id,
            len(tracks),
            round(avg_speed,2),
            state,
            accident,
            vehicle_ids
        ])

        cv2.imshow("Smart Tunnel", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 종료
    log_file.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()