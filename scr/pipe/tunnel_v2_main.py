# ==========================================
# tunnel_v2_main.py
# 실행 + 시각화 + 로그 (FULL DEBUG)
# ==========================================

import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from scr.pipe.pipeline_core import PipelineCore

# ==========================
# 설정
# ==========================
# VIDEO_PATH = "../../data/raw_video/test_video/test_congestion_1.mp4" #혼잡
VIDEO_PATH = "../../data/raw_video/test_video/test_accident_1.mp4" #사고

# 로그 폴더
LOG_DIR = "../../outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(LOG_DIR, f"log_{timestamp}.csv")


# ==========================
# 차선 추정 함수 (간단 버전)
# ==========================
def get_lane(cx, frame_width):
    """
    화면을 좌/우로 나눠서 차선 추정
    (간단 버전 → 나중에 ROI 기반으로 개선 가능)
    """
    if cx < frame_width // 2:
        return 1
    else:
        return 2


def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ 영상 열기 실패")
        return

    # YOLO 모델 (학습된 모델 사용 가능)
    model = YOLO("../../scripts/runs/train/tunnel_final/weights/best.pt")

    pipeline = PipelineCore()

    frame_id = 0

    # -----------------------------
    # 로그 파일 생성
    # -----------------------------
    log_file = open(LOG_PATH, "w", newline="")
    writer = csv.writer(log_file)

    writer.writerow([
        "frame",
        "vehicle_count",
        "avg_speed",
        "state",
        "accident",
        "vehicle_ids",
        "vehicle_speeds",
        "lanes"
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
        results = model.track(frame, persist=True)

        tracks = []

        vehicle_ids = []
        vehicle_speeds = []
        lanes = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                if box.id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tid = int(box.id[0])

                cx = int((x1 + x2) / 2)
                cy = y2

                # -----------------------------
                # 속도 계산 (간단)
                # -----------------------------
                speed = 0
                if tid in pipeline.state_model.track_history:
                    track = pipeline.state_model.track_history[tid]
                    if len(track) >= 2:
                        speed = abs(track[-1][1] - track[-2][1])

                # -----------------------------
                # 차선 추정
                # -----------------------------
                lane = get_lane(cx, w)

                tracks.append({
                    "id": tid,
                    "bbox": (x1, y1, x2, y2)
                })

                vehicle_ids.append(tid)
                vehicle_speeds.append(speed)
                lanes.append(lane)

                # -----------------------------
                # 바운딩 박스 표시
                # -----------------------------
                label = f"ID:{tid} S:{int(speed)} L:{lane}"

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        # -----------------------------
        # 파이프라인 처리
        # -----------------------------
        result = pipeline.process(frame_id, tracks)

        state = result["state"]
        accident = result["accident"]
        avg_speed = result["avg_speed"]

        print(f"[DEBUG] state:{state} accident:{accident}")

        # -----------------------------
        # 화면 표시
        # -----------------------------
        cv2.putText(frame, f"STATE: {state}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        cv2.putText(frame, f"AVG SPEED: {avg_speed:.2f}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        # event_text = "ACCIDENT" if accident else "NONE"
        # 대신 아래 코드로 대체

        if accident:
            event_text = "ACCIDENT"
        elif state == "JAM":
            event_text = "None"
        elif state == "CONGESTION":
            event_text = "None"
        else:
            event_text = "None"


        cv2.putText(frame, f"EVENT: {event_text}", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        if accident:
            cv2.putText(frame, "!!! ACCIDENT !!!", (300, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)

        # -----------------------------
        # 로그 저장 (FULL)
        # -----------------------------
        writer.writerow([
            frame_id,
            len(tracks),
            round(avg_speed,2),
            state,
            accident,
            vehicle_ids,
            vehicle_speeds,
            lanes
        ])

        # -----------------------------
        # 화면 출력
        # -----------------------------
        cv2.imshow("Smart Tunnel", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 종료 처리
    log_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()