# ==========================================
# 🚀 ACCIDENT V2.5 (최종)
# ✔ 거리 변화
# ✔ 속도 변화
# ✔ IOU 충돌 감지
# ✔ 수직 충돌 패턴
# ✔ 충돌 유지시간
# ✔ 로그 기록 (패턴 포함)
# ==========================================

import cv2
import numpy as np
import os
import csv
from datetime import datetime
from ultralytics import YOLO

# ==========================
# 설정
# ==========================
MODEL_PATH = "../../scripts/runs/train/tunnel_final/weights/best.pt"
# VIDEO_PATH = "../../data/raw_video/test_video/test_accident_3.mp4"
VIDEO_PATH = "../../data/raw_video/test_video/test_congestion_1.mp4"

CONF = 0.25
IOU = 0.5
ALPHA = 0.3

# 충돌 유지 프레임 (노이즈 제거용)
ACCIDENT_HOLD = 5

# ==========================
# 출력
# ==========================
OUTPUT_DIR = "../../outputs/accident_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, f"accident_{timestamp}.mp4")
LOG_PATH = os.path.join(OUTPUT_DIR, f"accident_{timestamp}.csv")

# ==========================
# 모델
# ==========================
model = YOLO(MODEL_PATH)

track_history = {}
speed_memory = {}

# 이전 상태 저장
pair_memory = {}
accident_counter = {}

# ==========================
# IOU 계산 함수
# ==========================
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0

# ==========================
# 실행
# ==========================
def run():

    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        VIDEO_OUT_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )

    log_file = open(LOG_PATH, "w", newline="")
    writer = csv.writer(log_file)

    # 🔥 로그 컬럼 (완전 확장)
    writer.writerow([
        "frame", "pair",
        "dist", "speed_gap",
        "iou", "vertical",
        "dist_drop", "gap_up", "after_slow",
        "accident",
        "reason_str"
    ])

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        results = model.track(
            frame,
            persist=True,
            conf=CONF,
            iou=IOU,
            tracker="bytetrack.yaml"
        )

        speeds = {}
        boxes = {}

        # ==========================
        # 차량 추적 + 속도 계산
        # ==========================
        if results[0].boxes is not None and results[0].boxes.id is not None:

            for box, tid in zip(results[0].boxes.xyxy,
                                results[0].boxes.id):

                x1,y1,x2,y2 = map(int, box)
                tid = int(tid)

                cx = int((x1+x2)/2)
                cy = y2

                boxes[tid] = (x1,y1,x2,y2)

                track_history.setdefault(tid, []).append((cx,cy))
                if len(track_history[tid]) > 20:
                    track_history[tid].pop(0)

                # 속도 계산 (y축 이동 기반)
                if len(track_history[tid]) >= 2:
                    (_, yp) = track_history[tid][-2]
                    (_, yc) = track_history[tid][-1]

                    dy = yc - yp
                    speed = abs(dy)

                    # EMA smoothing
                    if tid in speed_memory:
                        speed = ALPHA*speed + (1-ALPHA)*speed_memory[tid]

                    speed_memory[tid] = speed
                    speeds[tid] = speed

        # ==========================
        # 차량 간 비교
        # ==========================
        ids = list(boxes.keys())

        for i in range(len(ids)):
            for j in range(i+1, len(ids)):

                id1 = ids[i]
                id2 = ids[j]

                box1 = boxes[id1]
                box2 = boxes[id2]

                x1_1,y1_1,x2_1,y2_1 = box1
                x1_2,y1_2,x2_2,y2_2 = box2

                cx1 = int((x1_1+x2_1)/2)
                cy1 = y2_1
                cx2 = int((x1_2+x2_2)/2)
                cy2 = y2_2

                # 거리
                dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)

                # 속도
                s1 = speeds.get(id1, 0)
                s2 = speeds.get(id2, 0)
                speed_gap = abs(s1 - s2)

                # IOU
                iou = compute_iou(box1, box2)

                # 수직 충돌
                cx_diff = abs(cx1 - cx2)
                cy_diff = abs(cy1 - cy2)

                vertical = (cx_diff < 30 and cy_diff < 80)

                pair = f"{id1}-{id2}"

                prev = pair_memory.get(pair, {
                    "dist": dist,
                    "gap": speed_gap,
                    "s1": s1,
                    "s2": s2
                })

                # ==========================
                # 패턴 분석
                # ==========================
                dist_drop = dist < prev["dist"] * 0.6
                gap_up = speed_gap > prev["gap"] * 1.5
                after_slow = (s1 < prev["s1"]) and (s2 < prev["s2"])

                # ==========================
                # 사고 판단
                # ==========================
                accident = False

                if (
                    (dist_drop and gap_up)
                    or (iou > 0.3)
                    or vertical
                ):
                    accident = True

                # ==========================
                # 충돌 유지시간
                # ==========================
                accident_counter.setdefault(pair, 0)

                if accident:
                    accident_counter[pair] += 1
                else:
                    accident_counter[pair] = 0

                final_accident = accident_counter[pair] > ACCIDENT_HOLD

                # ==========================
                # 사고 원인 분석
                # ==========================
                reason = []

                if dist_drop: reason.append("DIST_DROP")
                if gap_up: reason.append("GAP_UP")
                if iou > 0.3: reason.append("IOU")
                if vertical: reason.append("VERTICAL")
                if after_slow: reason.append("AFTER_SLOW")

                reason_str = "|".join(reason)

                # ==========================
                # 시각화
                # ==========================
                color = (0,255,0)

                if final_accident:
                    color = (0,0,255)

                    cv2.putText(frame, "ACCIDENT",
                        (cx1, cy1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0,0,255), 3)

                cv2.rectangle(frame,
                    (x1_1,y1_1),(x2_1,y2_1), color, 2)
                cv2.rectangle(frame,
                    (x1_2,y1_2),(x2_2,y2_2), color, 2)

                # ==========================
                # 로그 저장
                # ==========================
                writer.writerow([
                    frame_idx,
                    pair,
                    round(dist,2),
                    round(speed_gap,2),
                    round(iou,2),
                    vertical,
                    dist_drop,
                    gap_up,
                    after_slow,
                    final_accident
                ])

                # 상태 업데이트
                pair_memory[pair] = {
                    "dist": dist,
                    "gap": speed_gap,
                    "s1": s1,
                    "s2": s2
                }

        cv2.putText(frame,f"FRAME:{frame_idx}",
                    (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(255,255,255),2)

        out.write(frame)
        cv2.imshow("ACCIDENT V2.5",frame)

        if cv2.waitKey(1)==27:
            break

    log_file.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()


