# ==========================================
# 🚀 ACCIDENT V2.7 (최종 안정화)
# ✔ 사고 유형 분류 (3가지) : 후방추돌, 측면충돌, 기타(애매한 충돌)
# ✔ reason 로그 추가 : 사고 찍을 때 어떤 패턴이 있었는지 (거리 급감, 속도 격차 증가, IOU, 수직접근, 갑자기 느려짐)
# ✔ hold 기반 안정화 (5프레임 연속 패턴 발생 시 사고로 판단)
# ✔ 분석용 CSV 강화 (사고 여부, 사고 유형, 패턴 발생 여부 등 세부 기록)
# ✔ V2.5 + V2.6 통합
# ✔ hold 기반 안정화
# ✔ 분석용 CSV 강화
# ==========================================

import cv2
import numpy as np
import os
import csv
import time
from datetime import datetime
from ultralytics import YOLO

# ==========================
# 설정
# ==========================
MODEL_PATH = "../../scripts/runs/train/tunnel_final/weights/best.pt"
VIDEO_PATH = "../../data/raw_video/test_video/test_accident_1.mp4"

CONF = 0.25
IOU = 0.5
ALPHA = 0.3

ACCIDENT_HOLD = 5

# ==========================
# 출력
# ==========================
OUTPUT_DIR = "../../outputs/accident_v2_7"
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
pair_memory = {}
accident_counter = {}

prev_time = 0

# ==========================
# IOU 계산
# ==========================
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    union = area1 + area2 - inter
    return inter/union if union>0 else 0

# ==========================
# 사고 타입 분류
# ==========================
def classify_type(cx1,cy1,cx2,cy2, iou, vertical):
    if vertical:
        return "REAR_END"
    elif iou > 0.3:
        return "SIDE"
    else:
        return "UNKNOWN"

# ==========================
# 실행
# ==========================
def run():
    global prev_time

    cap = cv2.VideoCapture(VIDEO_PATH)

    fps_video = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        VIDEO_OUT_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps_video, (w, h)
    )

    log_file = open(LOG_PATH, "w", newline="")
    writer = csv.writer(log_file)

    writer.writerow([
        "frame","pair",
        "dist","speed_gap","iou",
        "vertical",
        "dist_drop","gap_up","after_slow",
        "hold",
        "accident",
        "type",
        "reason"
    ])

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # FPS 계산
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

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
        # 차량 추적
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

                speed = 0
                if len(track_history[tid]) >= 2:
                    (_, yp) = track_history[tid][-2]
                    (_, yc) = track_history[tid][-1]

                    dy = yc - yp
                    speed = abs(dy)

                    if tid in speed_memory:
                        speed = ALPHA*speed + (1-ALPHA)*speed_memory[tid]

                    speed_memory[tid] = speed

                speeds[tid] = speed

                cv2.putText(frame,
                    f"ID:{tid} S:{int(speed)}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(255,255,0),2)

        # ==========================
        # 차량 간 비교
        # ==========================
        ids = list(boxes.keys())

        for i in range(len(ids)):
            for j in range(i+1, len(ids)):

                id1, id2 = ids[i], ids[j]
                box1, box2 = boxes[id1], boxes[id2]

                cx1 = int((box1[0]+box1[2])/2)
                cy1 = box1[3]
                cx2 = int((box2[0]+box2[2])/2)
                cy2 = box2[3]

                dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)

                s1 = speeds.get(id1,0)
                s2 = speeds.get(id2,0)
                gap = abs(s1 - s2)

                iou = compute_iou(box1, box2)

                # 🔥 vertical 복구
                cx_diff = abs(cx1 - cx2)
                cy_diff = abs(cy1 - cy2)
                vertical = (cx_diff < 30 and cy_diff < 80)

                pair = f"{id1}-{id2}"

                prev = pair_memory.get(pair, {
                    "dist": dist,
                    "gap": gap,
                    "s1": s1,
                    "s2": s2
                })

                # ==========================
                # 패턴
                # ==========================
                dist_drop = dist < prev["dist"] * 0.6
                gap_up = gap > prev["gap"] * 1.5
                after_slow = (s1 < prev["s1"]) and (s2 < prev["s2"])

                # ==========================
                # 사고 판단 (강화 버전)
                # ==========================
                accident = (
                    (dist_drop and gap_up) or
                    (iou > 0.3) or
                    (vertical and gap_up) or
                    (after_slow and dist_drop)
                )

                # ==========================
                # HOLD
                # ==========================
                accident_counter.setdefault(pair,0)

                if accident:
                    accident_counter[pair]+=1
                else:
                    accident_counter[pair]=0

                hold = accident_counter[pair]
                final = hold > ACCIDENT_HOLD

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
                # 사고 타입
                # ==========================
                acc_type = classify_type(cx1,cy1,cx2,cy2,iou,vertical)

                # ==========================
                # 시각화
                # ==========================
                if final:
                    mx = int((cx1+cx2)/2)
                    my = int((cy1+cy2)/2)

                    cv2.putText(frame,
                        f"ACCIDENT({acc_type})",
                        (mx,my),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,(0,0,255),3)

                color = (0,0,255) if final else (0,255,0)

                cv2.rectangle(frame,box1[:2],box1[2:],color,2)
                cv2.rectangle(frame,box2[:2],box2[2:],color,2)

                # ==========================
                # 로그
                # ==========================
                writer.writerow([
                    frame_idx,pair,
                    round(dist,2),
                    round(gap,2),
                    round(iou,2),
                    vertical,
                    dist_drop,
                    gap_up,
                    after_slow,
                    hold,
                    final,
                    acc_type,
                    reason_str
                ])

                pair_memory[pair] = {
                    "dist": dist,
                    "gap": gap,
                    "s1": s1,
                    "s2": s2
                }

        cv2.putText(frame,
            f"FPS:{int(fps)}",
            (20,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,(0,255,255),2)

        out.write(frame)
        cv2.imshow("ACCIDENT V2.7",frame)

        if cv2.waitKey(1)==27:
            break

    log_file.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()