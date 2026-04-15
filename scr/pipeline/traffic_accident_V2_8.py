# ==========================================
# 🚀 ACCIDENT V2.8 (최종 완성)
# ==========================================
# ✔ V2.7 전부 유지
# ✔ 방향 기반 같은 흐름 판단 (300 기준)
# ✔ 충돌 후 방향 변화 감지 (차선 이탈)
# ✔ 사고 3단계 판단 (발생 + 충돌 + 이후)
# ✔ 사고 유형 3가지
# ✔ reason 로그 완전 확장
# ✔ CSV 분석 가능 구조
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
OUTPUT_DIR = "../../outputs/accident_v2_8"
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
# 방향 계산 (핵심)
# ==========================
def get_direction(track):
    if len(track) < 5:
        return 0
    return track[-1][1] - track[0][1]

# ==========================
# 같은 흐름 판단 (300 기준)
# ==========================
def same_flow(track1, track2):
    d1 = get_direction(track1)
    d2 = get_direction(track2)

    same_dir = (d1 * d2 > 0)
    strong = (abs(d1) > 300 and abs(d2) > 300)

    return same_dir and strong

# ==========================
# 사고 유형 분류
# ==========================
def classify_type(rear_end, collision, post):
    if rear_end:
        return "REAR_END"
    elif collision:
        return "SIDE"
    elif post:
        return "POST"
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
        "dist","gap","iou",
        "same_flow",
        "dist_drop","gap_up","after_slow",
        "direction_change","speed_low",
        "hold","accident",
        "type","reason"
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
                    dy = track_history[tid][-1][1] - track_history[tid][-2][1]
                    speed = abs(dy)

                    if tid in speed_memory:
                        speed = ALPHA*speed + (1-ALPHA)*speed_memory[tid]

                    speed_memory[tid] = speed

                speeds[tid] = speed

                # 화면 표시
                direction = get_direction(track_history[tid])
                cv2.putText(frame,
                    f"ID:{tid} S:{int(speed)} D:{int(direction)}",
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

                box1 = boxes[id1]
                box2 = boxes[id2]

                cx1 = int((box1[0]+box1[2])/2)
                cy1 = box1[3]
                cx2 = int((box2[0]+box2[2])/2)
                cy2 = box2[3]

                dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)

                s1 = speeds.get(id1,0)
                s2 = speeds.get(id2,0)
                gap = abs(s1 - s2)

                iou = compute_iou(box1, box2)

                track1 = track_history.get(id1, [])
                track2 = track_history.get(id2, [])

                # 🔥 같은 흐름 판단
                same_flow_flag = same_flow(track1, track2)

                # 이전값
                prev = pair_memory.get(f"{id1}-{id2}", {
                    "dist": dist,
                    "gap": gap,
                    "dir1": get_direction(track1),
                    "dir2": get_direction(track2)
                })

                # 패턴
                dist_drop = dist < prev["dist"] * 0.6
                gap_up = gap > prev["gap"] * 1.5
                after_slow = (s1 < 2 and s2 < 2)

                # 🔥 방향 변화 (차선 이탈)
                curr_dir1 = get_direction(track1)
                curr_dir2 = get_direction(track2)

                direction_change = (
                    abs(curr_dir1 - prev["dir1"]) > 150 or
                    abs(curr_dir2 - prev["dir2"]) > 150
                )

                speed_low = (s1 < 2 and s2 < 2)

                # ==========================
                # 사고 로직 (핵심)
                # ==========================
                rear_end = (
                    same_flow_flag and dist_drop and gap_up
                )

                collision = (
                    iou > 0.4 or dist < 40
                )

                post = (
                    after_slow and direction_change
                )

                # ❗ 핵심: 단일 조건 제거
                accident = (
                    (rear_end and collision) or
                    (collision and post)
                )

                # HOLD
                key = f"{id1}-{id2}"
                accident_counter.setdefault(key,0)

                if accident:
                    accident_counter[key]+=1
                else:
                    accident_counter[key]=0

                hold = accident_counter[key]
                final = hold > ACCIDENT_HOLD

                # ==========================
                # reason 기록
                # ==========================
                reason = []
                if rear_end: reason.append("REAR_END")
                if collision: reason.append("COLLISION")
                if post: reason.append("POST")
                if dist_drop: reason.append("DIST_DROP")
                if gap_up: reason.append("GAP_UP")
                if iou > 0.3: reason.append("IOU")
                if direction_change: reason.append("DIR_CHANGE")

                reason_str = "|".join(reason)

                # 사고 타입
                acc_type = classify_type(rear_end, collision, post)

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

                # 로그
                writer.writerow([
                    frame_idx,key,
                    round(dist,2),
                    round(gap,2),
                    round(iou,2),
                    same_flow_flag,
                    dist_drop,
                    gap_up,
                    after_slow,
                    direction_change,
                    speed_low,
                    hold,
                    final,
                    acc_type,
                    reason_str
                ])

                pair_memory[key] = {
                    "dist": dist,
                    "gap": gap,
                    "dir1": curr_dir1,
                    "dir2": curr_dir2
                }

        cv2.putText(frame,
            f"FPS:{int(fps)}",
            (20,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,(0,255,255),2)

        out.write(frame)
        cv2.imshow("ACCIDENT V2.8",frame)

        if cv2.waitKey(1)==27:
            break

    log_file.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()