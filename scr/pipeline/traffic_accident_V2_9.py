# ==========================================
# 🚀 ACCIDENT V2.9 (최종 완성)
# ==========================================
# ✔ V2.5 기반 유지 (핵심 로직 유지)
# ✔ 궤적 기반 차선 (흐름) 추가
# ✔ 차선 이탈(옆 밀림 사고) 추가
# ✔ 화재(밝기 기반) 감지 추가
# ✔ 사고 유형 분류
# ✔ reason 로그 강화
# ✔ 시각화 (궤적 포함)
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
# VIDEO_PATH = "../../data/raw_video/test_video/test_accident_1.mp4"
VIDEO_PATH = "../../data/raw_video/test_video/test_congestion_1.mp4"

CONF = 0.25
IOU = 0.5
ALPHA = 0.3
ACCIDENT_HOLD = 5

# ==========================
# 출력
# ==========================
OUTPUT_DIR = "../../outputs/accident_v2_9"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, f"accident_{timestamp}.mp4")
LOG_PATH = os.path.join(OUTPUT_DIR, f"accident_{timestamp}.csv")

# ==========================
# 모델
# ==========================
model = YOLO(MODEL_PATH)

# ==========================
# 메모리 (추적, 속도, 관계)
# ==========================
track_history = {}
speed_memory = {}
pair_memory = {}
accident_counter = {}

prev_time = 0

# ==========================
# 함수들
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


# def get_lane_break(track):
#     if len(track) < 10:
#         return 0
#     return abs(track[-1][0] - track[0][0])


def get_movement_pattern(track):
    if len(track) < 10:
        return 0, 0
    dy = abs(track[-1][1] - track[0][1])
    dx = abs(track[-1][0] - track[0][0])
    return dy, dx


def vertical_overlap(box1, box2):
    y1_min, y1_max = box1[1], box1[3]
    y2_min, y2_max = box2[1], box2[3]
    overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    return overlap > 20


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
# 방향 (흐름)
# ==========================
def get_direction(track):
    if len(track) < 5:
        return 0
    return track[-1][1] - track[0][1]

# ==========================
# 차선 이탈 (옆 이동량)
# ==========================
def get_lane_break(track):
    if len(track) < 10:
        return 0
    return abs(track[-1][0] - track[0][0])

# ==========================
# 사고 유형 분류
# ==========================
def classify_type(rear, side, lane_break, fire):
    if fire:
        return "FIRE"
    elif rear:
        return "REAR_END"
    elif lane_break:
        return "LANE_BREAK"
    elif side:
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
        "dist","gap","iou",
        "vertical","dist_drop","gap_up","after_slow",
        "lane_break","fire",
        "hold","accident",
        "type","reason"
    ])

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ==========================
        # FPS 계산
        # ==========================
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        # ==========================
        # 🔥 화재 감지 (밝기 기반)
        # ==========================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fire = np.mean(gray) > 180   # 터널 오탐 위험 있음(헤드라이트, 반사 등)
        
        # ==========================
        # YOLO + ByteTrack
        # ==========================
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

                # 궤적 저장
                track_history.setdefault(tid, []).append((cx,cy))
                if len(track_history[tid]) > 30:
                    track_history[tid].pop(0)

                # 속도 계산
                speed = 0
                if len(track_history[tid]) >= 2:
                    dy = track_history[tid][-1][1] - track_history[tid][-2][1]
                    speed = abs(dy)

                    if tid in speed_memory:
                        speed = ALPHA*speed + (1-ALPHA)*speed_memory[tid]

                    speed_memory[tid] = speed

                speeds[tid] = speed

                # ==========================
                # 🔥 궤적 그리기 (핵심)
                # ==========================
                for k in range(1, len(track_history[tid])):
                    cv2.line(frame,
                        track_history[tid][k-1],
                        track_history[tid][k],
                        (0,255,255),2)

                # 정보 표시
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

                # ==========================
                # 패턴 계산
                # ==========================
                prev = pair_memory.get(f"{id1}-{id2}", {
                    "dist": dist,
                    "gap": gap
                })

                dist_drop = dist < prev["dist"] * 0.6
                gap_up = gap > prev["gap"] * 1.5
                after_slow = (s1 < 2 and s2 < 2)

                # 🔥 vertical (앞뒤 정렬)
                vertical = abs(cx1 - cx2) < 30 and abs(cy1 - cy2) < 80

                # 🔥 차선 이탈
                lane_break = (
                    get_lane_break(track_history.get(id1,[])) > 50 or
                    get_lane_break(track_history.get(id2,[])) > 50
                )

                # ==========================
                # 🚨 사고 로직 (V2.5 기반 유지)
                # ==========================
                rear = dist_drop and gap_up and vertical
                side = (iou > 0.3) and gap_up                 #사이드 단독 사용 금지. 혼잡/정체에서 오탐 제거
                lane_break_acc = lane_break and gap_up
                fire_acc = fire

                # 🔥 추가 (핵심)
                stuck = (
                    dist < 50 and
                    iou > 0.3 and
                    (s1 < 2 and s2 < 2)
                )

                accident = rear or lane_break_acc or fire_acc or (side and gap_up)  # 오탐제거 
                # ==========================
                # HOLD 안정화
                # ==========================
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
                if rear: reason.append("REAR")
                if side: reason.append("SIDE")
                if lane_break_acc: reason.append("LANE_BREAK")
                if fire_acc: reason.append("FIRE")

                reason_str = "|".join(reason)

                acc_type = classify_type(rear, side, lane_break_acc, fire_acc)

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

                # 로그 저장
                writer.writerow([
                    frame_idx,key,
                    round(dist,2),
                    round(gap,2),
                    round(iou,2),
                    vertical,dist_drop,gap_up,after_slow,
                    lane_break,fire,
                    hold,final,
                    acc_type,reason_str
                ])

                pair_memory[key] = {
                    "dist": dist,
                    "gap": gap
                }

        # ==========================
        # 🚨 적체 판단 (V2.9 개선 버전)
        # ==========================

        # 1️⃣ 차량 간 overlap 계산
        overlap_count = 0
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                if vertical_overlap(boxes[ids[i]], boxes[ids[j]]):
                    overlap_count += 1

        # 2️⃣ 차량 행동 분석
        side_move_count = 0
        stop_count = 0

        for tid, track in track_history.items():
            dy, dx = get_movement_pattern(track)

            # 앞으로 못 가는 차량
            if dy < 30:
                stop_count += 1

            # 옆으로 움직이는 차량
            if dx > 20:
                side_move_count += 1

        total = len(track_history)

        if total > 0:
            stop_ratio = stop_count / total
            side_ratio = side_move_count / total
        else:
            stop_ratio, side_ratio = 0, 0

        # 3️⃣ 최종 적체 판단
        jam = (
            overlap_count >= 1 and
            stop_ratio > 0.5 and
            side_ratio > 0.3
        )
        # 4️⃣ 시각화
        if jam:
            cv2.putText(frame,
                "TRAFFIC JAM",
                (50,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,(0,0,255),3)

        # 디버깅용 (강추)
        cv2.putText(frame,
            f"STOP:{stop_ratio:.2f} SIDE:{side_ratio:.2f}",
            (50,140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,(0,255,255),2)
        
        cv2.putText(frame,
            f"overlap:{overlap_count}",
            (50,180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,(0,255,255),2)

        cv2.putText(frame,
            f"STOP:{stop_ratio:.2f}",
            (50,210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,(0,255,255),2)

        cv2.putText(frame,
            f"SIDE:{side_ratio:.2f}",
            (50,240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,(0,255,255),2)

        # FPS 표시
        cv2.putText(frame,
            f"FPS:{int(fps)}",
            (20,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,(0,255,255),2)

        out.write(frame)
        cv2.imshow("ACCIDENT V2.9",frame)

        if cv2.waitKey(1)==27:
            break

    log_file.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()