# ==========================================
# 🚀 SMART ACCIDENT V2 (차선 + 밝기 포함)
# ✔ 차량 추적 + 속도
# ✔ 차선 자동 학습 (600프레임)
# ✔ 차선 변경 감지
# ✔ 거리 + 속도차 사고 로직
# ✔ 밝기(FLASH) 감지 추가
# ✔ 화면 표시 + 로그 기록
# ==========================================

import cv2
import numpy as np
import os
import csv
from datetime import datetime
from ultralytics import YOLO
from sklearn.cluster import KMeans

# ==========================
# 설정
# ==========================
MODEL_PATH = "../../scripts/runs/train/tunnel_final/weights/best.pt"
VIDEO_PATH = "../../data/raw_video/test_video/test_accident_1.mp4"

CONF = 0.25
IOU = 0.5
ALPHA = 0.3

LANE_LEARN_FRAMES = 600   # 차선 학습 구간

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

# 사고용
prev_dist = {}
prev_speed = {}

# 차선용
lane_points = []
lane_centers = None
vehicle_lane = {}

# 밝기용 🔥
prev_brightness = None

# ==========================
def get_lane_id(cx):
    if lane_centers is None:
        return -1
    distances = [abs(cx - c) for c in lane_centers]
    return np.argmin(distances)

# ==========================
def run():

    global lane_centers, prev_brightness

    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(VIDEO_OUT_PATH,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (w,h))

    log_file = open(LOG_PATH,"w",newline="")
    writer = csv.writer(log_file)

    writer.writerow([
        "frame","pair","dist","speed_gap",
        "dist_diff","speed_diff",
        "lane_change","brightness_diff",
        "brightness_event","accident"
    ])

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        accident_flag = False
        lane_change_flag = False
        brightness_event = False

        # ==========================
        # 🔥 밝기 감지
        # ==========================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if prev_brightness is not None:
            brightness_diff = brightness - prev_brightness

            if brightness_diff > 15:
                brightness_event = True
        else:
            brightness_diff = 0

        prev_brightness = brightness

        # ==========================
        # YOLO tracking
        # ==========================
        results = model.track(frame, persist=True,
                              conf=CONF, iou=IOU,
                              tracker="bytetrack.yaml")

        speeds = {}
        boxes = {}

        if results[0].boxes is not None and results[0].boxes.id is not None:

            for box, tid in zip(results[0].boxes.xyxy,
                                results[0].boxes.id):

                x1,y1,x2,y2 = map(int, box)
                tid = int(tid)

                cx = int((x1+x2)/2)
                cy = y2

                boxes[tid] = (x1,y1,x2,y2)

                # ==========================
                # 차선 학습
                # ==========================
                if frame_idx < LANE_LEARN_FRAMES:
                    lane_points.append(cx)

                # ==========================
                # 속도 계산
                # ==========================
                track_history.setdefault(tid, []).append((cx,cy))
                if len(track_history[tid]) > 10:
                    track_history[tid].pop(0)

                if len(track_history[tid]) >= 2:
                    (_, yp) = track_history[tid][-2]
                    (_, yc) = track_history[tid][-1]

                    dy = yc - yp
                    speed = abs(dy)

                    if tid in speed_memory:
                        speed = ALPHA*speed + (1-ALPHA)*speed_memory[tid]

                    speed_memory[tid] = speed
                    speeds[tid] = speed

        # ==========================
        # 차선 생성
        # ==========================
        if frame_idx == LANE_LEARN_FRAMES:
            X = np.array(lane_points).reshape(-1,1)

            kmeans = KMeans(n_clusters=3).fit(X)
            lane_centers = sorted(kmeans.cluster_centers_.flatten())

            print("차선 중심:", lane_centers)

        # ==========================
        # 차선 판단
        # ==========================
        for tid,(x1,y1,x2,y2) in boxes.items():

            cx = int((x1+x2)/2)
            lane_id = get_lane_id(cx)

            if tid in vehicle_lane:
                if vehicle_lane[tid] != lane_id:
                    lane_change_flag = True

            vehicle_lane[tid] = lane_id

            cv2.putText(frame,
                f"ID:{tid} L:{lane_id} S:{int(speeds.get(tid,0))}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,(0,255,255),2)

        # ==========================
        # 사고 판단
        # ==========================
        ids = list(boxes.keys())

        for i in range(len(ids)):
            for j in range(i+1, len(ids)):

                id1 = ids[i]
                id2 = ids[j]

                x1,y1,x2,y2 = boxes[id1]
                x3,y3,x4,y4 = boxes[id2]

                cx1 = (x1+x2)//2
                cy1 = y2
                cx2 = (x3+x4)//2
                cy2 = y4

                dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)

                s1 = speeds.get(id1,0)
                s2 = speeds.get(id2,0)
                speed_gap = abs(s1-s2)

                key = f"{id1}_{id2}"

                prev_d = prev_dist.get(key, dist)
                prev_s = prev_speed.get(key, speed_gap)

                dist_diff = prev_d - dist
                speed_diff = speed_gap - prev_s

                # 🚨 사고 조건 (핵심)
                if dist_diff > 15 and speed_diff > 10:
                    if s1 < 3 and s2 < 3:
                        if brightness_event or lane_change_flag:
                            accident_flag = True

                prev_dist[key] = dist
                prev_speed[key] = speed_gap

                writer.writerow([
                    frame_idx,
                    f"id{id1}_id{id2}",
                    round(dist,2),
                    round(speed_gap,2),
                    round(dist_diff,2),
                    round(speed_diff,2),
                    int(lane_change_flag),
                    round(brightness_diff,2),
                    int(brightness_event),
                    int(accident_flag)
                ])

        # ==========================
        # 시각화
        # ==========================
        for tid,(x1,y1,x2,y2) in boxes.items():
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        if brightness_event:
            cv2.putText(frame,"FLASH",(50,120),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

        if accident_flag:
            cv2.putText(frame,"ACCIDENT",(50,80),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)

        out.write(frame)
        cv2.imshow("ACCIDENT V2",frame)

        if cv2.waitKey(1)==27:
            break

    log_file.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# ==========================
if __name__ == "__main__":
    run()