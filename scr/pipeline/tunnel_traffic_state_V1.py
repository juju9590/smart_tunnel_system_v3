# ==========================================
# 🚀 SMART TUNNEL V1 (영상 저장 + 로그)
# ==========================================

import cv2
import numpy as np
import os
import csv
from datetime import datetime
from ultralytics import YOLO

# ==========================
# 🔧 설정
# ==========================
MODEL_PATH = "../../scripts/runs/train/tunnel_final/weights/best.pt"
VIDEO_PATH = "../../data/raw_video/test_video/test_normal_2.mp4" # 정상
# VIDEO_PATH = "../../data/raw_video/test_video/test_accident_2.mp4" # 사고
# VIDEO_PATH = "../../data/raw_video/test_video/test_congestion_2-1.mp4" # 혼잡
print(os.path.exists(MODEL_PATH))

CONF = 0.25
IOU = 0.5

# ROI 영역 설정 
ROI_Y1_RATIO = 0.3
ROI_Y2_RATIO = 0.8

# 상태 판단 파라미터(속도 기준 smoothing)
ALPHA = 0.3

# 상태 판단 임계값 (속도 기준)
JAM_ENTER = 2.3
CONGESTION_ENTER = 4.5

# 사고 감지 파라미터
STATE_BUFFER_SIZE = 300

# 사고 파라미터
DIST_TH = 40
DIST_DIFF_TH = 5
SPEED_GAP_TH = 1.5
IOU_TH = 0.05
LOW_SPEED_TH = 1.0

WINDOW_SIZE = 20
ACCIDENT_COUNT_TH = 5

# ==========================
# 📁 출력 경로
# ==========================

# 출력 폴더 생성
OUTPUT_DIR = "../../outputs/v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# 버전 자동 증가 함수
# ==========================
def get_next_version(output_dir, prefix="tunnel_traffic"):
    files = os.listdir(output_dir)

    versions = []
    for f in files:
        if f.startswith(prefix) and f.endswith(".mp4"):
            try:
                v = int(f.split("_V")[-1].split("_")[0])
                versions.append(v)
            except:
                continue

    return max(versions) + 1 if versions else 1

# 현재 시간 기반 타임스탬프 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 👉 버전 자동 생성
version = get_next_version(OUTPUT_DIR)

VIDEO_OUT_PATH = os.path.join(
    OUTPUT_DIR,
    f"tunnel_traffic_V{version}_{timestamp}.mp4"
)

LOG_PATH = os.path.join(
    OUTPUT_DIR,
    f"tunnel_traffic_V{version}_{timestamp}.csv"
)

# ==========================
# 모델
# ==========================
model = YOLO(MODEL_PATH)

# ==========================
# 메모리
# ==========================
track_history = {} # 차량 ID별 위치 기록
speed_memory = {} # 차량 ID별 속도 기록 (이전 프레임 속도 저장)
prev_speeds = {} # 사고 감지 위해 이전 프레임 속도 저장

distance_memory = {} # 차량 쌍별 이전 거리 저장 (사고 감지 위해)
collision_memory = {} # 차량 쌍별 충돌 여부 저장 (사고 감지 위해)
accident_history = [] # 최근 프레임별 사고 여부 기록 (사고 이벤트 안정화 위해)

state_buffer = [] # 최근 프레임별 평균 속도 기록 (상태 판단 안정화 위해)

print("LOG PATH:", LOG_PATH)

# ==========================
# 실행
# ==========================
def run():

    cap = cv2.VideoCapture(VIDEO_PATH)

    # 👉 영상 정보 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FPS: {fps}, SIZE: {w}x{h}")

    # 👉 결과 영상 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps, (w, h))

    # 👉 로그 파일
    log_file = open(LOG_PATH, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "frame_idx",
        "tid", 
        "speed",
        "prev_dist",
        "dist_diff",
        "state", 
        "event"])
    
    log_file.flush() # 버퍼링 방지 위해 매 프레임마다 기록

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ROI 계산
        ROI_Y1 = int(h * ROI_Y1_RATIO)
        ROI_Y2 = int(h * ROI_Y2_RATIO)

        # ==========================
        # ROI 시각화 (추가)
        # ==========================
        cv2.line(frame, (0, ROI_Y1), (w, ROI_Y1), (0,255,255), 2)
        cv2.line(frame, (0, ROI_Y2), (w, ROI_Y2), (0,255,255), 2)

        results = model.track(frame, persist=True,
                              conf=CONF, iou=IOU,
                              tracker="bytetrack.yaml")

        speeds = {}
        boxes = {}

        # ==========================
        # 🚗 차량 추적 + 속도
        # ==========================
        if results[0].boxes is not None and results[0].boxes.id is not None:

            for box, tid in zip(results[0].boxes.xyxy,
                                results[0].boxes.id):

                x1,y1,x2,y2 = map(int, box)
                tid = int(tid)

                cx = int((x1+x2)/2)
                cy = y2


                # ROI 밖은 속도 계산에서 제외 (노이즈 방지)
                if cy < ROI_Y1 or cy > ROI_Y2:
                    continue

                boxes[tid] = (x1,y1,x2,y2)

                track_history.setdefault(tid, []).append((cx,cy))
                if len(track_history[tid]) > 20:
                    track_history[tid].pop(0)

                if len(track_history[tid]) >= 2:

                    (_, yp) = track_history[tid][-2]
                    (_, yc) = track_history[tid][-1]

                    dy = yc - yp
                    scale = (cy-ROI_Y1)/(ROI_Y2-ROI_Y1+1e-6)

                    speed = abs(dy)/(scale+0.15)
                    speed *= (1.2+scale)

                    if tid in speed_memory:
                        speed = ALPHA*speed + (1-ALPHA)*speed_memory[tid]

                    speed_memory[tid] = speed
                    speeds[tid] = speed

        # ==========================
        # 🚨 사고 감지
        # ==========================
        accident_flag = False
        ids = list(boxes.keys())

        for i in range(len(ids)):
            for j in range(i+1, len(ids)):

                id1, id2 = ids[i], ids[j]

                if id1 not in speeds or id2 not in speeds:
                    continue

                (x1,y1) = track_history[id1][-1]
                (x2,y2) = track_history[id2][-1]

                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)

                key = tuple(sorted((id1,id2)))

                prev_dist = distance_memory.get(key, dist)
                distance_memory[key] = dist

                dist_diff = prev_dist - dist

                s1 = speeds[id1]
                s2 = speeds[id2]

                if y1 > y2:
                    front_s, rear_s = s1, s2
                    rear_id = id2
                else:
                    front_s, rear_s = s2, s1
                    rear_id = id1

                speed_gap = rear_s - front_s

                # 사고 예측
                if dist < DIST_TH and speed_gap > SPEED_GAP_TH and dist_diff > DIST_DIFF_TH:
                    accident_flag = True

                # IOU
                iou = compute_iou(boxes[id1], boxes[id2])

                if iou > IOU_TH:
                    collision_memory[key] = 1

                # 충돌 후 감속
                if collision_memory.get(key,0):

                    rear_speed = speeds.get(rear_id,0)
                    prev_rear = prev_speeds.get(rear_id, rear_speed)

                    if prev_rear > 3 and rear_speed < LOW_SPEED_TH:
                        accident_flag = True

        # ==========================
        # 이벤트 안정화
        # ==========================
        accident_history.append(1 if accident_flag else 0)

        if len(accident_history) > WINDOW_SIZE:
            accident_history.pop(0)

        event = "ACCIDENT" if sum(accident_history) >= ACCIDENT_COUNT_TH else "NONE"

        # ==========================
        # 상태 판단
        # ==========================
        avg_speed = np.mean(list(speeds.values())) if speeds else 0

        state_buffer.append(avg_speed)
        if len(state_buffer) > STATE_BUFFER_SIZE:
            state_buffer.pop(0)

        avg_global = np.mean(state_buffer)

        if avg_global < JAM_ENTER:
            state = "JAM"
        elif avg_global < CONGESTION_ENTER:
            state = "CONGESTION"
        else:
            state = "NORMAL"

        # ==========================
        # 시각화
        # ==========================
        for tid,(x1,y1,x2,y2) in boxes.items():

            color = (0,255,0)
            if event == "ACCIDENT":
                color = (0,0,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            if tid in speeds:
                cv2.putText(frame,
                            f"ID:{tid} S:{round(speeds[tid],1)}",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

            # 로그 기록
            writer.writerow([
            frame_idx,
            tid,
            speeds.get(tid,0),
            prev_dist if 'prev_dist' in locals() else 0,
            dist_diff if 'dist_diff' in locals() else 0,
            state,
            event
        ])
            
            
            

        # 상태 시각화
        cv2.putText(frame,f"STATE:{state}",(20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        # 평균 속도 시각화 (상태 판단 안정화 위해)
        cv2.putText(frame,f"AVG(10's):{round(avg_global,1)}",(20,70),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        # 사고 이벤트 시각화
        cv2.putText(frame,f"EVENT:{event}",(20,110),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        # 👉 영상 저장
        out.write(frame)

        cv2.imshow("SMART TUNNEL V1-1",frame)

        if cv2.waitKey(1)==27:
            break

        for tid in speeds:
            prev_speeds[tid] = speeds[tid]

    cap.release()
    out.release()
    log_file.close()
    cv2.destroyAllWindows()


def compute_iou(b1,b2):
    x1,y1,x2,y2 = b1
    x1g,y1g,x2g,y2g = b2

    xi1 = max(x1,x1g)
    yi1 = max(y1,y1g)
    xi2 = min(x2,x2g)
    yi2 = min(y2,y2g)

    inter = max(0,xi2-xi1)*max(0,yi2-yi1)

    area1 = (x2-x1)*(y2-y1)
    area2 = (x2g-x1g)*(y2g-y1g)

    union = area1+area2-inter+1e-6

    return inter/union


if __name__ == "__main__":
    run()