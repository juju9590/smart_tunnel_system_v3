# ==========================================
# 🚀 ACCIDENT V3 (참고 논문 반영 버전)
# ==========================================
# ✔ V2.9 유지 (충돌 기반)
# ✔ 논문 로직 추가
#   - IoU 기반 정지 판단
#   - 블록 점유율 편차 (flow break)
#   - 평균 속도 기반 이상 차량 탐지
#   - 프레임 변화 기반 파편 감지
#   - Confidence 정지 (옵션)
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
# VIDEO_PATH = "../../data/raw_video/test_video/test_congestion_2-2.mp4"

CONF = 0.25
IOU = 0.5
ALPHA = 0.3

ACCIDENT_HOLD = 5
IOU_STOP_HOLD = 5   # IoU 정지 유지 프레임

INIT_FRAMES = 300

# ==========================
# 출력
# ==========================
OUTPUT_DIR = "../../outputs/accident_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 입력 영상 이름에서 상태 추출
video_name = os.path.basename(VIDEO_PATH).lower()

if "accident" in video_name:
    state_name = "accident"
elif "congestion" in video_name:
    state_name = "congestion"
elif "normal" in video_name:
    state_name = "normal"
else:
    state_name = "unknown"

# 최종 파일명 생성
VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR,f"{state_name}_{timestamp}.mp4")
LOG_PATH = os.path.join(OUTPUT_DIR,f"{state_name}_{timestamp}.csv")

# ==========================
# 모델
# ==========================
model = YOLO(MODEL_PATH)

# ==========================
# 메모리
# ==========================
track_history = {}
speed_memory = {}
pair_memory = {}
accident_counter = {}
lanes = {}

# 차선
lane_initialized = False
lane_centers = []
cos_a, sin_a = 1, 0

# 개선 포인트 추가 
iou_stop_counter = {}     # IoU 정지 유지
# conf_stack = {}           # confidence 스택 (나중에 다시 생각)

prev_frame = None
prev_time = 0

noise_history = []

# ==========================
# IoU 계산
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
# 차선 이탈
# ==========================
def get_lane_break(track):
    if len(track) < 10:
        return 0

    # 시작점 / 끝점
    x0, y0 = track[0]
    x1, y1 = track[-1]

    # 🔥 회전 좌표 적용
    xr0 = x0 * cos_a + y0 * sin_a
    xr1 = x1 * cos_a + y1 * sin_a

    # 차선 방향 기준 이동량
    return abs(xr1 - xr0)

# ==========================
# 실행
# ==========================
def run():
    global prev_time, prev_frame
    global lane_initialized, lane_centers, cos_a, sin_a

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
    "frame","id1","id2",
    "cx1","cy1","cx2","cy2",
    "dist","s1","s2","gap","iou",
    "lane1","lane2",
    "prev_dist","prev_gap",
    "dist_drop","gap_up","vertical",
    "lane_break","stop_confirm","abnormal",
    "rear","side","lane_break_acc",
    "accident"
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
        # 🔥 파편 감지 (초기 300프레임 기준 - 안정형)
        # ==========================

        # 👉 전역 or 상단에 추가 (한 번만 선언)
        # noise_history = []

        fragment = False

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)
            noise = np.mean(diff)

            # ==========================
            # 1️⃣ 초기 구간 (baseline 학습)
            # ==========================
            noise_history.append(noise)

            if len(noise_history) > 300:
                noise_history.pop(0)

            baseline = np.mean(noise_history)

            # ==========================
            # 2️⃣ 파편 판단 (상대값 기준)
            # ==========================
            if len(noise_history) >= 50:  # 최소 안정 구간
                fragment = noise > baseline * 1.5

        # 이전 프레임 업데이트
        prev_frame = frame.copy()

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
        confs = {}

        # ==========================
        # 차량 추적 + 시각화 (V3 완성)
        # ==========================
        if results[0].boxes is not None and results[0].boxes.id is not None:

            for box, tid, conf in zip(
                results[0].boxes.xyxy,
                results[0].boxes.id,
                results[0].boxes.conf
            ):

                x1,y1,x2,y2 = map(int, box)
                tid = int(tid)

                cx = int((x1+x2)/2)
                cy = y2

                boxes[tid] = (x1,y1,x2,y2)
                confs[tid] = float(conf)

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

                # # confidence stack (나중에 다시 생각)
                # conf_stack.setdefault(tid, []).append(confs[tid])
                # if len(conf_stack[tid]) > 10:
                #     conf_stack[tid].pop(0)

                # ==========================
                #  바운딩 박스 색상
                # ==========================
                color = (0,255,0)   # 기본: 초록 (정상)

                # ==========================
                # 바운딩 박스
                # ==========================
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # ==========================
                # ID + 속도 표시
                # ==========================
                cv2.putText(frame,
                    f"ID:{tid} S:{int(speed)}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,0),
                    2
                )

                # ==========================
                # 궤적 표시 (디버깅 핵심)
                # ==========================
                for k in range(1, len(track_history[tid])):
                    cv2.line(frame,
                        track_history[tid][k-1],
                        track_history[tid][k],
                        (0,255,255), 2)
                    
        # ==========================
        # 방향 기반 차선 학습
        # ==========================
        if not lane_initialized and frame_idx > INIT_FRAMES:

            directions = []

            for track in track_history.values():
                if len(track) >= 10:
                    dx = track[-1][0] - track[0][0]
                    dy = track[-1][1] - track[0][1]
                    if abs(dx)+abs(dy) > 5:
                        directions.append([dx,dy])

            if len(directions) > 5:
                avg = np.mean(directions, axis=0)
                angle = np.arctan2(avg[1], avg[0])

                cos_a = np.cos(angle)
                sin_a = np.sin(angle)

                rotated = []

                for track in track_history.values():
                    for x,y in track:
                        xr = x*cos_a + y*sin_a
                        rotated.append(xr)

                from sklearn.cluster import KMeans

                data = np.array(rotated).reshape(-1,1)
                k = min(4, len(data))

                kmeans = KMeans(n_clusters=k, n_init=10).fit(data)
                lane_centers = sorted([c[0] for c in kmeans.cluster_centers_])

                lane_initialized = True
                print("🔥 차선 학습 완료")

        # ==========================
        # 차선 할당
        # ==========================
        for tid, box in boxes.items():
            cx = int((box[0]+box[2])/2)
            cy = box[3]

            lane = -1

            if lane_initialized:
                xr = cx*cos_a + cy*sin_a
                d = [abs(xr-c) for c in lane_centers]
                lane = d.index(min(d))+1

            lanes[tid] = lane

            # ==========================
            # 🔥 차선 표시 (분리)
            # ==========================
            if lane != -1:
                cv2.putText(frame,
                    f"L:{lane}",
                    (x1, y2+15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,255),
                    2
                )

        # ==========================
        # 평균 속도 (전체 흐름)
        # ==========================
        if len(speeds) > 0:
            avg_speed = np.mean(list(speeds.values()))
        else:
            avg_speed = 0

        # ==========================
        # 🔥 블록 점유율 분석 (논문 핵심)
        # ==========================
        upper = 0
        lower = 0

        for tid, box in boxes.items():
            cy = box[3]

            if cy < h/2:
                upper += 1
            else:
                lower += 1

        total = len(boxes)
        if total > 0:
            occupancy_diff = abs(upper - lower) / total
        else:
            occupancy_diff = 0

        flow_break = occupancy_diff > 0.3   # 👉 튜닝 가능

        # ==========================
        # 차량 간 비교
        # ==========================
        # stop_confirm = False

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
                # 🔥 차선 비교 (여기!!)
                # ==========================
                lane1 = lanes.get(id1, -1)
                lane2 = lanes.get(id2, -1)

                same_lane = (lane1 == lane2 and lane1 != -1)

                # ==========================
                # 앞차와 뒤차의 거리와 간격 비교
                # ==========================
                prev = pair_memory.get(f"{id1}-{id2}", {
                    "dist": dist,
                    "gap": gap
                })

                dist_drop = dist < prev["dist"] * 0.6
                # gap_up = gap > prev["gap"] * 1.5
                gap_up = gap > prev["gap"] * 1.2 or gap > 3 #(조건완화)

                vertical = abs(cx1 - cx2) < 30


                # 차선 기준에서 실제 옆 이동
                lane_break = (
                    get_lane_break(track_history.get(id1,[])) > 50 or
                    get_lane_break(track_history.get(id2,[])) > 50
                )

                # ==========================
                # IoU 정지 판단 (논문 핵심)
                # ==========================
                key = f"{id1}-{id2}"
                iou_stop_counter.setdefault(key, 0)

                if iou > 0.9:
                    iou_stop_counter[key] += 1
                else:
                    iou_stop_counter[key] = 0

                stop_confirm = iou_stop_counter[key] > IOU_STOP_HOLD

                # ==========================
                # 개별 이상 차량
                # ==========================
                abnormal = (
                    (s1 < 2 or s2 < 2) and avg_speed > 5
                )

                # ==========================
                # 🚨 최종 사고 로직 (V3 핵심)
                # ==========================
                rear = dist_drop and gap_up and vertical
                side = (iou > 0.3) and gap_up
                lane_break_acc = lane_break and gap_up
                
                accident = (same_lane and (rear 
                        or side
                        or (iou > 0.9 and stop_confirm)
                        or lane_break_acc 
                        or (fragment and abnormal and gap_up)))
                
                # rear 추돌, side 측면, iou+stop 정차충돌, lane_break 급차선 이탈

                # ==========================
                # HOLD 안정화
                # ==========================

                key = f"{id1}-{id2}"
                accident_counter.setdefault(key,0)

                if accident:
                    accident_counter[key]+=1
                else:
                    accident_counter[key]=0

                final = accident_counter[key] > ACCIDENT_HOLD                

                # ==========================
                # 🔥 색상 (3단계)
                # ==========================
                color = (0,255,0)  #초록

                if final:
                    color = (0,0,255) #빨강(확정)
                elif accident:
                    color = (0,255,255) #노랑(의심)

                # ==========================
                # 바운딩 박스
                # ==========================
                cv2.rectangle(frame, (box1[0], box1[1]), (box1[2], box1[3]), color, 2)
                cv2.rectangle(frame, (box2[0], box2[1]), (box2[2], box2[3]), color, 2)


                # ==========================
                # 사고 차량 연결선
                # ==========================
                if final:
                    cv2.line(frame,(cx1, cy1),(cx2, cy2),(0,0,255),3) 

                pair_memory[key] = {"dist": dist,"gap": gap}

                # ==========================
                # 🔥 상세 로그 (여기 추가)
                # ==========================
                writer.writerow([
                    frame_idx, id1, id2,

                    cx1, cy1, cx2, cy2,

                    round(dist,2),
                    round(s1,2),
                    round(s2,2),
                    round(gap,2),
                    round(iou,3),

                    lane1, lane2,

                    round(prev["dist"],2),
                    round(prev["gap"],2),

                    dist_drop,
                    gap_up,
                    vertical,

                    lane_break,
                    stop_confirm,
                    abnormal,

                    rear,
                    side,
                    lane_break_acc,

                    accident
                ])
                

                # ==========================
                # 사고 텍스트 (final만!)
                # ==========================
                if final:
                    mx = int((cx1+cx2)/2)
                    my = int((cy1+cy2)/2)

                    cv2.putText(frame,
                        "ACCIDENT",
                        (mx, my-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0,0,255),
                        3
                    )         

        # FPS 표시
        cv2.putText(frame,f"FPS:{int(fps)}",(20,30),
            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        
        cv2.putText(frame,f"flow:{flow_break} stop:{stop_confirm}",(20,60),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        cv2.putText(frame,f"abn:{abnormal} frag:{fragment}",(20,90),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        out.write(frame)
        cv2.imshow("ACCIDENT V3",frame)

        if cv2.waitKey(1)==27:
            break

    log_file.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()