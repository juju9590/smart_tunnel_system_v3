# 테스트 코드
# yolo11n 모델 + 영상 테스트 시스템 검증 스크립트
# 차량수, FPS, 상태 판단, 추적(ID 유지) 

import cv2
from ultralytics import YOLO
import time

# =========================
# 설정
# =========================
MODEL_PATH = r"D:\smart_tunnel_V3\scripts\runs\train\tunnel_final\weights\best.pt"
VIDEO_PATH = r"D:\smart_tunnel_V3\data\raw_video\test_normal_2.mp4" # 테스트 영상 경로

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ 영상 열기 실패:", VIDEO_PATH)
    exit()

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # 추론
    # =========================
    results = model.track(
        frame,
        persist=True,
        conf=0.3
    )

    boxes = results[0].boxes

    vehicle_count = 0

    if boxes is not None:
        vehicle_count = len(boxes)

    # =========================
    # FPS 계산
    # =========================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # =========================
    # 상태 판단 (임시 로직)
    # =========================
    if vehicle_count >= 10:
        state = "JAM"
    elif vehicle_count >= 5:
        state = "CONGESTION"
    else:
        state = "NORMAL"

    # =========================
    # 화면 출력
    # =========================
    annotated_frame = results[0].plot()

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(annotated_frame, f"State: {state}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Tunnel System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
