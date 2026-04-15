# ==========================================
# traffic_state_V1.py
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
# VIDEO_PATH = "../../data/raw_video/test_video/test_normal_2.mp4" # 정상
# VIDEO_PATH = "../../data/raw_video/test_video/test_accident_2.mp4" # 사고
# VIDEO_PATH = "../../data/raw_video/test_video/test_congestion_2-1.mp4" # 혼잡
VIDEO_PATH = "../../data/raw_video/test_video/test_normal_1.mp4"
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
class TrafficState:
    def __init__(self):
        self.track_history = {}
        self.speed_memory = {}
        self.state_buffer = []

    def update(self, frame_id, tracks):

        speeds = {}

        for t in tracks:
            tid = t["id"]
            x1,y1,x2,y2 = t["bbox"]

            cx = int((x1+x2)/2)
            cy = y2

            self.track_history.setdefault(tid, []).append((cx,cy))
            if len(self.track_history[tid]) > 20:
                self.track_history[tid].pop(0)

            speed = 0
            if len(self.track_history[tid]) >= 2:
                dy = self.track_history[tid][-1][1] - self.track_history[tid][-2][1]
                speed = abs(dy)

            speeds[tid] = speed

        avg_speed = sum(speeds.values())/len(speeds) if speeds else 0

        self.state_buffer.append(avg_speed)
        if len(self.state_buffer) > 50:
            self.state_buffer.pop(0)

        avg_global = sum(self.state_buffer)/len(self.state_buffer)

        if avg_global < 2:
            return "JAM"
        elif avg_global < 5:
            return "CONGESTION"
        else:
            return "NORMAL"