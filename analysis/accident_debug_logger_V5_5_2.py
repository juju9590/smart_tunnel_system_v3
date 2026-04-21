# ==========================================
# 파일명: accident_debug_logger_V5_5_2.py
# 설명:
# SMART TUNNEL V5_5_2 사고 디버그 로거
#
# 기능
# 1) 테스트 영상을 읽는다
# 2) YOLO + ByteTrack으로 차량 추적
# 3) 간단한 speed(EMA) 계산
# 4) 초기 lane bootstrap 후 lane_id 추정
# 5) traffic_accident_V5_5_2.py 로 사고 pair 분석
# 6) pair 디버그 CSV 저장
# 7) 디버그 영상 저장
#
# 출력
# - accident_debug_v5_5_2_YYYYMMDD_HHMMSS.csv
# - accident_debug_v5_5_2_YYYYMMDD_HHMMSS.mp4
#
# 주의
# - lane bootstrap은 이 파일 안의 "간단 버전"임
# - 네 기존 lane_template / track_analyzer가 더 정확하면
#   그쪽으로 교체해서 써도 됨
# ==========================================

import os
import sys
import csv
import cv2
import traceback
from datetime import datetime
from collections import defaultdict, deque

import numpy as np
from ultralytics import YOLO

# =========================================================
# 0. 경로 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# 외부 데이터 / 출력 폴더
DATA_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "smart_tunnel_V3_data")
OUTPUT_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "smart_tunnel_V3_outputs")

# 사고로직 폴더
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "scr", "pipeline_V5_5(사고로직_개선)")
if not os.path.exists(PIPELINE_DIR):
    # 혹시 실제 폴더명이 pipeline_V5_5 라면 fallback
    alt_dir = os.path.join(PROJECT_ROOT, "scr", "pipeline_V5_5")
    if os.path.exists(alt_dir):
        PIPELINE_DIR = alt_dir

if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

from traffic_accident_V5_5_2 import AccidentDetector

print("🚀 accident_debug_logger_V5_5_2 시작")

# =========================================================
# 1) 기본 경로 / 파일
# =========================================================
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "analysis_v5_5_2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 테스트 영상 선택
# ---------------------------------------------------------
VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_congestion_2-1.mp4"
# VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_congestion_2-2.mp4"
# VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_accident_1-1.mp4"
# VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_accident_3.mp4"

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")
TRACKER_CONFIG = "bytetrack.yaml"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = os.path.join(OUTPUT_DIR, f"accident_debug_v5_5_2_{timestamp}.csv")
VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, f"accident_debug_v5_5_2_{timestamp}.mp4")

print("PROJECT_ROOT:", PROJECT_ROOT)
print("PIPELINE_DIR:", PIPELINE_DIR)
print("traffic_accident_V5_5_2 exists:", os.path.exists(os.path.join(PIPELINE_DIR, "traffic_accident_V5_5_2.py")))
print("VIDEO_PATH:", VIDEO_PATH)
print("영상 존재 여부:", os.path.exists(VIDEO_PATH))
print("MODEL_PATH:", MODEL_PATH)
print("모델 존재 여부:", os.path.exists(MODEL_PATH))
print("CSV_PATH:", CSV_PATH)
print("VIDEO_OUT_PATH:", VIDEO_OUT_PATH)

# =========================================================
# 2) 설정값
# =========================================================
CLASS_VEHICLE_IDS = {0, 1, 2, 3, 5, 7}
CONF_THRES = 0.25
IOU_THRES = 0.45

# 속도 계산
EMA_ALPHA = 0.30
SPEED_SPIKE_CLAMP = 12.0
MIN_VALID_DY = 0.0

# lane bootstrap
LANE_BOOTSTRAP_FRAMES = 100
LANE_UPDATE_INTERVAL = 10
LANE_MIN_SAMPLES = 40
LANE_K_CANDIDATES = [2, 3, 4]

# 화면 표시
SHOW_WINDOW = True
DRAW_BOX = True
DRAW_TRACK = True
DRAW_PAIR_TEXT = True

# =========================================================
# 3) 유틸 함수
# =========================================================
def bottom_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2

def choose_lane_count_kmeans(x_values, candidates=(2, 3, 4)):
    if not SKLEARN_AVAILABLE:
        return 2

    x_values = np.array(x_values, dtype=np.float32).reshape(-1, 1)
    if len(x_values) < 10:
        return 2

    best_k = 2
    best_score = None

    for k in candidates:
        if len(x_values) < k * 5:
            continue

        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(x_values)
            inertia = km.inertia_
            centers = sorted([c[0] for c in km.cluster_centers_])

            if len(centers) >= 2:
                sep = np.mean(np.diff(centers))
            else:
                sep = 1.0

            score = inertia / max(sep, 1.0)

            if best_score is None or score < best_score:
                best_score = score
                best_k = k
        except Exception:
            pass

    return best_k

# =========================================================
# 4) 속도 추적기
# =========================================================
class SpeedTracker:
    """
    차량별 bottom y 변화량(abs(dy)) 기반 EMA 속도 계산
    """
    def __init__(self, ema_alpha=0.3, spike_clamp=12.0):
        self.ema_alpha = ema_alpha
        self.spike_clamp = spike_clamp
        self.prev_bottom = {}
        self.ema_speed = {}
        self.traces = defaultdict(lambda: deque(maxlen=30))

    def update(self, track_id, bbox):
        cx, cy = bottom_center(bbox)

        if track_id in self.prev_bottom:
            _, prev_cy = self.prev_bottom[track_id]
            dy = abs(cy - prev_cy)
            dy = max(dy, MIN_VALID_DY)
            dy = min(dy, self.spike_clamp)
        else:
            dy = 0.0

        prev_speed = self.ema_speed.get(track_id, 0.0)
        speed = (self.ema_alpha * dy) + ((1 - self.ema_alpha) * prev_speed)

        self.prev_bottom[track_id] = (cx, cy)
        self.ema_speed[track_id] = speed
        self.traces[track_id].append((int(cx), int(cy)))

        return speed

    def get_trace(self, track_id):
        return list(self.traces.get(track_id, []))

# =========================================================
# 5) lane bootstrap 추정기
# =========================================================
class SimpleLaneBootstrap:
    """
    bottom center x 기준 간단 lane bootstrap
    """
    def __init__(self):
        self.sample_x = []
        self.lane_centers = None
        self.lane_count = 0

    def collect(self, x):
        self.sample_x.append(float(x))

    def maybe_fit(self, frame_id):
        if frame_id < LANE_BOOTSTRAP_FRAMES:
            return

        if len(self.sample_x) < LANE_MIN_SAMPLES:
            return

        if self.lane_centers is not None and (frame_id % LANE_UPDATE_INTERVAL != 0):
            return

        x_values = np.array(self.sample_x, dtype=np.float32).reshape(-1, 1)

        if len(x_values) < LANE_MIN_SAMPLES:
            return

        if SKLEARN_AVAILABLE:
            k = choose_lane_count_kmeans(self.sample_x, LANE_K_CANDIDATES)
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(x_values)
                centers = sorted([float(c[0]) for c in km.cluster_centers_])
                self.lane_centers = centers
                self.lane_count = len(centers)
            except Exception:
                self.lane_centers = None
                self.lane_count = 0
        else:
            xmin = float(np.min(x_values))
            xmax = float(np.max(x_values))
            mid = (xmin + xmax) / 2.0
            self.lane_centers = [mid - 50, mid + 50]
            self.lane_count = 2

    def assign_lane(self, x):
        if not self.lane_centers:
            return -1

        dists = [abs(x - c) for c in self.lane_centers]
        lane_id = int(np.argmin(dists))
        return lane_id

# =========================================================
# 6) YOLO 결과 -> track dict 변환
# =========================================================
def extract_tracks_from_result(result):
    tracks = []

    boxes = result.boxes
    if boxes is None:
        return tracks

    if boxes.id is None:
        return tracks

    ids = boxes.id.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()
    cls_arr = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(ids), dtype=int)
    conf_arr = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(ids), dtype=float)

    for tid, box, cls_id, conf in zip(ids, xyxy, cls_arr, conf_arr):
        if int(cls_id) not in CLASS_VEHICLE_IDS:
            continue

        x1, y1, x2, y2 = box.tolist()
        tracks.append({
            "track_id": int(tid),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "cls": int(cls_id),
            "conf": float(conf)
        })

    return tracks

# =========================================================
# 7) CSV 헤더
# =========================================================
CSV_HEADER = [
    "frame_id",
    "vehicle_count",
    "lane_count",

    "frame_accident_prediction",
    "recent_prediction_count",
    "frame_accident",
    "accident_locked",
    "accident_start_frame",

    "pair_id1",
    "pair_id2",

    "lane1",
    "lane2",
    "same_lane",

    "dist",
    "gap",

    "dist_drop",
    "gap_up",
    "vertical",
    "vertical_or_lane",

    "lane_break",
    "lane_break_acc",

    "impact_persist",
    "persist_evidence",
    "stop_evidence",
    "lane_break_evidence",
    "smoke_fire",

    "abnormal",
    "abnormal_stop",
    "abnormal_pose",

    "rear_core",
    "gap_weak",
    "gap_strong",
    "post_evidence",

    "weak_pair_candidate",
    "strong_pair_candidate",
    "pair_accident_candidate",

    "pair_repeat_candidate",
    "pair_consecutive_candidate",
    "repeat_count_window",
    "repeat_strong_candidate",

    "early_repeat_candidate",
    "early_repeat_count_window",
    "early_repeat_valid",

    "pair_score",
    "pair_high_score"
]

# =========================================================
# 8) 메인
# =========================================================
def main():
    if not os.path.exists(VIDEO_PATH):
        print("❌ 영상이 존재하지 않음:", VIDEO_PATH)
        return

    if not os.path.exists(MODEL_PATH):
        print("❌ 모델이 존재하지 않음:", MODEL_PATH)
        return

    model = YOLO(MODEL_PATH)
    detector = AccidentDetector()
    speed_tracker = SpeedTracker(ema_alpha=EMA_ALPHA, spike_clamp=SPEED_SPIKE_CLAMP)
    lane_bootstrap = SimpleLaneBootstrap()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ 영상 열기 실패")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0

    writer = cv2.VideoWriter(
        VIDEO_OUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_HEADER)

    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # -------------------------------------------------
            # YOLO + ByteTrack
            # -------------------------------------------------
            results = model.track(
                source=frame,
                persist=True,
                tracker=TRACKER_CONFIG,
                conf=CONF_THRES,
                iou=IOU_THRES,
                verbose=False
            )

            yolo_tracks = extract_tracks_from_result(results[0]) if results else []

            # -------------------------------------------------
            # speed / lane
            # -------------------------------------------------
            enriched_tracks = []
            boxes = {}
            speeds = {}
            lane_map = {}

            for tr in yolo_tracks:
                tid = tr["track_id"]
                bbox = tr["bbox"]

                cx, cy = bottom_center(bbox)

                if frame_id < LANE_BOOTSTRAP_FRAMES:
                    lane_bootstrap.collect(cx)

                speed = speed_tracker.update(tid, bbox)
                lane_id = lane_bootstrap.assign_lane(cx)

                enriched_tracks.append({
                    "track_id": tid,
                    "bbox": bbox,
                    "speed": speed,
                    "lane_id": lane_id,
                    "cls": tr["cls"],
                    "conf": tr["conf"]
                })

                boxes[tid] = bbox
                speeds[tid] = speed
                lane_map[tid] = lane_id

            lane_bootstrap.maybe_fit(frame_id)

            avg_speed = float(np.mean(list(speeds.values()))) if len(speeds) > 0 else 0.0

            analysis = {
                "boxes": boxes,
                "speeds": speeds,
                "avg_speed": avg_speed,
                "lane_map": lane_map,
                "smoke_fire_map": {}
            }

            # -------------------------------------------------
            # 사고 탐지
            # -------------------------------------------------
            accident_result = detector.update(frame_id, enriched_tracks, analysis)
            debug_info = detector.get_debug_info()

            frame_accident_prediction = debug_info.get("frame_accident_prediction", False)
            recent_prediction_count = debug_info.get("recent_prediction_count", 0)
            accident_locked = debug_info.get("accident_locked", False)
            accident_start_frame = debug_info.get("accident_start_frame", None)
            frame_accident = debug_info.get("accident", False)
            pair_debug = debug_info.get("pairs", [])

            # -------------------------------------------------
            # CSV 저장
            # -------------------------------------------------
            if len(pair_debug) == 0:
                csv_writer.writerow([
                    frame_id,
                    len(enriched_tracks),
                    lane_bootstrap.lane_count,

                    frame_accident_prediction,
                    recent_prediction_count,
                    frame_accident,
                    accident_locked,
                    accident_start_frame,

                    "", "",
                    "", "", "",
                    "", "",
                    "", "", "", "",
                    "", "",
                    "", "", "", "", "",
                    "", "", "",
                    "", "", "", "",
                    "", "", "",
                    "", "", "", "",
                    "", "", "",
                    "", ""
                ])
            else:
                for p in pair_debug:
                    pair_id1, pair_id2 = p.get("pair", ("", ""))

                    csv_writer.writerow([
                        frame_id,
                        len(enriched_tracks),
                        lane_bootstrap.lane_count,

                        frame_accident_prediction,
                        recent_prediction_count,
                        frame_accident,
                        accident_locked,
                        accident_start_frame,

                        pair_id1,
                        pair_id2,

                        p.get("lane1", ""),
                        p.get("lane2", ""),
                        p.get("same_lane", ""),

                        p.get("dist", ""),
                        p.get("gap", ""),

                        p.get("dist_drop", ""),
                        p.get("gap_up", ""),
                        p.get("vertical", ""),
                        p.get("vertical_or_lane", ""),

                        p.get("lane_break", ""),
                        p.get("lane_break_acc", ""),

                        p.get("impact_persist", ""),
                        p.get("persist_evidence", ""),
                        p.get("stop_evidence", ""),
                        p.get("lane_break_evidence", ""),
                        p.get("smoke_fire", ""),

                        p.get("abnormal", ""),
                        p.get("abnormal_stop", ""),
                        p.get("abnormal_pose", ""),

                        p.get("rear_core", ""),
                        p.get("gap_weak", ""),
                        p.get("gap_strong", ""),
                        p.get("post_evidence", ""),

                        p.get("weak_pair_candidate", ""),
                        p.get("strong_pair_candidate", ""),
                        p.get("pair_accident_candidate", ""),

                        p.get("pair_repeat_candidate", ""),
                        p.get("pair_consecutive_candidate", ""),
                        p.get("repeat_count_window", ""),
                        p.get("repeat_strong_candidate", ""),

                        p.get("early_repeat_candidate", ""),
                        p.get("early_repeat_count_window", ""),
                        p.get("early_repeat_valid", ""),

                        p.get("pair_score", ""),
                        p.get("pair_high_score", "")
                    ])

            # -------------------------------------------------
            # 디버그 시각화
            # -------------------------------------------------
            vis = frame.copy()

            panel_x = 10
            panel_y = 20
            line_h = 24

            info_lines = [
                f"Frame: {frame_id}",
                f"Vehicles: {len(enriched_tracks)}",
                f"Lane Count: {lane_bootstrap.lane_count}",
                f"Avg Speed: {avg_speed:.2f}",
                f"Prediction: {frame_accident_prediction}",
                f"Recent Pred Count: {recent_prediction_count}",
                f"Accident Locked: {accident_locked}",
                f"Accident Start: {accident_start_frame}",
            ]

            for i, txt in enumerate(info_lines):
                y = panel_y + i * line_h
                cv2.putText(vis, txt, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # 박스 / 라벨 / trace
            for tr in enriched_tracks:
                tid = tr["track_id"]
                x1, y1, x2, y2 = map(int, tr["bbox"])
                speed = tr["speed"]
                lane_id = tr["lane_id"]

                color = (0, 255, 0)
                if accident_locked:
                    color = (0, 0, 255)
                elif frame_accident_prediction:
                    color = (0, 165, 255)

                if DRAW_BOX:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                label = f"ID:{tid} S:{speed:.2f} L:{lane_id}"
                cv2.putText(
                    vis,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                if DRAW_TRACK:
                    trace = speed_tracker.get_trace(tid)
                    if len(trace) >= 2:
                        for j in range(1, len(trace)):
                            cv2.line(vis, trace[j - 1], trace[j], (255, 255, 0), 2)

            # 강한 pair 텍스트
            if DRAW_PAIR_TEXT and len(pair_debug) > 0:
                strong_rows = [
                    p for p in pair_debug
                    if p.get("strong_pair_candidate", False)
                    or p.get("pair_repeat_candidate", False)
                    or p.get("pair_high_score", False)
                    or p.get("early_repeat_valid", False)
                ]

                strong_rows = strong_rows[:6]
                start_y = height - 170

                for i, p in enumerate(strong_rows):
                    pair_id1, pair_id2 = p.get("pair", ("", ""))
                    txt = (
                        f"P({pair_id1},{pair_id2}) "
                        f"D:{p.get('dist', '')} G:{p.get('gap', '')} "
                        f"S:{p.get('strong_pair_candidate', False)} "
                        f"R:{p.get('pair_repeat_candidate', False)} "
                        f"E:{p.get('early_repeat_valid', False)} "
                        f"H:{p.get('pair_high_score', False)}"
                    )
                    cv2.putText(
                        vis,
                        txt,
                        (10, start_y + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2
                    )

            if frame_accident:
                cv2.putText(
                    vis,
                    "ACCIDENT LOCKED",
                    (width - 260, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )
            elif frame_accident_prediction:
                cv2.putText(
                    vis,
                    "ACCIDENT PREDICTION",
                    (width - 320, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 165, 255),
                    3
                )

            writer.write(vis)

            if SHOW_WINDOW:
                cv2.imshow("accident_debug_v5_5_2", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

            if frame_id % 50 == 0:
                print(
                    f"[Frame {frame_id}] "
                    f"vehicles={len(enriched_tracks)} "
                    f"pred={frame_accident_prediction} "
                    f"recent={recent_prediction_count} "
                    f"locked={accident_locked}"
                )

            frame_id += 1

    except Exception as e:
        print("❌ 실행 중 오류 발생")
        print(e)
        traceback.print_exc()

    finally:
        cap.release()
        writer.release()
        csv_file.close()
        cv2.destroyAllWindows()

        print("✅ 완료")
        print("CSV 저장:", CSV_PATH)
        print("영상 저장:", VIDEO_OUT_PATH)


if __name__ == "__main__":
    main()