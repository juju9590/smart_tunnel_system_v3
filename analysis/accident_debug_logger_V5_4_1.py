# ==========================================
# 파일명: accident_debug_logger_V5_4.py
# 위치: analysis/
# 설명:
# - pipeline 코드는 수정하지 않고
# - analysis 폴더에서 별도 실행하여
# - V5.4 사고 디버그 CSV를 저장하는 스크립트
# - TrackAnalyzer + LaneTemplateEstimator 연결 버전
# ==========================================

import os
import sys
import csv
from datetime import datetime

import cv2
from ultralytics import YOLO


# =========================================================
# 0. 경로 설정
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 외부 데이터 폴더
DATA_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "smart_tunnel_V3_data")

# 외부 출력 폴더
OUTPUT_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "smart_tunnel_V3_outputs")

# V5.4 사고로직 폴더 경로
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "scr", "pipeline_V5_4(사고로직_개선)")
sys.path.append(PIPELINE_DIR)

# 출력 폴더
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "analysis_v5_4_1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 영상 경로 선택
# =========================================================
# VIDEO_PATH = os.path.join(DATA_ROOT, "raw_video", "test_video", "test_accident_1-1.mp4")
# VIDEO_PATH = os.path.join(DATA_ROOT, "raw_video", "test_video", "test_accident_2.mp4")
VIDEO_PATH = os.path.join(DATA_ROOT, "raw_video", "test_video", "test_accident_3.mp4")
# VIDEO_PATH = os.path.join(DATA_ROOT, "raw_video", "test_video", "test_congestion_2-1.mp4")  # 대조 혼잡

# 모델 경로
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")

# ByteTrack 설정 경로
TRACKER_CONFIG = "bytetrack.yaml"


# =========================================================
# 1. 파이프라인 모듈 import
# =========================================================
try:
    from traffic_accident_V5_4_1 import AccidentDetector
except Exception as e:
    raise ImportError(
        f"[오류] traffic_accident_V5_4 import 실패\n"
        f"PIPELINE_DIR 확인: {PIPELINE_DIR}\n"
        f"에러: {e}"
    )

try:
    from track_analyzer_V5_3 import TrackAnalyzer
except Exception as e:
    raise ImportError(
        f"[오류] track_analyzer_V5_3 import 실패\n"
        f"파일명/클래스명 확인 필요\n"
        f"에러: {e}"
    )

try:
    from lane_template_V5_3 import LaneTemplateEstimator
except Exception as e:
    raise ImportError(
        f"[오류] lane_template_V5_3 import 실패\n"
        f"파일명/클래스명 확인 필요\n"
        f"에러: {e}"
    )


# =========================================================
# 2. CSV 로거
# =========================================================
class AccidentDebugCSVLogger:
    def __init__(self, save_path):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.file = open(save_path, mode="w", newline="", encoding="utf-8-sig")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "frame_id",
            "frame_accident",
            "frame_acc_ratio",

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
            "impact_persist_strong",
            "smoke_fire",
            "flow_break",

            "abnormal",
            "abnormal_stop",

            "rear",

            "collision_sign",
            "abnormal_pose",
            "post_evidence",
            "group_count",

            "candidate_A",
            "candidate_B",
            "accident_candidate",

            "accident_score",
            "confirmed",

            "lane_count",
            "template_phase",
            "template_confirmed"
        ])

    def log(self, debug_info):
        frame_id = debug_info.get("frame_id", 0)
        frame_accident = debug_info.get("accident", False)
        frame_acc_ratio = debug_info.get("acc_ratio", 0.0)
        pairs = debug_info.get("pairs", [])

        lane_count = debug_info.get("lane_count", "")
        template_phase = debug_info.get("template_phase", "")
        template_confirmed = debug_info.get("template_confirmed", "")

        if not pairs:
            self.writer.writerow([
                frame_id,
                frame_accident,
                frame_acc_ratio,

                "", "",
                "", "", "",

                "", "",

                "", "", "", "",

                "", "",

                "", "", "", "",

                "", "",

                "",

                "", "", "", "",

                "", "", "",

                "", "",

                lane_count,
                template_phase,
                template_confirmed
            ])
            self.file.flush()
            return

        for pair in pairs:
            pair_key = pair.get("pair", ("", ""))
            pair_id1 = pair_key[0] if len(pair_key) > 0 else ""
            pair_id2 = pair_key[1] if len(pair_key) > 1 else ""

            self.writer.writerow([
                frame_id,
                frame_accident,
                frame_acc_ratio,

                pair_id1,
                pair_id2,

                pair.get("lane1", ""),
                pair.get("lane2", ""),
                pair.get("same_lane", False),

                pair.get("dist", 0.0),
                pair.get("gap", 0.0),

                pair.get("dist_drop", False),
                pair.get("gap_up", False),
                pair.get("vertical", False),
                pair.get("vertical_or_lane", False),

                pair.get("lane_break", False),
                pair.get("lane_break_acc", False),

                pair.get("impact_persist", False),
                pair.get("impact_persist_strong", False),
                pair.get("smoke_fire", False),
                pair.get("flow_break", False),

                pair.get("abnormal", False),
                pair.get("abnormal_stop", False),

                pair.get("rear", False),

                pair.get("collision_sign", False),
                pair.get("abnormal_pose", False),
                pair.get("post_evidence", False),
                pair.get("group_count", 0),

                pair.get("candidate_A", False),
                pair.get("candidate_B", False),
                pair.get("accident_candidate", False),

                pair.get("accident_score", 0),
                pair.get("confirmed", False),

                lane_count,
                template_phase,
                template_confirmed
            ])

        self.file.flush()

    def close(self):
        self.file.close()


# =========================================================
# 3. YOLO 결과 -> tracks 변환
# =========================================================
def convert_yolo_tracks(result):
    """
    반환 예시:
    [
        {"id": 3, "bbox": [x1,y1,x2,y2], "cls": 0, "conf": 0.88},
        ...
    ]
    """
    tracks = []

    if result.boxes is None or result.boxes.id is None:
        return tracks

    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id.cpu().numpy().astype(int)
    clss = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    for box, tid, cls_id, conf in zip(boxes, ids, clss, confs):
        x1, y1, x2, y2 = map(int, box.tolist())
        tracks.append({
            "id": int(tid),
            "bbox": [x1, y1, x2, y2],
            "cls": int(cls_id),
            "conf": float(conf)
        })

    return tracks


# =========================================================
# 4. 실행
# =========================================================
def main():
    print("🚀 V5.4 사고 디버그 CSV 추출 시작")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_ROOT:", DATA_ROOT)
    print("VIDEO_PATH:", VIDEO_PATH)
    print("VIDEO_EXISTS:", os.path.exists(VIDEO_PATH))

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 영상 없음: {VIDEO_PATH}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 없음: {MODEL_PATH}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"accident_debug_v5_4_{timestamp}.csv")

    model = YOLO(MODEL_PATH)
    analyzer = TrackAnalyzer()
    lane_estimator = LaneTemplateEstimator(output_dir=OUTPUT_DIR)
    accident_model = AccidentDetector()
    logger = AccidentDebugCSVLogger(csv_path)

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # YOLO + ByteTrack
            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CONFIG,
                verbose=False
            )

            if not results:
                logger.log({
                    "frame_id": frame_id,
                    "accident": False,
                    "acc_ratio": 0.0,
                    "pairs": [],
                    "lane_count": 0,
                    "template_phase": "NO_RESULT",
                    "template_confirmed": False
                })
                continue

            result = results[0]
            tracks = convert_yolo_tracks(result)

            # 1) 공통 분석기: boxes / speeds / avg_speed / history
            analysis = analyzer.update(frame_id, tracks)

            # 2) 차선 생성기: lane_map / lane_count / template 상태
            lane_result = lane_estimator.update(frame_id, analysis)

            # 3) 사고로직이 쓰기 쉽도록 하나로 합치기
            analysis.update(lane_result)

            # 4) 사고 판단
            accident_model.update(frame_id, tracks, analysis)
            debug_info = accident_model.get_debug_info()

            # frame-level lane 상태도 같이 저장
            debug_info["lane_count"] = analysis.get("lane_count", 0)
            debug_info["template_phase"] = analysis.get("template_phase", "")
            debug_info["template_confirmed"] = analysis.get("template_confirmed", False)

            # 디버그 출력
            if frame_id <= 5 or frame_id % 100 == 0:
                print("analysis keys:", list(analysis.keys()))
                print("lane_count:", analysis.get("lane_count"))
                print("template_phase:", analysis.get("template_phase"))
                print("template_confirmed:", analysis.get("template_confirmed"))
                print("lane_map sample:", dict(list(analysis.get("lane_map", {}).items())[:10]))

            # CSV 기록
            logger.log(debug_info)

            if frame_id % 100 == 0:
                print(f"  ... {frame_id} frame 처리 중")

    finally:
        cap.release()
        logger.close()

    print("✅ 완료")
    print(f"📁 저장 위치: {csv_path}")


if __name__ == "__main__":
    main()