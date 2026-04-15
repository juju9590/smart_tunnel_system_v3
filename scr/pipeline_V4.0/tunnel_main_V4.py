# ==========================================
# tunnel_main_V4_FINAL.py
# V4 최종 메인
# - 영상 저장
# - frame 로그 저장
# - pair 로그 저장
# - 디버깅 표시 강화
# ==========================================

import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from pipeline_core_V4 import PipelineCore


# ==========================
# 설정
# ==========================
VIDEO_PATH = "../../data/raw_video/test_video/test_accident_1.mp4"
# VIDEO_PATH = "../../data/raw_video/test_video/test_congestion_2-1.mp4"
# VIDEO_PATH = "../../data/raw_video/test_video/test_normal_1.mp4"

MODEL_PATH = "../../scripts/runs/train/tunnel_final/weights/best.pt"

OUTPUT_DIR = "../../outputs/pipeline_v4.0"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_name = os.path.basename(VIDEO_PATH).lower()

if "accident" in video_name:
    state_name = "accident"
    gt_state = "ACCIDENT"
    gt_accident = True
elif "congestion" in video_name:
    state_name = "congestion"
    gt_state = "CONGESTION"
    gt_accident = False
elif "normal" in video_name:
    state_name = "normal"
    gt_state = "NORMAL"
    gt_accident = False
elif "jam" in video_name:
    state_name = "jam"
    gt_state = "JAM"
    gt_accident = False
else:
    state_name = "unknown"
    gt_state = "UNKNOWN"
    gt_accident = ""

VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, f"{state_name}_{timestamp}.mp4")
FRAME_LOG_PATH = os.path.join(OUTPUT_DIR, f"{state_name}_{timestamp}_frame.csv")
PAIR_LOG_PATH = os.path.join(OUTPUT_DIR, f"{state_name}_{timestamp}_pair.csv")


def get_lane(cx, frame_width):
    return 1 if cx < frame_width // 2 else 2


def safe_round(value, ndigits=2):
    try:
        return round(float(value), ndigits)
    except Exception:
        return value


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ 영상 열기 실패:", VIDEO_PATH)
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 20

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        VIDEO_OUT_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    model = YOLO(MODEL_PATH)
    pipeline = PipelineCore()

    frame_id = 0

    # -----------------------------
    # 프레임 로그
    # -----------------------------
    frame_log_file = open(FRAME_LOG_PATH, "w", newline="", encoding="utf-8-sig")
    frame_writer = csv.writer(frame_log_file)

    frame_writer.writerow([
        "frame_id",
        "video_name",
        "gt_state",
        "gt_accident",
        "vehicle_count",
        "vehicle_ids",
        "state",
        "raw_state",
        "accident",
        "accident_hint",
        "accident_candidate",
        "acc_ratio",
        "avg_speed",
        "speed_std",
        "roi_vehicle_count",
        "flow_break",
        "fragment",
        "abnormal",
        "stop_confirm"
    ])

    # -----------------------------
    # pair 로그
    # -----------------------------
    pair_log_file = open(PAIR_LOG_PATH, "w", newline="", encoding="utf-8-sig")
    pair_writer = csv.writer(pair_log_file)

    pair_writer.writerow([
        "frame_id",
        "id1", "id2",
        "cx1", "cy1", "cx2", "cy2",
        "dist", "s1", "s2", "gap", "iou",
        "lane1", "lane2",
        "prev_dist", "prev_gap",
        "dist_drop", "gap_up", "vertical",
        "lane_break", "stop_confirm", "abnormal",
        "rear", "side", "lane_break_acc",
        "accident", "final"
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        h, w, _ = frame.shape

        # -----------------------------
        # YOLO + ByteTrack
        # -----------------------------
        results = model.track(
            frame,
            persist=False,
            conf=0.25,
            iou=0.5,
            tracker="bytetrack.yaml"
        )

        tracks = []
        vehicle_ids = []

        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue

            for box, tid in zip(r.boxes.xyxy, r.boxes.id):
                x1, y1, x2, y2 = map(int, box)
                tid = int(tid)

                tracks.append({
                    "id": tid,
                    "bbox": (x1, y1, x2, y2)
                })
                vehicle_ids.append(tid)

        # -----------------------------
        # 파이프라인 실행
        # -----------------------------
        result = pipeline.process(frame_id, frame, tracks)

        state = result["state"]
        raw_state = result["raw_state"]
        accident = result["accident"]
        avg_speed = result["avg_speed"]
        speed_std = result["speed_std"]
        accident_hint = result["accident_hint"]
        accident_candidate = result["accident_candidate"]
        acc_ratio = result["acc_ratio"]

        state_result = result.get("state_result", {})
        accident_result = result.get("accident_result", None)

        roi_vehicle_count = state_result.get("vehicle_count", 0)
        flow_break = False
        fragment = False
        abnormal = False
        stop_confirm = False

        if accident_result is not None:
            flow_break = accident_result.get("flow_break", False)
            fragment = accident_result.get("fragment", False)
            abnormal = accident_result.get("abnormal", False)
            stop_confirm = accident_result.get("stop_confirm", False)

        # -----------------------------
        # ROI 표시
        # -----------------------------
        roi_y1 = state_result.get("roi_y1", int(h * 0.3))
        roi_y2 = state_result.get("roi_y2", int(h * 0.8))

        cv2.line(frame, (0, roi_y1), (w, roi_y1), (0, 255, 255), 2)
        cv2.line(frame, (0, roi_y2), (w, roi_y2), (0, 255, 255), 2)

        # -----------------------------
        # 차량 시각화
        # -----------------------------
        speeds_map = state_result.get("speeds", {})
        lanes_map = {}

        if accident_result is not None:
            lanes_map = accident_result.get("lanes", {})

        # 사고 확정 pair 모아두기
        final_accident_ids = set()
        suspect_accident_ids = set()

        if accident_result is not None:
            for pair in accident_result.get("pairs", []):
                if pair.get("final", False):
                    final_accident_ids.add(pair["id1"])
                    final_accident_ids.add(pair["id2"])
                elif pair.get("accident", False):
                    suspect_accident_ids.add(pair["id1"])
                    suspect_accident_ids.add(pair["id2"])

        for t in tracks:
            tid = t["id"]
            x1, y1, x2, y2 = t["bbox"]

            speed = safe_round(speeds_map.get(tid, 0), 1)
            lane = lanes_map.get(tid, get_lane(int((x1 + x2) / 2), w))

            color = (0, 255, 0)

            if tid in final_accident_ids:
                color = (0, 0, 255)
            elif tid in suspect_accident_ids:
                color = (0, 255, 255)
            elif state == "JAM":
                color = (255, 0, 0)
            elif state == "CONGESTION":
                color = (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"ID:{tid} S:{speed} L:{lane}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

        # -----------------------------
        # 사고 pair 연결선 표시
        # -----------------------------
        if accident_result is not None:
            for pair in accident_result.get("pairs", []):
                if pair.get("final", False):
                    cx1, cy1 = pair["cx1"], pair["cy1"]
                    cx2, cy2 = pair["cx2"], pair["cy2"]
                    cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 3)

                    mx = int((cx1 + cx2) / 2)
                    my = int((cy1 + cy2) / 2)
                    cv2.putText(
                        frame,
                        "ACCIDENT",
                        (mx, my - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        3
                    )

        # -----------------------------
        # 상단 요약 표시
        # -----------------------------
        cv2.putText(frame, f"STATE: {state}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(frame, f"RAW: {raw_state}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 2)

        cv2.putText(frame, f"AVG_SPEED: {avg_speed:.2f}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"SPEED_STD: {speed_std:.2f}", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"VEHICLES: {len(tracks)} / ROI: {roi_vehicle_count}", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame, f"HINT:{accident_hint} CAND:{accident_candidate} RATIO:{acc_ratio:.2f}", (20, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        cv2.putText(frame, f"FLOW:{flow_break} FRAG:{fragment} ABN:{abnormal} STOP:{stop_confirm}", (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        if accident:
            cv2.putText(frame, "!!! FINAL ACCIDENT !!!", (w // 2 - 180, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        # -----------------------------
        # 영상 저장
        # -----------------------------
        out.write(frame)

        # -----------------------------
        # frame 로그 저장
        # -----------------------------
        frame_writer.writerow([
            frame_id,
            video_name,
            gt_state,
            gt_accident,
            len(tracks),
            vehicle_ids,
            state,
            raw_state,
            accident,
            accident_hint,
            accident_candidate,
            acc_ratio,
            safe_round(avg_speed, 2),
            safe_round(speed_std, 2),
            roi_vehicle_count,
            flow_break,
            fragment,
            abnormal,
            stop_confirm
        ])

        # -----------------------------
        # pair 로그 저장
        # -----------------------------
        if accident_result is not None:
            for pair in accident_result.get("pairs", []):
                pair_writer.writerow([
                    frame_id,
                    pair.get("id1"),
                    pair.get("id2"),
                    pair.get("cx1"),
                    pair.get("cy1"),
                    pair.get("cx2"),
                    pair.get("cy2"),
                    pair.get("dist"),
                    pair.get("s1"),
                    pair.get("s2"),
                    pair.get("gap"),
                    pair.get("iou"),
                    pair.get("lane1"),
                    pair.get("lane2"),
                    pair.get("prev_dist"),
                    pair.get("prev_gap"),
                    pair.get("dist_drop"),
                    pair.get("gap_up"),
                    pair.get("vertical"),
                    pair.get("lane_break"),
                    pair.get("stop_confirm"),
                    pair.get("abnormal"),
                    pair.get("rear"),
                    pair.get("side"),
                    pair.get("lane_break_acc"),
                    pair.get("accident"),
                    pair.get("final")
                ])

        cv2.imshow("Smart Tunnel V4", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    frame_log_file.close()
    pair_log_file.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("✅ 완료")
    print("🎥 영상:", VIDEO_OUT_PATH)
    print("📝 프레임 로그:", FRAME_LOG_PATH)
    print("📝 페어 로그:", PAIR_LOG_PATH)


if __name__ == "__main__":
    main()