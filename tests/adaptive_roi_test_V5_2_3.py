# ==========================================
# 파일명: adaptive_roi_test_V5_2_3.py
# 설명:
# adaptive ROI 테스트 코드
#
# 기능
# 1) 초기 100프레임 동안 y2 buffer 수집
# 2) 화면 높이 기준 중앙 60% ROI 계산
# 3) 초기 100프레임 동안 샘플 수집
# 4) bootstrap 종료 시 ROI 1회 확정 후 고정
# 5) 최소 span 보장
# 6) 화면에 ROI 선 시각화
# 7) CSV 로그 저장
#
# 목적
# - CCTV 고정 환경에서 화면 높이 기준 중앙 60% ROI가
#   안정적으로 고정 적용되는지 검증
# ==========================================

import os
import csv
import cv2
import traceback
from collections import deque
from datetime import datetime
from ultralytics import YOLO


# ==========================================
# Adaptive ROI 클래스
# ==========================================
class AdaptiveROI:
    def __init__(self, frame_height=720):
        # -----------------------------
        # 기본 설정
        # -----------------------------
        self.frame_height = frame_height

        # fallback 비율
        self.FALLBACK_Y1_RATIO = 0.20
        self.FALLBACK_Y2_RATIO = 0.80

        # bootstrap 설정
        self.BOOTSTRAP_FRAMES = 100

        # 초기 100프레임 동안만 수집되는 y2 buffer
        self.recent_y2_buffer = deque()

        self.roi_fixed = False
        self.fixed_roi_y1 = int(frame_height * self.FALLBACK_Y1_RATIO)
        self.fixed_roi_y2 = int(frame_height * self.FALLBACK_Y2_RATIO)

        # 최소 span 보장
        self.MIN_SPAN = 120

        # 화면 경계 여유
        self.TOP_MARGIN = 0
        self.BOTTOM_MARGIN = frame_height - 1

    # =========================================================
    # 유틸
    # =========================================================
    def _clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def update_y2_buffer(self, tracks):
        """
        현재 프레임의 차량 bbox 하단 y2들을 버퍼에 추가
        """
        frame_y2 = []

        for t in tracks:
            _, _, _, y2 = t["bbox"]
            frame_y2.append(int(y2))

        # 현재 프레임에 차량이 있으면 추가
        if len(frame_y2) > 0:
            self.recent_y2_buffer.extend(frame_y2)

    # =========================================================
    # 화면 기준 ROI 계산
    # =========================================================
    def _compute_raw_roi(self):
        """
        화면 전체 높이 기준 중앙 60% raw ROI 계산
        """
        sample_count = len(self.recent_y2_buffer)
        raw_y1 = int(self.frame_height * self.FALLBACK_Y1_RATIO)
        raw_y2 = int(self.frame_height * self.FALLBACK_Y2_RATIO)

        return {
            "raw_y1": raw_y1,
            "raw_y2": raw_y2,
            "sample_count": sample_count,
            "used_fallback": False
        }

    # =========================================================
    # 최소 span 보장
    # =========================================================
    def _ensure_min_span(self, y1, y2):
        """
        ROI가 너무 좁아지지 않도록 최소 높이 보장
        """
        span = y2 - y1

        if span >= self.MIN_SPAN:
            return y1, y2

        center = (y1 + y2) / 2.0
        half = self.MIN_SPAN / 2.0

        new_y1 = int(center - half)
        new_y2 = int(center + half)

        new_y1 = self._clamp(new_y1, self.TOP_MARGIN, self.BOTTOM_MARGIN)
        new_y2 = self._clamp(new_y2, self.TOP_MARGIN, self.BOTTOM_MARGIN)

        # clamp 이후 다시 span 부족하면 아래쪽 우선 보정
        if new_y2 - new_y1 < self.MIN_SPAN:
            need = self.MIN_SPAN - (new_y2 - new_y1)
            new_y1 = self._clamp(new_y1 - need, self.TOP_MARGIN, self.BOTTOM_MARGIN)

        if new_y2 - new_y1 < self.MIN_SPAN:
            need = self.MIN_SPAN - (new_y2 - new_y1)
            new_y2 = self._clamp(new_y2 + need, self.TOP_MARGIN, self.BOTTOM_MARGIN)

        return int(new_y1), int(new_y2)

    # =========================================================
    # ROI 업데이트
    # =========================================================
    def update(self, tracks, frame_id):
        """
        1) 초기 100프레임 동안만 y2 수집
        2) 100프레임 시점에 ROI 1회 확정
        3) 이후는 고정 ROI 반환
        """
        if self.roi_fixed:
            return {
                "roi_y1": self.fixed_roi_y1,
                "roi_y2": self.fixed_roi_y2,
                "raw_y1": self.fixed_roi_y1,
                "raw_y2": self.fixed_roi_y2,
                "sample_count": len(self.recent_y2_buffer),
                "used_fallback": False,
                "span": self.fixed_roi_y2 - self.fixed_roi_y1,
                "roi_fixed": True
            }

        if frame_id <= self.BOOTSTRAP_FRAMES:
            self.update_y2_buffer(tracks)

        roi_info = self._compute_raw_roi()
        raw_y1 = roi_info["raw_y1"]
        raw_y2 = roi_info["raw_y2"]

        final_y1, final_y2 = self._ensure_min_span(raw_y1, raw_y2)

        self.current_roi_y1 = final_y1
        self.current_roi_y2 = final_y2

        if frame_id >= self.BOOTSTRAP_FRAMES:
            self.fixed_roi_y1 = final_y1
            self.fixed_roi_y2 = final_y2
            self.roi_fixed = True

        return {
            "roi_y1": final_y1,
            "roi_y2": final_y2,
            "raw_y1": raw_y1,
            "raw_y2": raw_y2,
            "sample_count": roi_info["sample_count"],
            "used_fallback": roi_info["used_fallback"],
            "span": final_y2 - final_y1,
            "roi_fixed": self.roi_fixed
        }


# ==========================================
# 시각화
# ==========================================
def draw_tracks(frame, tracks):
    for t in tracks:
        tid = t["id"]
        x1, y1, x2, y2 = t["bbox"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID:{tid} y2:{y2}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


def draw_roi(frame, roi_info):
    h, w = frame.shape[:2]

    roi_y1 = roi_info["roi_y1"]
    roi_y2 = roi_info["roi_y2"]
    raw_y1 = roi_info["raw_y1"]
    raw_y2 = roi_info["raw_y2"]

    # raw ROI
    cv2.line(frame, (0, raw_y1), (w, raw_y1), (0, 0, 255), 1)
    cv2.line(frame, (0, raw_y2), (w, raw_y2), (0, 0, 255), 1)

    # final ROI
    cv2.line(frame, (0, roi_y1), (w, roi_y1), (255, 255, 0), 2)
    cv2.line(frame, (0, roi_y2), (w, roi_y2), (255, 255, 0), 2)

    lines = [
        f"RAW ROI: {raw_y1} ~ {raw_y2}",
        f"FINAL ROI: {roi_y1} ~ {roi_y2}",
        f"SAMPLES: {roi_info['sample_count']}",
        f"FALLBACK: {roi_info['used_fallback']}",
        f"SPAN: {roi_info['span']}",
        f"ROI FIXED: {roi_info.get('roi_fixed', False)}",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        y += 24


# ==========================================
# 로그
# ==========================================
def write_log_header(writer):
    writer.writerow([
        "frame_id",
        "vehicle_count",
        "raw_y1",
        "raw_y2",
        "roi_y1",
        "roi_y2",
        "span",
        "sample_count",
        "used_fallback",
        "roi_fixed"
    ])


def write_log_row(writer, frame_id, tracks, roi_info):
    writer.writerow([
        frame_id,
        len(tracks),
        roi_info["raw_y1"],
        roi_info["raw_y2"],
        roi_info["roi_y1"],
        roi_info["roi_y2"],
        roi_info["span"],
        roi_info["sample_count"],
        roi_info["used_fallback"],
        roi_info["roi_fixed"]
    ])


# ==========================================
# 메인
# ==========================================
print("🚀 ADAPTIVE ROI TEST 시작")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_congestion_2-1.mp4"
VIDEO_PATH = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_data/raw_video/test_video/test_normal_2.mp4"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")

OUTPUT_DIR = r"d:/Finalpj_tunnel_V3/smart_tunnel_V3_outputs/adaptive_roi_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, f"adaptive_roi_test_{timestamp}.mp4")
LOG_PATH = os.path.join(OUTPUT_DIR, f"adaptive_roi_log_{timestamp}.csv")

print("VIDEO_PATH:", VIDEO_PATH)
print("MODEL_PATH:", MODEL_PATH)
print("OUTPUT_DIR:", OUTPUT_DIR)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ 영상 열기 실패")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps, (width, height))

    model = YOLO(MODEL_PATH)
    roi_estimator = AdaptiveROI(frame_height=height)

    print("✅ 분석 시작")
    print("   └ 저장 영상:", VIDEO_OUT_PATH)
    print("   └ 저장 로그:", LOG_PATH)

    with open(LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        write_log_header(writer)

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            yolo_results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=0.25,
                verbose=False
            )

            tracks = []

            if yolo_results and len(yolo_results) > 0:
                r = yolo_results[0]

                if r.boxes is not None and r.boxes.id is not None:
                    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                    ids = r.boxes.id.cpu().numpy().astype(int)

                    for box, tid in zip(boxes_xyxy, ids):
                        x1, y1, x2, y2 = box.tolist()
                        tracks.append({
                            "id": int(tid),
                            "bbox": (int(x1), int(y1), int(x2), int(y2))
                        })

            roi_info = roi_estimator.update(tracks, frame_id)

            draw_tracks(frame, tracks)
            draw_roi(frame, roi_info)

            out.write(frame)
            write_log_row(writer, frame_id, tracks, roi_info)

            cv2.imshow("ADAPTIVE ROI TEST", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("✅ 종료 완료")
    print("   └ 영상:", VIDEO_OUT_PATH)
    print("   └ 로그:", LOG_PATH)
    print("   └ 폴더:", OUTPUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ 실행 중 오류:", e)
        traceback.print_exc()
