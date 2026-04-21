# ==========================================
# 파일명: track_analyzer_V5_3.py
# 설명:
# 공통 추적 분석기 (Adaptive ROI 연동 버전)
# - track history 관리
# - 차량 속도 계산
# - bbox / speed / history 제공
# - ROI는 adaptive_roi 모듈에서 받아 사용
# ==========================================

import numpy as np


class TrackAnalyzer:
    def __init__(self):
        self.track_history = {}      # tid -> [(cx, cy), ...]
        self.prev_speeds = {}        # tid -> smoothed speed

        self.last_debug = {
            "vehicle_count": 0,
            "avg_speed": 0.0,
            "speeds": {},
            "boxes": {},
            "track_points": {},
        }

        self.frame_height = 720

        self.MAX_HISTORY = 60
        self.MAX_DY_JUMP = 40
        self.MAX_DX_JUMP = 60

        self.MAX_SPEED = 20
        self.SPEED_JUMP_LIMIT = 10

    # =========================================================
    # 유틸
    # =========================================================
    def _clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def _update_frame_height_from_tracks(self, tracks):
        max_y2 = 0
        for t in tracks:
            _, _, _, y2 = t["bbox"]
            max_y2 = max(max_y2, int(y2))

        if max_y2 > 0:
            self.frame_height = max(self.frame_height, int(max_y2 * 1.05))

    # =========================================================
    # tracks -> boxes / history / speeds
    # =========================================================
    def _update_tracks_and_speeds(self, tracks, roi_y1, roi_y2):
        boxes = {}
        speeds = {}
        track_points = {}

        for t in tracks:
            tid = t["id"]
            x1, y1, x2, y2 = t["bbox"]

            cx = int((x1 + x2) / 2)
            cy = int(y2)

            boxes[tid] = (x1, y1, x2, y2)
            track_points[tid] = (cx, cy)

            self.track_history.setdefault(tid, []).append((cx, cy))
            if len(self.track_history[tid]) > self.MAX_HISTORY:
                self.track_history[tid].pop(0)

            # ROI 밖이면 이전 속도 유지
            if cy < roi_y1 or cy > roi_y2:
                speeds[tid] = self.prev_speeds.get(tid, 0.0)
                continue

            if len(self.track_history[tid]) < 3:
                speed = self.prev_speeds.get(tid, 0.0)
                self.prev_speeds[tid] = speed
                speeds[tid] = speed
                continue

            xp, yp = self.track_history[tid][-2]
            xc, yc = self.track_history[tid][-1]

            dx = abs(xc - xp)
            dy = yc - yp

            if abs(dy) > self.MAX_DY_JUMP or dx > self.MAX_DX_JUMP:
                speed = self.prev_speeds.get(tid, 0.0)
                speeds[tid] = speed
                continue

            # 원근 보정: adaptive ROI 기준
            scale = (yc - roi_y1) / (max(roi_y2 - roi_y1, 1) + 1e-6)
            scale = self._clamp(scale, 0.1, 1.0)

            speed = abs(dy) / (scale + 0.15)
            speed *= (1.2 + scale)

            prev_speed = self.prev_speeds.get(tid, speed)

            if speed > self.MAX_SPEED:
                speed = prev_speed

            if abs(speed - prev_speed) > self.SPEED_JUMP_LIMIT:
                speed = prev_speed

            speed = 0.3 * speed + 0.7 * prev_speed
            speed = self._clamp(speed, 0, self.MAX_SPEED)

            self.prev_speeds[tid] = speed
            speeds[tid] = speed

        valid = [s for s in speeds.values() if s < self.MAX_SPEED]
        avg_speed = float(np.mean(valid)) if len(valid) > 0 else 0.0

        return boxes, speeds, avg_speed, track_points

    # =========================================================
    # 외부 호출
    # =========================================================
    def update(self, frame_id, tracks, roi_info=None):
        """
        roi_info:
            {
                "roi_y1": ...,
                "roi_y2": ...
            }
        """
        self._update_frame_height_from_tracks(tracks)

        # ROI가 아직 없으면 기본값 사용
        if roi_info is None:
            roi_y1 = int(self.frame_height * 0.20)
            roi_y2 = int(self.frame_height * 0.80)
        else:
            roi_y1 = roi_info["roi_y1"]
            roi_y2 = roi_info["roi_y2"]

        boxes, speeds, avg_speed, track_points = self._update_tracks_and_speeds(
            tracks,
            roi_y1,
            roi_y2
        )

        analysis = {
            "frame_id": frame_id,
            "vehicle_count": len(tracks),
            "boxes": boxes,
            "speeds": speeds,
            "avg_speed": round(avg_speed, 2),
            "track_points": track_points,
            "track_history": self.track_history,
            "frame_height": self.frame_height,
            "roi_y1": roi_y1,
            "roi_y2": roi_y2,
            "roi_fixed": roi_info["roi_fixed"] if roi_info else False,
        }

        self.last_debug = analysis
        return analysis

    def get_debug_info(self):
        return self.last_debug