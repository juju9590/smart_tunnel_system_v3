# ==========================================
# 파일명: adaptive_roi_V5_2_3.py
# 설명:
# 파이프라인용 Adaptive ROI 모듈
#
# 기능
# 1) 초기 100프레임 동안 y2 buffer 수집
# 2) percentile(20/80) 기반 ROI 자동 계산
# 3) 표본 부족 시 fallback 사용
# 4) bootstrap 종료 시 ROI 1회 확정 후 고정
# 5) ROI 최소 높이를 화면의 60%로 보장
# ==========================================

import numpy as np
from collections import deque


class AdaptiveROI:
    def __init__(self, frame_height=720):
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

        # percentile 기준: 중앙 60%
        self.LOW_PERCENTILE = 20
        self.HIGH_PERCENTILE = 80

        # 표본 부족 기준
        self.MIN_SAMPLES = 20

        # ROI 최소 높이를 화면 높이의 60%로 강제
        self.MIN_SPAN_RATIO = 0.60
        self.MIN_SPAN = int(frame_height * self.MIN_SPAN_RATIO)

        # 화면 경계
        self.TOP_MARGIN = 0
        self.BOTTOM_MARGIN = frame_height - 1

    # =========================================================
    # 유틸
    # =========================================================
    def _clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    # =========================================================
    # y2 수집
    # =========================================================
    def update_y2_buffer(self, tracks):
        """
        현재 프레임의 차량 bbox 하단 y2들을 buffer에 추가
        """
        frame_y2 = []

        for t in tracks:
            _, _, _, y2 = t["bbox"]
            frame_y2.append(int(y2))

        if len(frame_y2) > 0:
            self.recent_y2_buffer.extend(frame_y2)

    # =========================================================
    # raw ROI 계산
    # =========================================================
    def _compute_raw_roi(self):
        """
        수집된 y2 분포를 바탕으로 raw ROI 계산
        """
        y2_values = list(self.recent_y2_buffer)

        if len(y2_values) < self.MIN_SAMPLES:
            raw_y1 = int(self.frame_height * self.FALLBACK_Y1_RATIO)
            raw_y2 = int(self.frame_height * self.FALLBACK_Y2_RATIO)

            return {
                "raw_y1": raw_y1,
                "raw_y2": raw_y2,
                "sample_count": len(y2_values),
                "used_fallback": True
            }

        p_low = np.percentile(y2_values, self.LOW_PERCENTILE)
        p_high = np.percentile(y2_values, self.HIGH_PERCENTILE)

        raw_y1 = int(p_low)
        raw_y2 = int(p_high)

        return {
            "raw_y1": raw_y1,
            "raw_y2": raw_y2,
            "sample_count": len(y2_values),
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

        if new_y2 - new_y1 < self.MIN_SPAN:
            need = self.MIN_SPAN - (new_y2 - new_y1)
            new_y1 = self._clamp(new_y1 - need, self.TOP_MARGIN, self.BOTTOM_MARGIN)

        if new_y2 - new_y1 < self.MIN_SPAN:
            need = self.MIN_SPAN - (new_y2 - new_y1)
            new_y2 = self._clamp(new_y2 + need, self.TOP_MARGIN, self.BOTTOM_MARGIN)

        return int(new_y1), int(new_y2)

    # =========================================================
    # 외부 호출
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