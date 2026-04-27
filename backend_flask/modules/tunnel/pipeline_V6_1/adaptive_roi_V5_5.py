# ==========================================
# 파일명: adaptive_roi_V5_5.py
# 설명:
# 파이프라인용 Adaptive ROI 모듈
#
# 기능
# 1) 시작 시 fallback ROI 사용
# 2) 중앙 영역 차량 2대 이상일 때만 y2 buffer 수집 시작
# 3) percentile(20/80) 기반 ROI 자동 계산
# 4) 충분히 수집되면 ROI 1회 확정 후 고정
# 5) CCTV 변경 시 다시 초기화
# 6) ROI 최소 높이를 화면의 60%로 보장
# ==========================================

import numpy as np
from collections import deque


class AdaptiveROI:
    def __init__(self, frame_height=720):
        # -----------------------------------------------------
        # 기본 프레임 높이
        # -----------------------------------------------------
        self.frame_height = frame_height

        # -----------------------------------------------------
        # fallback ROI 비율
        # 차가 아직 부족하거나 샘플이 없을 때 사용하는 기본 ROI
        # -----------------------------------------------------
        self.FALLBACK_Y1_RATIO = 0.20
        self.FALLBACK_Y2_RATIO = 0.80

        # -----------------------------------------------------
        # 중앙영역 조건 설정
        # "화면 중앙에 차량이 2대 이상 있을 때부터 ROI 자동설정 시작"
        # 를 구현하기 위한 기준값
        # -----------------------------------------------------
        self.CENTER_X1_RATIO = 0.25
        self.CENTER_X2_RATIO = 0.75
        self.CENTER_Y1_RATIO = 0.20
        self.CENTER_Y2_RATIO = 0.85

        self.CENTER_VEHICLE_COUNT_THR = 2   # 중앙영역 차량 2대 이상일 때만 시작

        # -----------------------------------------------------
        # 샘플 수집 상태
        # SAMPLE_START = False 인 동안은 fallback ROI만 사용
        # 조건 만족 시 True로 바뀌고, 그때부터 y2 샘플 누적 시작
        # -----------------------------------------------------
        self.SAMPLE_START = False
        self.SAMPLE_LOCK_FRAMES = 30        # ROI 확정을 위한 최소 샘플 프레임 수
        self.sample_frame_count = 0         # 실제 샘플을 누적한 프레임 수

        # -----------------------------------------------------
        # y2 buffer
        # 차량 bbox 하단 y2를 계속 누적
        # -----------------------------------------------------
        self.recent_y2_buffer = deque()

        # -----------------------------------------------------
        # ROI 고정 상태
        # 한 번 확정되면 고정
        # -----------------------------------------------------
        self.roi_fixed = False
        self.fixed_roi_y1 = int(frame_height * self.FALLBACK_Y1_RATIO)
        self.fixed_roi_y2 = int(frame_height * self.FALLBACK_Y2_RATIO)

        # -----------------------------------------------------
        # percentile 기준
        # 중앙 60% 영역을 쓰기 위해 20 / 80 percentile 사용
        # -----------------------------------------------------
        self.LOW_PERCENTILE = 20
        self.HIGH_PERCENTILE = 80

        # -----------------------------------------------------
        # 표본 부족 기준
        # 샘플이 너무 적으면 percentile ROI를 믿지 않고 fallback 사용
        # -----------------------------------------------------
        self.MIN_SAMPLES = 20

        # -----------------------------------------------------
        # ROI 최소 높이 보장
        # ROI가 너무 얇아지지 않도록 화면 높이의 60% 이상 유지
        # -----------------------------------------------------
        self.MIN_SPAN_RATIO = 0.60
        self.MIN_SPAN = int(frame_height * self.MIN_SPAN_RATIO)

        # -----------------------------------------------------
        # 화면 경계
        # -----------------------------------------------------
        self.TOP_MARGIN = 0
        self.BOTTOM_MARGIN = frame_height - 1

    # =========================================================
    # 유틸
    # =========================================================
    def _clamp(self, v, lo, hi):
        """값을 화면 범위 안으로 강제 제한"""
        return max(lo, min(hi, v))

    # =========================================================
    # 중앙영역 차량 수 계산
    # =========================================================
    def _count_center_vehicles(self, tracks, frame_width):
        """
        중앙영역 안에 들어온 차량 수를 셈
        기준점은 bbox의 bottom-center(cx, y2)
        """
        center_x1 = int(frame_width * self.CENTER_X1_RATIO)
        center_x2 = int(frame_width * self.CENTER_X2_RATIO)
        center_y1 = int(self.frame_height * self.CENTER_Y1_RATIO)
        center_y2 = int(self.frame_height * self.CENTER_Y2_RATIO)

        count = 0

        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            cx = int((x1 + x2) / 2)
            by = int(y2)

            if center_x1 <= cx <= center_x2 and center_y1 <= by <= center_y2:
                count += 1

        return count

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
        샘플이 부족하면 fallback ROI 사용
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
    def update(self, tracks, frame_id, frame_width):
        """
        동작 순서
        1) 시작 시 fallback ROI 사용
        2) 중앙영역 차량 2대 이상이면 샘플 수집 시작
        3) 샘플 수집 후 percentile 기반 ROI 계산
        4) 충분히 수집되면 ROI 고정
        5) 이후는 고정 ROI만 반환
        """

        # -----------------------------------------------------
        # 이미 ROI가 확정된 경우
        # 고정 ROI 그대로 반환
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # 중앙영역 차량 수 계산
        # -----------------------------------------------------
        center_vehicle_count = self._count_center_vehicles(tracks, frame_width)

        # -----------------------------------------------------
        # 아직 샘플링 시작 전이면
        # 중앙영역 차량 수가 기준 이상일 때만 샘플링 시작
        # -----------------------------------------------------
        if not self.SAMPLE_START:
            if center_vehicle_count >= self.CENTER_VEHICLE_COUNT_THR:
                self.SAMPLE_START = True

        # -----------------------------------------------------
        # 샘플링 시작 후에만 y2 누적
        # -----------------------------------------------------
        if self.SAMPLE_START:
            self.update_y2_buffer(tracks)
            self.sample_frame_count += 1

        # -----------------------------------------------------
        # 아직 샘플링 시작 전이면 fallback ROI 유지
        # -----------------------------------------------------
        if not self.SAMPLE_START:
            raw_y1 = int(self.frame_height * self.FALLBACK_Y1_RATIO)
            raw_y2 = int(self.frame_height * self.FALLBACK_Y2_RATIO)
            final_y1, final_y2 = self._ensure_min_span(raw_y1, raw_y2)

            return {
                "roi_y1": final_y1,
                "roi_y2": final_y2,
                "raw_y1": raw_y1,
                "raw_y2": raw_y2,
                "sample_count": 0,
                "used_fallback": True,
                "span": final_y2 - final_y1,
                "roi_fixed": False
            }

        # -----------------------------------------------------
        # 샘플링이 시작된 뒤에는 raw ROI 계산
        # -----------------------------------------------------
        roi_info = self._compute_raw_roi()
        raw_y1 = roi_info["raw_y1"]
        raw_y2 = roi_info["raw_y2"]

        final_y1, final_y2 = self._ensure_min_span(raw_y1, raw_y2)

        # -----------------------------------------------------
        # 충분히 수집되면 ROI 확정
        # -----------------------------------------------------
        if self.sample_frame_count >= self.SAMPLE_LOCK_FRAMES:
            self.fixed_roi_y1 = final_y1
            self.fixed_roi_y2 = final_y2
            self.roi_fixed = True

        return {
            "roi_y1": final_y1,
            "roi_y2": final_y2,
            "raw_y1": raw_y1,
            "raw_y2": raw_y2,
            "sample_count": roi_info["sample_count"],
            "used_fallback": roi_info["used_fallback"] if self.SAMPLE_START else True,
            "span": final_y2 - final_y1,
            "roi_fixed": self.roi_fixed
        }