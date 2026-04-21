# ==========================================
# 파일명: pipeline_core_V5_2.py
# 설명:
# V5_2 파이프라인 코어
# - TrackAnalyzer 호출
# - TrafficState 호출
# - AccidentDetector(V5_2) 호출
# - 최종 결과 dict 반환
# ==========================================

from track_analyzer_V5_2 import TrackAnalyzer
from traffic_state_V5_1 import TrafficState
from traffic_accident_V5_2_3 import AccidentDetector


class PipelineCore:
    def __init__(self):
        # -----------------------------
        # 공통 기반 분석기
        # -----------------------------
        self.track_analyzer = TrackAnalyzer()

        # -----------------------------
        # 상태 / 사고 판단기
        # -----------------------------
        self.state_model = TrafficState()
        self.accident_model = AccidentDetector()

    def process(self, frame_id, tracks):
        """
        입력:
            frame_id : 현재 프레임 번호
            tracks   : [
                {"id": 1, "bbox": (x1, y1, x2, y2)},
                ...
            ]

        출력:
            result : dict
        """

        # ==================================================
        # 1) 공통 분석
        # ==================================================
        analysis = self.track_analyzer.update(frame_id, tracks)

        boxes = analysis.get("boxes", {})
        speeds = analysis.get("speeds", {})
        avg_speed = float(analysis.get("avg_speed", 0.0))
        lane_map = analysis.get("lane_map", {})
        raw_lane_map = analysis.get("raw_lane_map", {})
        lane_count = int(analysis.get("lane_count", 0))
        centerlines = analysis.get("centerlines", [])
        vehicle_count = int(analysis.get("vehicle_count", len(tracks)))

        # ==================================================
        # 2) 상태 판단
        # ==================================================
        state_result = self.state_model.update(
            frame_id=frame_id,
            tracks=tracks,
            analysis=analysis
        )

        state = state_result.get("state", "NORMAL")

        # ==================================================
        # 3) 사고 판단
        # ==================================================
        accident_result = self.accident_model.update(
            frame_id=frame_id,
            tracks=tracks,
            analysis=analysis
        )

        accident_flag = bool(accident_result.get("accident", False))
        acc_ratio = float(accident_result.get("acc_ratio", 0.0))

        accident_debug = self.accident_model.get_debug_info()

        # ==================================================
        # 4) 최종 반환
        # ==================================================
        result = {
            # -----------------------------
            # 메인 판단 결과
            # -----------------------------
            "state": state,
            "accident": accident_flag,
            "acc_ratio": acc_ratio,

            # -----------------------------
            # 공통 분석 결과
            # -----------------------------
            "vehicle_count": vehicle_count,
            "avg_speed": avg_speed,
            "boxes": boxes,
            "speeds": speeds,

            # -----------------------------
            # 차선 정보
            # -----------------------------
            "lane_map": lane_map,
            "raw_lane_map": raw_lane_map,
            "lane_count": lane_count,
            "centerlines": centerlines,

            # -----------------------------
            # 사고 디버그
            # -----------------------------
            "accident_debug": accident_debug,
        }

        return result