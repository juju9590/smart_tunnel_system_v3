# ==========================================
# 파일명: pipeline_core_V5_3.py
# 설명:
# V5_3 통합 파이프라인
# - AdaptiveROI
# - TrackAnalyzer
# - LaneTemplateEstimator
# - TrafficState
# - AccidentDetector
# ==========================================

from adaptive_roi_V5_3 import AdaptiveROI
from track_analyzer_V5_3 import TrackAnalyzer
from lane_template_V5_3 import LaneTemplateEstimator
from traffic_state_V5_4 import TrafficState
from traffic_accident_V5_4_3 import AccidentDetector


class PipelineCore:
    def __init__(self, frame_height=720, lane_output_dir=None):
        self.roi_estimator = AdaptiveROI(frame_height=frame_height)
        self.track_analyzer = TrackAnalyzer()
        self.lane_estimator = LaneTemplateEstimator(output_dir=lane_output_dir)

        self.state_model = TrafficState()
        self.accident_model = AccidentDetector()

    def process(self, frame_id, tracks):
        """
        입력:
            tracks = [
                {"id": tid, "bbox": (x1, y1, x2, y2)},
                ...
            ]
        """

        # 1) Adaptive ROI 먼저 계산
        roi_info = self.roi_estimator.update(tracks, frame_id)

        # 2) ROI를 반영해서 공통 추적/속도 분석
        analysis = self.track_analyzer.update(frame_id, tracks, roi_info=roi_info)

        # 3) 차선 추정
        lane_result = self.lane_estimator.update(frame_id, analysis)

        # --------------------------------------------------
        # 4) state_model에서 쓰기 쉬운 roi_box 생성
        #    x는 전체 폭 사용, y는 AdaptiveROI 결과 사용
        # --------------------------------------------------
        roi_box = (
            0,
            int(roi_info["raw_y1"]),
            99999,
            int(roi_info["raw_y2"]),
        )

        # 5) 합치기
        merged_analysis = {
            **analysis,

            "roi_raw_y1": roi_info["raw_y1"],
            "roi_raw_y2": roi_info["raw_y2"],
            "roi_sample_count": roi_info["sample_count"],
            "roi_used_fallback": roi_info["used_fallback"],
            "roi_span": roi_info["span"],
            "roi_fixed": roi_info["roi_fixed"],

            # state_model에서 바로 사용할 ROI 박스
            "roi_box": roi_box,

            "lane_map": lane_result["lane_map"],
            "raw_lane_map": lane_result["raw_lane_map"],
            "lane_count": lane_result["lane_count"],
            "centerlines": lane_result["centerlines"],
            "lane_debug": lane_result["lane_debug"],
            "template_phase": lane_result["template_phase"],
            "template_confirmed": lane_result["template_confirmed"],
            "clusters_stage1": lane_result.get("clusters_stage1", []),
            "clusters_stage2": lane_result.get("clusters_stage2", []),
            "clusters": lane_result.get("clusters", []),
        }

        # 6) 상태 판단
        state_result = self.state_model.update(frame_id, tracks, merged_analysis)

        # 7) 사고 판단
        accident_result = self.accident_model.update(frame_id, tracks, merged_analysis)

        return {
            "analysis": merged_analysis,
            "state": state_result,
            "accident": accident_result,
        }