# ==========================================
# 파일명: pipeline_core_V5_5.py
# 설명:
# V5_5 통합 파이프라인
# - AdaptiveROI
# - TrackAnalyzer
# - LaneTemplateEstimator
# - TrafficState
# - AccidentDetector
#
# 수정 내용
# 1) AdaptiveROI.update()에 frame_width 전달
# 2) 중앙영역 차량 수 기준 ROI 자동설정을 지원
# ==========================================

from adaptive_roi_V5_5 import AdaptiveROI
from track_analyzer_V5_5 import TrackAnalyzer
from lane_template_V5_5 import LaneTemplateEstimator
from traffic_state_V5_5 import TrafficState
from traffic_accident_V5_5 import AccidentDetector


class PipelineCore:
    def __init__(self, frame_height=720, lane_output_dir=None):
        # ------------------------------------------
        # ROI 추정기
        # 중앙영역 차량 수 조건을 보고
        # fallback -> 자동설정 -> 고정 흐름으로 동작
        # ------------------------------------------
        self.roi_estimator = AdaptiveROI(frame_height=frame_height)

        # ------------------------------------------
        # 공통 추적/속도/기초 분석
        # ------------------------------------------
        self.track_analyzer = TrackAnalyzer()

        # ------------------------------------------
        # 차선 추정기
        # ------------------------------------------
        self.lane_estimator = LaneTemplateEstimator(output_dir=lane_output_dir)

        # ------------------------------------------
        # 상태 / 사고 판단 모델
        # ------------------------------------------
        self.state_model = TrafficState()
        self.accident_model = AccidentDetector()

    def process(self, frame_id, tracks, frame_width, cctv_name=None):
        """
        입력:
            frame_id : 현재 프레임 번호
            tracks = [
                {"id": tid, "bbox": (x1, y1, x2, y2)},
                ...
            ]
            frame_width : 현재 영상의 가로 길이

        이유:
            AdaptiveROI가 "중앙영역 차량 2대 이상" 조건을 판단하려면
            화면의 가로 길이(frame_width)가 필요함
        """

        # --------------------------------------------------
        # 1) Adaptive ROI 계산
        # 기존에는 frame_id 기준 bootstrap이었지만,
        # 이제는 중앙영역 차량 수 조건을 보고 시작하므로
        # frame_width도 함께 전달
        # --------------------------------------------------
        roi_info = self.roi_estimator.update(tracks, frame_id, frame_width)

        # --------------------------------------------------
        # 2) ROI를 반영해서 공통 추적/속도 분석
        # --------------------------------------------------
        analysis = self.track_analyzer.update(frame_id, tracks, roi_info=roi_info)

        # --------------------------------------------------
        # 3) 차선 추정
        # --------------------------------------------------
        # lane_template에서 같은 CCTV 이름을 읽을 수 있도록 전달        
        analysis["cctv_name"] = cctv_name
        lane_result = self.lane_estimator.update(frame_id, analysis)

        # --------------------------------------------------
        # 4) state_model에서 쓰기 쉬운 roi_box 생성
        # x는 전체 폭 사용, y는 AdaptiveROI 결과 사용
        # --------------------------------------------------
        roi_box = (
            0,
            int(roi_info["raw_y1"]),
            99999,
            int(roi_info["raw_y2"]),
        )

        # --------------------------------------------------
        # 5) 분석 결과 병합
        # --------------------------------------------------
        merged_analysis = {
            **analysis,

            "cctv_name": cctv_name,

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

        # --------------------------------------------------
        # 6) 상태 판단
        # --------------------------------------------------
        state_result = self.state_model.update(frame_id, tracks, merged_analysis)

        # --------------------------------------------------
        # 7) 사고 판단
        # --------------------------------------------------
        accident_result = self.accident_model.update(frame_id, tracks, merged_analysis)

        return {
            "analysis": merged_analysis,
            "state": state_result,
            "accident": accident_result,
        }