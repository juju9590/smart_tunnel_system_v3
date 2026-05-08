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

import cv2
import numpy as np

from adaptive_roi_V5_5 import AdaptiveROI
from track_analyzer_V5_5 import TrackAnalyzer
from lane_template_V5_5 import LaneTemplateEstimator
from traffic_state_V5_5 import TrafficState
from traffic_accident_V6 import AccidentDetector


SANGJU_DEMO_VIDEO_FILENAME = "accident_tunnel_sangju.mp4"


def detect_fire_like_region(frame):
    """
    발표 시연용 최소 휴리스틱.
    별도 fire/smoke 모델 없이 밝고 채도가 높은 red/orange/yellow 영역을
    smoke_fire_map 입력 후보로만 사용한다.
    """
    result = {
        "fire_candidate": False,
        "area_ratio": 0.0,
        "max_area": 0.0,
        "bbox": [],
    }

    if frame is None:
        return result

    h, w = frame.shape[:2]
    frame_area = max(float(h * w), 1.0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # OpenCV HSV hue range: 0~179. Red wraps around 0.
    red_low = cv2.inRange(hsv, np.array([0, 70, 105]), np.array([12, 255, 255]))
    red_high = cv2.inRange(hsv, np.array([170, 70, 105]), np.array([179, 255, 255]))
    orange_yellow = cv2.inRange(hsv, np.array([13, 60, 115]), np.array([45, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(red_low, red_high), orange_yellow)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 220.0
    valid_contours = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area >= min_contour_area:
            valid_contours.append((area, contour))

    if not valid_contours:
        return result

    total_area = sum(area for area, _ in valid_contours)
    max_area, max_contour = max(valid_contours, key=lambda item: item[0])
    x, y, bw, bh = cv2.boundingRect(max_contour)
    area_ratio = total_area / frame_area

    result.update({
        "fire_candidate": bool(area_ratio >= 0.0012 or max_area >= 650.0),
        "area_ratio": round(float(area_ratio), 6),
        "max_area": round(float(max_area), 2),
        "bbox": [int(x), int(y), int(bw), int(bh)],
    })
    return result


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

    def process(
        self,
        frame_id,
        tracks,
        frame_width,
        cctv_name=None,
        frame=None,
        source_type=None,
        source_name=None,
        demo_video_filename=None,
    ):
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
            "source_type": source_type or "cctv",
            "source_name": source_name or cctv_name,
            "demo_video_filename": demo_video_filename,

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
            "target_lane_count": lane_result.get(
                "target_lane_count",
                getattr(self.lane_estimator, "manual_lane_count", None)
            ),
            "clusters_stage1": lane_result.get("clusters_stage1", []),
            "clusters_stage2": lane_result.get("clusters_stage2", []),
            "clusters": lane_result.get("clusters", []),
            
        }

        fire_heuristic_enabled = (
            source_type == "demo_video"
            and demo_video_filename == SANGJU_DEMO_VIDEO_FILENAME
        )
        fire_debug = detect_fire_like_region(frame) if fire_heuristic_enabled else {
            "fire_candidate": False,
            "area_ratio": 0.0,
            "max_area": 0.0,
            "bbox": [],
        }

        merged_analysis["smoke_fire_map"] = (
            {"fire_like_region": True}
            if bool(fire_debug.get("fire_candidate", False))
            else {}
        )
        merged_analysis["smoke_fire_candidate"] = bool(fire_debug.get("fire_candidate", False))
        merged_analysis["smoke_fire_area_ratio"] = float(fire_debug.get("area_ratio", 0.0) or 0.0)
        merged_analysis["smoke_fire_max_area"] = float(fire_debug.get("max_area", 0.0) or 0.0)
        merged_analysis["smoke_fire_bbox"] = fire_debug.get("bbox", [])
        merged_analysis["smoke_fire_debug"] = fire_debug

        # --------------------------------------------------
        # 6) 상태 판단
        # --------------------------------------------------
        state_result = self.state_model.update(frame_id, tracks, merged_analysis)

        # --------------------------------------------------
        # 7) 사고 판단
        # --------------------------------------------------
        # 사고 탐지는 혼잡/정체 상태를 알아야 정체성 고정 셀과
        # 대형차 가림을 방어할 수 있다. 기존 merged_analysis를 복사해
        # state 결과만 보강해서 전달하면 전체 pipeline 구조는 유지된다.
        accident_analysis = dict(merged_analysis)

        if isinstance(state_result, dict):
            state_debug = state_result.get("debug", {})
            accident_analysis["traffic_state"] = state_result.get("state", "NORMAL")
            accident_analysis["state_avg_speed"] = state_debug.get("state_speed", 0.0)
            accident_analysis["traffic_buffer_avg_speed"] = state_debug.get("buffer_avg_speed", 0.0)
        else:
            accident_analysis["traffic_state"] = str(state_result)

        accident_result = self.accident_model.update(frame_id, tracks, accident_analysis)

        return {
            "analysis": merged_analysis,
            "state": state_result,
            "accident": accident_result,
        }
