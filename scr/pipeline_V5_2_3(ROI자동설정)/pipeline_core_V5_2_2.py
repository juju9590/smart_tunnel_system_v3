from track_analyzer_V5_2_1 import TrackAnalyzer
from scr.pipeline_V5_2_1.lane_template_V5_2_2 import LaneTemplateEstimator
from traffic_state_V5_1 import TrafficState
from traffic_accident_V5_2_3 import AccidentDetector


class PipelineCore:
    def __init__(self):
        self.track_analyzer = TrackAnalyzer()
        self.lane_estimator = LaneTemplateEstimator(
            output_dir=r"D:\Finalpj_tunnel_V3\smart_tunnel_V3_outputs\pipeline_v5_2_1"
            )

        self.state_model = TrafficState()
        self.accident_model = AccidentDetector()

    def process(self, frame_id, tracks):
        analysis = self.track_analyzer.update(frame_id, tracks)
        lane_result = self.lane_estimator.update(frame_id, analysis)

        merged_analysis = {
            **analysis,
            "lane_map": lane_result["lane_map"],
            "raw_lane_map": lane_result["raw_lane_map"],
            "lane_count": lane_result["lane_count"],
            "centerlines": lane_result["centerlines"],
            "lane_debug": lane_result["lane_debug"],
            "template_phase": lane_result["template_phase"],
            "template_confirmed": lane_result["template_confirmed"],
            "clusters": lane_result["clusters"],
        }

        state_result = self.state_model.update(frame_id, tracks, merged_analysis)
        accident_result = self.accident_model.update(frame_id, tracks, merged_analysis)

        return {
            "analysis": merged_analysis,
            "state": state_result,
            "accident": accident_result,
        }