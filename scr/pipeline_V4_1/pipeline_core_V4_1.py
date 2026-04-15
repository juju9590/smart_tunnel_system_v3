from traffic_state_V4_1 import TrafficState
from traffic_accident_V4_1 import AccidentDetector


class PipelineCore:
    def __init__(self):
        self.state_model = TrafficState()
        self.accident_model = AccidentDetector()

    def process(self, frame_id, frame, tracks):
        # -----------------------------
        # 상태 판단
        # -----------------------------
        state_result = self.state_model.update(frame_id, frame, tracks)

        state = state_result.get("state", "NORMAL")
        avg_speed = state_result.get("avg_speed", 0.0)

        # relative speed 기준 표준편차
        relative_speeds = list(state_result.get("relative_speeds", {}).values())
        if len(relative_speeds) > 0:
            speed_std = float(__import__("numpy").std(relative_speeds))
        else:
            speed_std = 0.0

        # -----------------------------
        # 사고 힌트 / 후보
        # -----------------------------
        accident_hint = False
        accident_candidate = False
        acc_ratio = 0.0

        rel_map = state_result.get("relative_speeds", {})
        if len(rel_map) > 0:
            low_count = sum(1 for v in rel_map.values() if v < 0.35)
            total_count = len(rel_map)
            acc_ratio = low_count / max(1, total_count)

            accident_hint = (low_count >= 1)
            accident_candidate = (acc_ratio >= 0.3) or (state == "JAM")

        # -----------------------------
        # 사고 판단
        # -----------------------------
        if accident_candidate:
            accident_result = self.accident_model.update(frame_id, tracks, frame)
            accident_raw = accident_result.get("accident", False)
        else:
            accident_result = None
            accident_raw = False

        # -----------------------------
        # 최종 사고 판단
        # -----------------------------
        accident = bool(accident_raw)

        result = {
            "state": state,
            "raw_state": state_result.get("state", state),
            "accident": accident,
            "accident_hint": accident_hint,
            "accident_candidate": accident_candidate,
            "acc_ratio": round(float(acc_ratio), 4),
            "avg_speed": round(float(avg_speed), 4),
            "speed_std": round(float(speed_std), 4),
            "state_result": state_result,
            "accident_result": accident_result
        }

        return result