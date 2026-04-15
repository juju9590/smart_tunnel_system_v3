# ==========================================
# pipeline_core_V5_FINAL.py
# ==========================================

from traffic_state_V4 import TrafficState
from traffic_accident_V4 import AccidentDetector


class PipelineCore:
    def __init__(self):
        self.state_model = TrafficState()
        self.accident_model = AccidentDetector()

        # 🔥 HOLD 버퍼
        self.accident_window = []

    def process(self, frame_id, tracks):

        # -----------------------------
        # 1️⃣ 상태 결과
        # -----------------------------
        state_result = self.state_model.update(frame_id, tracks)

        state = state_result["state"]
        avg_speed = state_result["avg_speed"]
        speed_std = state_result["speed_std"]
        accident_hint = state_result["accident_hint"]

        vehicle_count = len(tracks)

        # -----------------------------
        # 🔥 2️⃣ 사고 후보 판단 (강화)
        # -----------------------------
        accident_candidate = (
            accident_hint
            and avg_speed > 0.8
            and vehicle_count >= 3
        )

        # -----------------------------
        # 🔥 3️⃣ 사고 판단 (1번만!)
        # -----------------------------
        if accident_candidate:
            accident_raw = self.accident_model.update(frame_id, tracks)
        else:
            accident_raw = False

        # -----------------------------
        # 🔥 4️⃣ HOLD (60프레임 비율)
        # -----------------------------
        self.accident_window.append(accident_raw)

        if len(self.accident_window) > 60:
            self.accident_window.pop(0)

        acc_ratio = sum(self.accident_window) / len(self.accident_window)

        final_accident = acc_ratio > 0.4

        # -----------------------------
        # 🔥 5️⃣ 최종 상태
        # -----------------------------
        if final_accident:
            final_state = "ACCIDENT"
        else:
            final_state = state

        return {
            "state": final_state,
            "raw_state": state,
            "accident": final_accident,   # 🔥 여기 중요
            "avg_speed": avg_speed,
            "speed_std": speed_std,
            "accident_candidate": accident_candidate,
            "acc_ratio": round(acc_ratio, 2)  # 디버깅용
        }