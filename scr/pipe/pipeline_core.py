# ==========================================
# pipeline_core.py
# 상태 + 사고 판단 통합
# ==========================================

from traffic_state_V1 import TrafficState
from traffic_accident_V2_9 import AccidentDetector


class PipelineCore:
    def __init__(self):
        # 상태 판단 모듈
        self.state_model = TrafficState()

        # 사고 판단 모듈
        self.accident_model = AccidentDetector()

    def process(self, frame_id, tracks):
        """
        frame_id : 현재 프레임 번호
        tracks : ByteTrack 결과
        """

        # -----------------------------
        # 1️⃣ 상태 판단
        # -----------------------------
        state = self.state_model.update(frame_id, tracks)

        # -----------------------------
        # 2️⃣ 사고 판단
        # -----------------------------
        accident = self.accident_model.update(frame_id, tracks)

        # -----------------------------
        # 3️⃣ 평균 속도 계산 (로그용)
        # -----------------------------
        speeds = []
        for t in tracks:
            tid = t["id"]

            if tid in self.state_model.track_history and len(self.state_model.track_history[tid]) >= 2:
                dy = self.state_model.track_history[tid][-1][1] - self.state_model.track_history[tid][-2][1]
                speeds.append(abs(dy))

        avg_speed = sum(speeds)/len(speeds) if speeds else 0

        return {
            "state": state,
            "accident": accident,
            "avg_speed": avg_speed
        }