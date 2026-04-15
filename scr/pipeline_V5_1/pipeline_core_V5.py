# ==========================================
# pipeline_core_V5.py (STABLE FINAL)
# ==========================================

from traffic_state_V5 import TrafficState
from traffic_accident_V5_1 import AccidentDetector


class PipelineCore:
    def __init__(self):
        self.state_model = TrafficState()
        self.accident_model = AccidentDetector()

        # -----------------------------
        # 🔥 사고 HOLD 버퍼 (시간 기반 안정화)
        # -----------------------------
        self.accident_window = []

        # 최소 프레임 (초기 안정화용)
        self.MIN_FRAMES = 10

    def process(self, frame_id, tracks):

        # -----------------------------
        # 🔥 예외: 차량 없음 (중요)
        # -----------------------------
        if len(tracks) == 0:
            return {
                "state": "NORMAL",
                "raw_state": "NORMAL",
                "accident": False,
                "avg_speed": 0,
                "speed_std": 0,
                "accident_candidate": False,
                "acc_ratio": 0.0
            }

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
        # 2️⃣ 사고 후보 판단 (필터링 단계)
        # -----------------------------
        accident_candidate = (
            accident_hint and         # 속도 분산 이상
            avg_speed > 1.0 and       # 완전 정지는 제외
            vehicle_count >= 3        # 최소 차량 수
        )

        # -----------------------------
        # 🔥 JAM 보호 (완화 버전)
        # -----------------------------
        # 기존: JAM이면 무조건 False → 문제
        # 수정: JAM + 완전 저속일 때만 차단
        if state == "JAM" and avg_speed < 1:
            accident_candidate = False

        # -----------------------------
        # 3️⃣ 사고 판단
        # -----------------------------
        if accident_candidate:
            accident_raw = self.accident_model.update(frame_id, tracks)
        else:
            accident_raw = False

        # -----------------------------
        # 4️⃣ HOLD 버퍼 (시간 기반 안정화)
        # -----------------------------
        self.accident_window.append(1 if accident_raw else 0)

        # 버퍼 크기 유지
        if len(self.accident_window) > 60:
            self.accident_window.pop(0)

        # -----------------------------
        # 🔥 초기 프레임 보호
        # -----------------------------
        if len(self.accident_window) < self.MIN_FRAMES:
            acc_ratio = 0
        else:
            acc_ratio = sum(self.accident_window) / len(self.accident_window)

        # -----------------------------
        # 5️⃣ 최종 사고 판단
        # -----------------------------
        final_accident = acc_ratio > 0.4

        # -----------------------------
        # 6️⃣ 최종 상태 결정
        # -----------------------------
        if final_accident:
            final_state = "ACCIDENT"
        else:
            final_state = state

        # -----------------------------
        # 7️⃣ 결과 반환
        # -----------------------------
        return {
            "state": final_state,              # 최종 상태
            "raw_state": state,                # 원래 상태
            "accident": final_accident,        # 사고 여부
            "avg_speed": avg_speed,
            "speed_std": speed_std,
            "accident_candidate": accident_candidate,
            "acc_ratio": round(acc_ratio, 2)
        }