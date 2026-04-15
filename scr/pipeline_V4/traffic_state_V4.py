# ==========================================
# traffic_state_V3.py (수정 완료)
# ==========================================

class TrafficState:
    def __init__(self):
        self.track_history = {}
        self.state_buffer = []

    def update(self, frame_id, tracks):

        speeds = {}

        # -----------------------------
        # 1️⃣ 속도 계산
        # -----------------------------
        for t in tracks:
            tid = t["id"]
            x1, y1, x2, y2 = t["bbox"]

            cx = int((x1 + x2) / 2)
            cy = y2

            self.track_history.setdefault(tid, []).append((cx, cy))

            if len(self.track_history[tid]) > 20:
                self.track_history[tid].pop(0)

            speed = 0
            if len(self.track_history[tid]) >= 2:
                dy = self.track_history[tid][-1][1] - self.track_history[tid][-2][1]
                speed = abs(dy)

            speeds[tid] = speed

        # -----------------------------
        # 2️⃣ 평균 속도
        # -----------------------------
        avg_speed = sum(speeds.values()) / len(speeds) if speeds else 0

        self.state_buffer.append(avg_speed)
        if len(self.state_buffer) > 50:
            self.state_buffer.pop(0)

        avg_global = sum(self.state_buffer) / len(self.state_buffer)

        # -----------------------------
        # 3️⃣ 분산 계산 (핵심)
        # -----------------------------
        speed_values = list(speeds.values())
        speed_std = 0

        if len(speed_values) > 1:
            mean = avg_speed
            variance = sum((s - mean)**2 for s in speed_values) / len(speed_values)
            speed_std = variance ** 0.5

        # -----------------------------
        # 4️⃣ 상태 판단 (네 기준)
        # -----------------------------
        if avg_global < 4:
            state = "JAM"
        elif avg_global < 12:
            state = "CONGESTION"
        else:
            state = "NORMAL"

        # -----------------------------
        # 🔥 사고 힌트 (핵심)
        # -----------------------------
        accident_hint = speed_std > 2.5

        return {
            "state": state,
            "avg_speed": avg_global,
            "speed_std": speed_std,
            "accident_hint": accident_hint
        }