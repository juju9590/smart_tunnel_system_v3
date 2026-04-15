# ==========================================
# traffic_state_V5.py (STABLE FIX)
# 속도 정규화 + smoothing + 상태 판단
# ==========================================

class TrafficState:
    def __init__(self):
        self.track_history = {}
        self.state_buffer = []
        self.prev_speeds = {}

    def update(self, frame_id, tracks):

        speeds = {}
        frame_height = 720

        ROI_Y1 = int(frame_height * 0.3)
        ROI_Y2 = int(frame_height * 0.8)

        # -----------------------------
        # 1️⃣ 속도 계산
        # -----------------------------
        for t in tracks:
            tid = t["id"]
            x1, y1, x2, y2 = t["bbox"]

            cx = int((x1 + x2) / 2)
            cy = y2

            # -----------------------------
            # ROI 필터
            # -----------------------------
            if cy < ROI_Y1 or cy > ROI_Y2:
                # 🔥 중요: ROI 밖에서도 prev_speed 유지
                speeds[tid] = self.prev_speeds.get(tid, 0)
                continue

            self.track_history.setdefault(tid, []).append((cx, cy))

            if len(self.track_history[tid]) > 20:
                self.track_history[tid].pop(0)

            # -----------------------------
            # 초기 프레임 보호
            # -----------------------------
            if len(self.track_history[tid]) < 3:
                speed = self.prev_speeds.get(tid, 0)
                self.prev_speeds[tid] = speed
                speeds[tid] = speed
                continue

            # -----------------------------
            # 좌표 추출 (🔥 수정 핵심)
            # -----------------------------
            xp, yp = self.track_history[tid][-2]
            xc, yc = self.track_history[tid][-1]

            dx = abs(xc - xp)
            dy = yc - yp

            # -----------------------------
            # 이동 이상 제거
            # -----------------------------
            if abs(dy) > 40 or dx > 60:
                speed = self.prev_speeds.get(tid, 0)
                speeds[tid] = speed
                continue

            # -----------------------------
            # 원근 보정
            # -----------------------------
            scale = (yc - ROI_Y1) / (ROI_Y2 - ROI_Y1 + 1e-6)
            scale = max(0.1, min(scale, 1.0))

            speed = abs(dy) / (scale + 0.15)
            speed *= (1.2 + scale)

            prev_speed = self.prev_speeds.get(tid, speed)

            # -----------------------------
            # 절대값 필터
            # -----------------------------
            if speed > 20:
                speed = prev_speed

            # -----------------------------
            # 급변 방지
            # -----------------------------
            if abs(speed - prev_speed) > 10:
                speed = prev_speed

            # -----------------------------
            # smoothing (🔥 안정화)
            # -----------------------------
            speed = 0.3 * speed + 0.7 * prev_speed

            # -----------------------------
            # 최종 제한
            # -----------------------------
            speed = max(0, min(speed, 20))

            self.prev_speeds[tid] = speed
            speeds[tid] = speed

        # -----------------------------
        # 2️⃣ 평균 속도
        # -----------------------------
        valid = [s for s in speeds.values() if s < 20]

        # 🔥 speeds 비었을 때 보호
        if len(valid) == 0:
            avg_speed = self.state_buffer[-1] if self.state_buffer else 0
        else:
            avg_speed = sum(valid) / len(valid)

        self.state_buffer.append(avg_speed)

        if len(self.state_buffer) > 50:
            self.state_buffer.pop(0)

        avg_global = sum(self.state_buffer) / len(self.state_buffer)

        # -----------------------------
        # 3️⃣ 분산 계산
        # -----------------------------
        speed_values = list(speeds.values())
        speed_std = 0

        if len(speed_values) > 1:
            mean = avg_speed
            variance = sum((s - mean)**2 for s in speed_values) / len(speed_values)
            speed_std = variance ** 0.5

        # -----------------------------
        # 4️⃣ 상태 판단
        # -----------------------------
        if avg_global < 4:
            state = "JAM"
        elif avg_global < 12:
            state = "CONGESTION"
        else:
            state = "NORMAL"

        # -----------------------------
        # 5️⃣ 사고 힌트
        # -----------------------------
        accident_hint = speed_std > 2.5

        return {
            "state": state,
            "avg_speed": avg_global,
            "speed_std": speed_std,
            "accident_hint": accident_hint
        }