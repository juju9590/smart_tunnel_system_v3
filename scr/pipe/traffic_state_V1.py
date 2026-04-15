# ==========================================
# traffic_state_V1.py (PIPELINE VERSION)
# 교통 상태 판단 (속도 기반)
# ==========================================

class TrafficState:
    def __init__(self):
        # 차량 ID별 위치 기록
        self.track_history = {}

        # 전체 평균 속도 버퍼 (노이즈 제거용)
        self.state_buffer = []

    def update(self, frame_id, tracks):
        """
        frame_id : 현재 프레임 번호
        tracks : ByteTrack 결과 (id + bbox)
        """

        speeds = {}

        # -----------------------------
        # 1️⃣ 차량별 속도 계산
        # -----------------------------
        for t in tracks:
            tid = t["id"]
            x1, y1, x2, y2 = t["bbox"]

            # 차량 기준점 (아래 중심)
            cx = int((x1 + x2) / 2)
            cy = y2

            # 이동 기록 저장
            self.track_history.setdefault(tid, []).append((cx, cy))

            # 기록 길이 제한 (메모리 보호)
            if len(self.track_history[tid]) > 20:
                self.track_history[tid].pop(0)

            # 속도 계산 (y축 이동량 기준)
            speed = 0
            if len(self.track_history[tid]) >= 2:
                dy = self.track_history[tid][-1][1] - self.track_history[tid][-2][1]
                speed = abs(dy)

            speeds[tid] = speed

        # -----------------------------
        # 2️⃣ 평균 속도 계산
        # -----------------------------
        avg_speed = sum(speeds.values()) / len(speeds) if speeds else 0

        # 버퍼에 저장 (프레임 단위 노이즈 제거)
        self.state_buffer.append(avg_speed)

        if len(self.state_buffer) > 50:
            self.state_buffer.pop(0)

        avg_global = sum(self.state_buffer) / len(self.state_buffer)

        

        # -----------------------------
        # 3️⃣ 상태 판단
        # -----------------------------
        if avg_global < 2:
            return "JAM"           # 정체
        elif avg_global < 5:
            return "CONGESTION"   # 혼잡
        else:
            return "NORMAL"       # 정상