# ==========================================
# traffic_state.py
# 상태 판단 로직
# - 최근 window_size 프레임 기준 평균 속도 계산
# - speed == 0 인 값은 평균 계산에서 제외
# - pipeline_core_V5_3.py 와 호환되도록
#   update(frame_id, tracks, merged_analysis) 형태 지원
# ==========================================

from collections import deque


class TrafficState:
    def __init__(self, window_size=300):
        # 최근 몇 프레임으로 상태를 판단할지
        self.window_size = window_size

        # 최근 프레임 평균속도 저장
        self.speed_buffer = deque(maxlen=window_size)

        # 최근 프레임 차량 수 저장
        self.vehicle_count_buffer = deque(maxlen=window_size)

        # 최근 상태 저장
        self.state_buffer = deque(maxlen=window_size)

    def update(self, frame_id, tracks, merged_analysis=None):
        """
        frame_id : 현재 프레임 번호
        tracks   : 추적 결과 리스트
        merged_analysis : pipeline_core에서 넘겨주는 추가 분석값
                          (현재는 안 써도 되므로 기본값 None)

        tracks 예시:
        [
            {"id": 1, "speed": 12.3},
            {"id": 2, "speed": 8.7},
        ]

        반환 예시:
        {
            "frame_id": 1,
            "vehicle_count": 3,
            "frame_avg_speed": 10.25,
            "avg_speed": 12.84,
            "speed_std": 2.11,
            "state": "NORMAL"
        }
        """

        # -----------------------------
        # 1) 현재 프레임 차량 수
        # -----------------------------
        vehicle_count = len(tracks) if tracks else 0

        # -----------------------------
        # 2) 현재 프레임 속도 목록 추출
        # -----------------------------
        speed_values = []

        for t in tracks:
            if isinstance(t, dict) and "speed" in t:
                try:
                    speed_values.append(float(t["speed"]))
                except:
                    pass

        # -----------------------------
        # 3) 현재 프레임 평균속도 계산
        # -----------------------------
        if speed_values:
            frame_avg_speed = sum(speed_values) / len(speed_values)
        else:
            frame_avg_speed = 0

        # -----------------------------
        # 4) 버퍼 저장
        # -----------------------------
        self.speed_buffer.append(frame_avg_speed)
        self.vehicle_count_buffer.append(vehicle_count)

        # -----------------------------
        # 5) 최근 구간 평균속도 계산
        #    핵심: speed == 0 제외
        # -----------------------------
        valid_speeds = [s for s in self.speed_buffer if s > 0]

        if valid_speeds:
            avg_speed = sum(valid_speeds) / len(valid_speeds)
        else:
            avg_speed = 0

        # -----------------------------
        # 6) 속도 표준편차 계산
        # -----------------------------
        speed_std = 0
        if len(valid_speeds) > 1:
            mean = avg_speed
            variance = sum((s - mean) ** 2 for s in valid_speeds) / len(valid_speeds)
            speed_std = variance ** 0.5

        # -----------------------------
        # 7) 상태 판단
        #    임계값은 실험 후 조정
        # -----------------------------
        if avg_speed >= 20:
            state = "NORMAL"
        elif avg_speed >= 7:
            state = "CONGESTION"
        else:
            state = "JAM"

        self.state_buffer.append(state)

        # -----------------------------
        # 8) 결과 반환
        # -----------------------------
        return {
            "frame_id": frame_id,
            "vehicle_count": vehicle_count,
            "frame_avg_speed": round(frame_avg_speed, 2),
            "avg_speed": round(avg_speed, 2),
            "speed_std": round(speed_std, 2),
            "state": state
        }