# ==========================================
# 파일명: traffic_state_V5_1.py
# 설명:
# V5_1 교통 상태 판단 로직
# - TrackAnalyzer가 계산한 공통 분석 결과 사용
# - avg_speed, vehicle_count 중심으로 상태 판단
# - lane 정보는 보조적으로만 활용
# ==========================================


class TrafficState:
    def __init__(self):
        # -----------------------------
        # 기본 임계값
        # -----------------------------
        self.JAM_SPEED_THR = 2.0
        self.CONGESTION_SPEED_THR = 5.0

        self.JAM_COUNT_THR = 6
        self.CONGESTION_COUNT_THR = 3

        # -----------------------------
        # 상태 안정화용 메모리
        # 최근 상태를 보고 급변 방지
        # -----------------------------
        self.prev_state = "NORMAL"
        self.state_hold_count = 0

        self.STATE_HOLD_FRAMES = 3

        # 디버그 정보
        self.last_debug = {
            "state": "NORMAL",
            "avg_speed": 0.0,
            "vehicle_count": 0,
            "lane_count": 0,
            "reason": ""
        }

    def _decide_state_once(self, avg_speed, vehicle_count, lane_count):
        """
        1차 상태 판단
        lane_count는 현재는 보조 정보로만 받음
        """

        # 차량이 거의 없으면 정상
        if vehicle_count <= 1:
            return "NORMAL", "vehicle_count<=1"

        # 매우 저속 + 차량 수 많음 => JAM
        if avg_speed <= self.JAM_SPEED_THR and vehicle_count >= self.JAM_COUNT_THR:
            return "JAM", "low_speed_and_many_vehicles"

        # 저속 + 차량 어느 정도 있음 => CONGESTION
        if avg_speed <= self.CONGESTION_SPEED_THR and vehicle_count >= self.CONGESTION_COUNT_THR:
            return "CONGESTION", "mid_low_speed_and_some_vehicles"

        # 차량 수만 많고 속도는 조금 나오는 경우
        if vehicle_count >= 8 and avg_speed <= 6.5:
            return "CONGESTION", "many_vehicles_with_reduced_speed"

        return "NORMAL", "default"

    def _smooth_state(self, new_state):
        """
        상태가 너무 자주 바뀌지 않도록 hold
        """
        if new_state == self.prev_state:
            self.state_hold_count = min(self.state_hold_count + 1, 999)
            return self.prev_state

        # 새 상태가 잠깐 나왔다고 바로 바꾸지 않음
        if self.state_hold_count < self.STATE_HOLD_FRAMES:
            self.state_hold_count += 1
            return self.prev_state

        # 충분히 유지되면 상태 전환
        self.prev_state = new_state
        self.state_hold_count = 0
        return new_state

    def update(self, frame_id, tracks, analysis):
        """
        입력:
            frame_id : 현재 프레임 번호
            tracks   : [{"id":..., "bbox":...}, ...]
            analysis : TrackAnalyzer 결과 dict

        출력:
            {
                "state": "NORMAL" / "CONGESTION" / "JAM"
            }
        """

        avg_speed = float(analysis.get("avg_speed", 0.0))
        vehicle_count = int(analysis.get("vehicle_count", len(tracks)))
        lane_count = int(analysis.get("lane_count", 0))

        # 1차 판단
        raw_state, reason = self._decide_state_once(
            avg_speed=avg_speed,
            vehicle_count=vehicle_count,
            lane_count=lane_count
        )

        # 상태 안정화
        final_state = self._smooth_state(raw_state)

        # 디버그 저장
        self.last_debug = {
            "frame_id": frame_id,
            "state": final_state,
            "raw_state": raw_state,
            "avg_speed": avg_speed,
            "vehicle_count": vehicle_count,
            "lane_count": lane_count,
            "reason": reason
        }

        return {
            "state": final_state
        }

    def get_debug_info(self):
        return self.last_debug