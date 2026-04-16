# ==========================================
# 파일명: traffic_state_V5_3.py
# 설명:
# V3 상태로직 복원 + 최소 보강 버전
# - bottom point(y2) 기반 추적 유지
# - 속도 계산만 2D 이동거리로 개선
# - 최근 300프레임 평균속도 기반 상태 판단
# - 차량 수 조건 사용 안 함
# - 빈 프레임에서는 0을 넣지 않고 기존 버퍼 유지
# - 시작 상태는 NORMAL
# - hold 로직 추가
# - 디버그 정보 저장
# - pipeline_core_V5_3.py 시그니처와 호환
# ==========================================

class TrafficState:
    def __init__(self):
        # -----------------------------
        # 차량별 위치 이력
        # {track_id: [(cx, cy), ...]}
        # cy는 bottom point(y2)
        # -----------------------------
        self.track_history = {}

        # -----------------------------
        # 최근 프레임 평균속도 버퍼
        # "프레임 평균속도"를 저장
        # 상태 판단은 이 버퍼의 평균값으로 수행
        # -----------------------------
        self.state_buffer = []

        # -----------------------------
        # 파라미터
        # -----------------------------
        self.TRACK_HISTORY_SIZE = 20     # 차량별 최근 좌표 저장 개수
        self.STATE_BUFFER_SIZE = 300     # 최근 300프레임 평균속도 사용

        # 상태 임계값 (기존 V3 스타일 유지)
        self.JAM_SPEED_THR = 2.0
        self.CONGESTION_SPEED_THR = 5.0

        # -----------------------------
        # 상태 안정화용 hold 로직
        # - 후보 상태가 바뀌어도 바로 반영하지 않음
        # - 일정 프레임 연속 확인되면 상태 변경
        # -----------------------------
        self.prev_state = "NORMAL"
        self.state_hold_count = 0
        self.STATE_HOLD_FRAMES = 3

        # -----------------------------
        # 디버그 정보
        # 매 프레임 update 후 확인 가능
        # -----------------------------
        self.last_debug = {
            "frame_id": -1,
            "vehicle_ids": [],
            "vehicle_speeds": {},
            "frame_avg_speed": 0.0,
            "buffer_avg_speed": 0.0,
            "candidate_state": "NORMAL",
            "final_state": "NORMAL",
            "hold_count": 0,
            "buffer_size": 0,
            "empty_frame": True,
        }

    def update(self, frame_id, tracks, analysis=None):
        """
        입력:
            frame_id : 현재 프레임 번호
            tracks = [
                {"id": tid, "bbox": (x1, y1, x2, y2)},
                ...
            ]
            analysis : pipeline에서 전달하는 공통 분석 결과
                       (현재 상태로직에서는 필수 사용은 아니지만
                        파이프라인 시그니처 호환 위해 받음)

        반환:
            {
                "state": "NORMAL" / "CONGESTION" / "JAM",
                "debug": {...}
            }
        """

        speeds = {}

        # ==================================================
        # 1) track별 bottom point(cx, y2) 저장
        # 2) 직전 좌표와 비교해서 2D 이동거리 속도 계산
        # ==================================================
        for t in tracks:
            tid = t["id"]
            x1, y1, x2, y2 = t["bbox"]

            # bottom point 사용
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            if tid not in self.track_history:
                self.track_history[tid] = []

            self.track_history[tid].append((cx, cy))

            if len(self.track_history[tid]) > self.TRACK_HISTORY_SIZE:
                self.track_history[tid].pop(0)

            # 기본 속도
            speed = 0.0

            # 좌표가 2개 이상 있어야 이동거리 계산 가능
            if len(self.track_history[tid]) >= 2:
                prev_x, prev_y = self.track_history[tid][-2]
                curr_x, curr_y = self.track_history[tid][-1]

                # V3 방식: bottom point의 y축 이동량만 사용
                dy = curr_y - prev_y
                speed = abs(dy)

            speeds[tid] = round(speed, 3)

        # ==================================================
        # 3) 현재 프레임 평균속도 계산
        #    - 차량이 있으면 평균속도 계산
        #    - 차량이 없으면 frame_avg_speed는 "버퍼 평균" 사용
        #      (단, 버퍼도 비어 있으면 0)
        # ==================================================
        if len(speeds) > 0:
            frame_avg_speed = sum(speeds.values()) / len(speeds)
            empty_frame = False

            # 차량이 있을 때만 버퍼에 추가
            self.state_buffer.append(frame_avg_speed)
            if len(self.state_buffer) > self.STATE_BUFFER_SIZE:
                self.state_buffer.pop(0)

        else:
            empty_frame = True

            # 빈 프레임이면 0을 넣지 않는다
            # 기존 버퍼 유지
            if len(self.state_buffer) > 0:
                frame_avg_speed = sum(self.state_buffer) / len(self.state_buffer)
            else:
                frame_avg_speed = 0.0

        # ==================================================
        # 4) 최근 300프레임 평균속도 계산
        #    - 상태 판단은 이 값으로 수행
        #    - 버퍼가 비었으면 시작 상태 NORMAL 유지
        # ==================================================
        if len(self.state_buffer) > 0:
            buffer_avg_speed = sum(self.state_buffer) / len(self.state_buffer)
        else:
            buffer_avg_speed = 0.0

        # ==================================================
        # 5) 후보 상태(candidate state) 계산
        #    - 평균속도만 사용
        #    - 차량 수 사용 안 함
        # ==================================================
        if len(self.state_buffer) == 0:
            # 시작 직후 데이터가 없으면 NORMAL
            candidate_state = "NORMAL"
        else:
            if buffer_avg_speed < self.JAM_SPEED_THR:
                candidate_state = "JAM"
            elif buffer_avg_speed < self.CONGESTION_SPEED_THR:
                candidate_state = "CONGESTION"
            else:
                candidate_state = "NORMAL"

        # ==================================================
        # 6) hold 로직 적용
        #    - 후보 상태가 갑자기 바뀌더라도
        #      일정 프레임 연속 확인 후 최종 반영
        # ==================================================
        if candidate_state == self.prev_state:
            # 기존 상태와 같으면 hold 카운트 초기화
            self.state_hold_count = 0
            final_state = self.prev_state
        else:
            # 상태가 다르면 바로 바꾸지 않고 잠깐 관찰
            self.state_hold_count += 1

            if self.state_hold_count >= self.STATE_HOLD_FRAMES:
                self.prev_state = candidate_state
                self.state_hold_count = 0

            final_state = self.prev_state

        # ==================================================
        # 7) 디버그 정보 저장
        # ==================================================
        self.last_debug = {
            "frame_id": frame_id,
            "vehicle_ids": list(speeds.keys()),
            "vehicle_speeds": speeds,
            "frame_avg_speed": round(frame_avg_speed, 3),
            "buffer_avg_speed": round(buffer_avg_speed, 3),
            "candidate_state": candidate_state,
            "final_state": final_state,
            "hold_count": self.state_hold_count,
            "buffer_size": len(self.state_buffer),
            "empty_frame": empty_frame,
        }

        return {
            "state": final_state,
            "debug": self.last_debug,
        }

    def get_debug_info(self):
        """마지막 프레임 디버그 정보 반환"""
        return self.last_debug