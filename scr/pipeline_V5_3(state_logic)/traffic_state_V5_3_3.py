# ==========================================
# 파일명: traffic_state_V5_3_2.py
# 설명:
# V5_3_2 상태로직
# - ROI 안 차량만 속도 계산
# - bottom point(y2) 기반 abs(dy) 사용
# - 화면 위치별 3구간 보정
# - 차량별 EMA 적용
# - 현재 프레임 평균속도 계산
# - 최근 600프레임 평균속도와 혼합하여 최종 상태 판단
# - 차량 수 조건 사용 안 함
# - 속도 0은 평균 계산 / 버퍼 누적에서 제외
# - 유효 속도(>0)가 없는 프레임에서는 0을 넣지 않고 기존 버퍼 유지
# - 시작 상태는 NORMAL
# - hold 로직 유지
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
        # 차량별 EMA 속도 저장
        # {track_id: ema_speed}
        # -----------------------------
        self.track_ema_speed = {}

        # -----------------------------
        # 최근 프레임 평균속도 버퍼
        # "0 제외된 프레임 평균속도"만 저장
        # 최근 600프레임 평균 계산용
        # -----------------------------
        self.state_buffer = []

        # -----------------------------
        # 파라미터
        # -----------------------------
        self.TRACK_HISTORY_SIZE = 20
        self.STATE_BUFFER_SIZE = 600

        # EMA 계수
        self.EMA_ALPHA = 0.3

        # 현재 프레임 평균 vs 최근 600프레임 평균 혼합 비율
        self.SHORT_WEIGHT = 0.7
        self.LONG_WEIGHT = 0.3

        # 상태 임계값
        self.JAM_SPEED_THR = 1.5
        self.CONGESTION_SPEED_THR = 3.0

        # -----------------------------
        # 상태 안정화용 hold 로직
        # -----------------------------
        self.prev_state = "NORMAL"
        self.state_hold_count = 0
        self.STATE_HOLD_FRAMES = 3

        # -----------------------------
        # 디버그 정보
        # -----------------------------
        self.last_debug = {
            "frame_id": -1,
            "vehicle_ids": [],
            "vehicle_speeds_raw": {},
            "vehicle_speeds_corrected": {},
            "vehicle_speeds_ema": {},
            "valid_vehicle_ids": [],
            "valid_vehicle_speeds_ema": {},
            "frame_avg_speed": 0.0,
            "buffer_avg_speed": 0.0,
            "final_speed": 0.0,
            "candidate_state": "NORMAL",
            "final_state": "NORMAL",
            "hold_count": 0,
            "buffer_size": 0,
            "empty_frame": True,
            "valid_speed_frame": False,
            "roi_box": None,
            "vehicle_dy": {},
        }

    def _get_roi_box(self, analysis):
        """
        pipeline_core에서 넘겨준 roi_box 사용
        형식: (x1, y1, x2, y2)

        없으면 roi_fixed를 이용해 세로 ROI만 사용
        """
        if analysis is None:
            return None

        roi_box = analysis.get("roi_box", None)
        if roi_box is not None:
            return roi_box

        roi_fixed = analysis.get("roi_fixed", None)
        if roi_fixed is not None and len(roi_fixed) == 2:
            y1, y2 = roi_fixed
            return (0, int(y1), 99999, int(y2))

        return None

    def _is_inside_roi(self, cx, cy, roi_box):
        """
        ROI 내부 여부 확인
        ROI가 None이면 전체 화면 허용
        """
        if roi_box is None:
            return True

        rx1, ry1, rx2, ry2 = roi_box
        return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

    def _get_position_scale(self, cy, roi_box):
        """
        화면 위치별 3구간 보정
        - ROI 기준 상/중/하 3구간
        """
        if roi_box is None:
            return 1.0

        _, ry1, _, ry2 = roi_box
        roi_h = max(1, ry2 - ry1)
        rel_y = cy - ry1

        if rel_y < roi_h * 0.33:
            return 2.0   # 상단(먼 쪽)
        elif rel_y < roi_h * 0.66:
            return 1.0   # 중단
        else:
            return 0.5   # 하단(가까운 쪽)

    def update(self, frame_id, tracks, analysis=None):
        """
        입력:
            frame_id : 현재 프레임 번호
            tracks = [
                {"id": tid, "bbox": (x1, y1, x2, y2)},
                ...
            ]
            analysis : pipeline에서 전달하는 공통 분석 결과

        반환:
            {
                "state": "NORMAL" / "CONGESTION" / "JAM",
                "debug": {...}
            }
        """

        roi_box = self._get_roi_box(analysis)

        raw_speeds = {}
        corrected_speeds = {}
        ema_speeds = {}
        dy_values = {}

        # ==================================================
        # 1) ROI 안 차량만 사용
        # 2) bottom point(cx, y2) 저장
        # 3) abs(dy) 기반 속도 계산
        # 4) 위치 보정
        # 5) 차량별 EMA 적용
        # ==================================================
        for t in tracks:
            tid = t["id"]
            x1, y1, x2, y2 = t["bbox"]

            # bottom point 사용
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            # ROI 밖 차량 제외
            if not self._is_inside_roi(cx, cy, roi_box):
                continue

            if tid not in self.track_history:
                self.track_history[tid] = []

            self.track_history[tid].append((cx, cy))

            if len(self.track_history[tid]) > self.TRACK_HISTORY_SIZE:
                self.track_history[tid].pop(0)

            raw_speed = 0.0
            corrected_speed = 0.0

            # 좌표가 2개 이상 있어야 이동량 계산 가능
            if len(self.track_history[tid]) >= 2:
                prev_x, prev_y = self.track_history[tid][-2]
                curr_x, curr_y = self.track_history[tid][-1]

                # bottom point의 y축 이동량만 사용
                dy = curr_y - prev_y
                raw_speed = abs(dy)
                dy_values[tid] = round(dy, 3)

                # 위치 보정
                pos_scale = self._get_position_scale(curr_y, roi_box)
                corrected_speed = raw_speed * pos_scale

            # 차량별 EMA 적용
            if tid not in self.track_ema_speed:
                ema_speed = corrected_speed
            else:
                ema_speed = (
                    self.EMA_ALPHA * corrected_speed
                    + (1 - self.EMA_ALPHA) * self.track_ema_speed[tid]
                )

            self.track_ema_speed[tid] = ema_speed

            raw_speeds[tid] = round(raw_speed, 3)
            corrected_speeds[tid] = round(corrected_speed, 3)
            ema_speeds[tid] = round(ema_speed, 3)

        # ==================================================
        # 6) 현재 프레임 평균속도 계산
        #    - ROI 안 차량 중 "속도 > 0" 인 값만 평균
        #    - 유효 속도가 없으면 0을 버퍼에 넣지 않음
        # ==================================================
        empty_frame = (len(ema_speeds) == 0)

        valid_ema_speeds = {
            tid: speed for tid, speed in ema_speeds.items()
            if speed > 0
        }

        valid_speed_frame = (len(valid_ema_speeds) > 0)

        if valid_speed_frame:
            frame_avg_speed = sum(valid_ema_speeds.values()) / len(valid_ema_speeds)

            # 유효 속도가 있는 프레임만 버퍼에 추가
            self.state_buffer.append(frame_avg_speed)
            if len(self.state_buffer) > self.STATE_BUFFER_SIZE:
                self.state_buffer.pop(0)
        else:
            # 유효 속도가 없으면 0을 넣지 않고 기존 버퍼 유지
            if len(self.state_buffer) > 0:
                frame_avg_speed = sum(self.state_buffer) / len(self.state_buffer)
            else:
                frame_avg_speed = 0.0

        # ==================================================
        # 7) 최근 600프레임 평균속도 계산
        #    - state_buffer에는 이미 0 제외 프레임 평균만 들어있음
        # ==================================================
        if len(self.state_buffer) > 0:
            buffer_avg_speed = sum(self.state_buffer) / len(self.state_buffer)
        else:
            buffer_avg_speed = 0.0

        # ==================================================
        # 8) 현재 프레임 평균 + 최근 600프레임 평균 혼합
        # ==================================================
        if len(self.state_buffer) == 0:
            final_speed = 0.0
        else:
            final_speed = (
                self.SHORT_WEIGHT * frame_avg_speed
                + self.LONG_WEIGHT * buffer_avg_speed
            )

        # ==================================================
        # 9) 후보 상태(candidate state) 계산
        #    - 속도만 사용
        # ==================================================
        state_speed = buffer_avg_speed

        if len(self.state_buffer) == 0:
                candidate_state = "NORMAL"
        else:
                if state_speed < self.JAM_SPEED_THR:
                    candidate_state = "JAM"
                elif state_speed < self.CONGESTION_SPEED_THR:
                    candidate_state = "CONGESTION"
                else:
                    candidate_state = "NORMAL"

        # ==================================================
        # 10) hold 로직 적용
        # ==================================================
        if candidate_state == self.prev_state:
            self.state_hold_count = 0
            final_state = self.prev_state
        else:
            self.state_hold_count += 1

            if self.state_hold_count >= self.STATE_HOLD_FRAMES:
                self.prev_state = candidate_state
                self.state_hold_count = 0

            final_state = self.prev_state

        # ==================================================
        # 11) 디버그 정보 저장
        # ==================================================
        self.last_debug = {
            "frame_id": frame_id,
            "vehicle_ids": list(ema_speeds.keys()),
            "vehicle_speeds_raw": raw_speeds,
            "vehicle_speeds_corrected": corrected_speeds,
            "vehicle_speeds_ema": ema_speeds,
            "valid_vehicle_ids": list(valid_ema_speeds.keys()),
            "valid_vehicle_speeds_ema": valid_ema_speeds,
            "frame_avg_speed": round(frame_avg_speed, 3),
            "buffer_avg_speed": round(buffer_avg_speed, 3),
            "final_speed": round(final_speed, 3),
            "state_speed": round(state_speed, 3),
            "candidate_state": candidate_state,
            "final_state": final_state,
            "hold_count": self.state_hold_count,
            "buffer_size": len(self.state_buffer),
            "empty_frame": empty_frame,
            "valid_speed_frame": valid_speed_frame,
            "roi_box": roi_box,
            "vehicle_dy": dy_values,
        }

        return {
            "state": final_state,
            "debug": self.last_debug,
        }

    def get_debug_info(self):
        """마지막 프레임 디버그 정보 반환"""
        return self.last_debug