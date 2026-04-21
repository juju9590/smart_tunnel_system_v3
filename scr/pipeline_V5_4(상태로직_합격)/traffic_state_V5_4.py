# ==========================================
# 파일명: traffic_state_V5_5.py
# 설명:
# V5_5 상태로직
# - ROI 안 차량만 속도 계산
# - bottom point(cx, y2) 기반
# - dy만 쓰지 않고 (dx, dy) 이동벡터 사용
# - 최근 이동벡터 평균으로 진행방향 추정
# - 현재 이동벡터를 진행방향에 투영한 값을 속도로 사용
# - 점프값 / 너무 작은 bbox / 초기 프레임 제외
# - 차량별 EMA 적용
# - 최근 300프레임 평균(buffer_avg_speed) 기준 상태 판단
# - pipeline_core_V5_3.py 시그니처와 호환
# ==========================================

import math


class TrafficState:
    def __init__(self):
        # -----------------------------
        # 차량별 위치 이력
        # {track_id: [(cx, cy), ...]}
        # -----------------------------
        self.track_history = {}

        # -----------------------------
        # 차량별 EMA 속도 저장
        # -----------------------------
        self.track_ema_speed = {}

        # -----------------------------
        # 최근 프레임 평균속도 버퍼
        # -----------------------------
        self.state_buffer = []

        # -----------------------------
        # 파라미터
        # -----------------------------
        self.TRACK_HISTORY_SIZE = 20
        self.STATE_BUFFER_SIZE = 300

        # EMA 계수
        self.EMA_ALPHA = 0.3

        # 상태 임계값
        self.JAM_SPEED_THR = 2.3
        self.CONGESTION_SPEED_THR = 5.0

        # hold
        self.prev_state = "NORMAL"
        self.state_hold_count = 0
        self.STATE_HOLD_FRAMES = 3

        # -----------------------------
        # 속도 계산 파라미터
        # -----------------------------
        # 최근 몇 개 이동벡터로 진행방향 추정할지
        self.DIR_HISTORY_STEPS = 5

        # 너무 큰 점프 제거용
        self.MAX_JUMP_PIXELS = 40.0

        # bbox 너무 작으면 신뢰도 낮아서 제외
        self.MIN_BBOX_HEIGHT = 35

        # track 시작 직후는 제외
        self.MIN_HISTORY_FOR_SPEED = 3

        # -----------------------------
        # 디버그 정보
        # -----------------------------
        self.last_debug = {
            "frame_id": -1,
            "vehicle_ids": [],
            "vehicle_dx": {},
            "vehicle_dy": {},
            "vehicle_move_mag": {},
            "vehicle_proj_speed": {},
            "vehicle_speeds_ema": {},
            "valid_vehicle_ids": [],
            "valid_vehicle_speeds_ema": {},
            "frame_avg_speed": 0.0,
            "buffer_avg_speed": 0.0,
            "final_speed": 0.0,
            "state_speed": 0.0,
            "candidate_state": "NORMAL",
            "final_state": "NORMAL",
            "hold_count": 0,
            "buffer_size": 0,
            "empty_frame": True,
            "valid_speed_frame": False,
            "roi_box": None,
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

    def _safe_norm(self, vx, vy):
        mag = math.sqrt(vx * vx + vy * vy)
        if mag <= 1e-6:
            return (0.0, 0.0, 0.0)
        return (vx / mag, vy / mag, mag)

    def _estimate_direction_unit(self, history):
        """
        최근 이동벡터들의 평균 방향으로 진행방향 추정
        history: [(cx, cy), ...]
        반환: (ux, uy)
        """
        if len(history) < 2:
            return (0.0, -1.0)

        steps = min(self.DIR_HISTORY_STEPS, len(history) - 1)

        sum_dx = 0.0
        sum_dy = 0.0

        for i in range(-steps, 0):
            x_prev, y_prev = history[i - 1]
            x_curr, y_curr = history[i]
            sum_dx += (x_curr - x_prev)
            sum_dy += (y_curr - y_prev)

        ux, uy, mag = self._safe_norm(sum_dx, sum_dy)

        # 방향 계산 실패 시 기본값: 위쪽 진행
        if mag <= 1e-6:
            return (0.0, -1.0)

        return (ux, uy)

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

        dx_values = {}
        dy_values = {}
        move_mag_values = {}
        proj_speed_values = {}
        ema_speeds = {}

        for t in tracks:
            tid = t["id"]
            x1, y1, x2, y2 = t["bbox"]

            # bottom point
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            # bbox 크기
            bbox_height = int(y2 - y1)

            # ROI 밖 제외
            if not self._is_inside_roi(cx, cy, roi_box):
                continue

            # 너무 작은 bbox 제외
            if bbox_height < self.MIN_BBOX_HEIGHT:
                continue

            if tid not in self.track_history:
                self.track_history[tid] = []

            self.track_history[tid].append((cx, cy))

            if len(self.track_history[tid]) > self.TRACK_HISTORY_SIZE:
                self.track_history[tid].pop(0)

            history = self.track_history[tid]

            dx = 0.0
            dy = 0.0
            move_mag = 0.0
            proj_speed = 0.0

            # 충분한 이력이 있어야 속도 계산
            if len(history) >= self.MIN_HISTORY_FOR_SPEED:
                prev_x, prev_y = history[-2]
                curr_x, curr_y = history[-1]

                dx = curr_x - prev_x
                dy = curr_y - prev_y
                move_mag = math.sqrt(dx * dx + dy * dy)

                # 점프값 제거
                if move_mag <= self.MAX_JUMP_PIXELS:
                    # 진행방향 추정
                    dir_ux, dir_uy = self._estimate_direction_unit(history[:-1] + [history[-1]])

                    # 현재 이동벡터를 진행방향에 투영
                    proj = dx * dir_ux + dy * dir_uy

                    # 속도는 절대값 사용
                    proj_speed = abs(proj)
                else:
                    proj_speed = 0.0

            # EMA
            if tid not in self.track_ema_speed:
                ema_speed = proj_speed
            else:
                ema_speed = (
                    self.EMA_ALPHA * proj_speed
                    + (1 - self.EMA_ALPHA) * self.track_ema_speed[tid]
                )

            self.track_ema_speed[tid] = ema_speed

            dx_values[tid] = round(dx, 3)
            dy_values[tid] = round(dy, 3)
            move_mag_values[tid] = round(move_mag, 3)
            proj_speed_values[tid] = round(proj_speed, 3)
            ema_speeds[tid] = round(ema_speed, 3)

        # -----------------------------
        # 현재 프레임 평균속도
        # 0 초과인 값만 사용
        # -----------------------------
        empty_frame = (len(ema_speeds) == 0)

        valid_ema_speeds = {
            tid: speed for tid, speed in ema_speeds.items()
            if speed > 0
        }

        valid_speed_frame = (len(valid_ema_speeds) > 0)

        if valid_speed_frame:
            frame_avg_speed = sum(valid_ema_speeds.values()) / len(valid_ema_speeds)

            self.state_buffer.append(frame_avg_speed)
            if len(self.state_buffer) > self.STATE_BUFFER_SIZE:
                self.state_buffer.pop(0)
        else:
            if len(self.state_buffer) > 0:
                frame_avg_speed = sum(self.state_buffer) / len(self.state_buffer)
            else:
                frame_avg_speed = 0.0

        # -----------------------------
        # 최근 300프레임 평균속도
        # -----------------------------
        if len(self.state_buffer) > 0:
            buffer_avg_speed = sum(self.state_buffer) / len(self.state_buffer)
        else:
            buffer_avg_speed = 0.0

        final_speed = buffer_avg_speed
        state_speed = buffer_avg_speed

        # -----------------------------
        # 상태 판단
        # -----------------------------
        if len(self.state_buffer) == 0:
            candidate_state = "NORMAL"
        else:
            if state_speed < self.JAM_SPEED_THR:
                candidate_state = "JAM"
            elif state_speed < self.CONGESTION_SPEED_THR:
                candidate_state = "CONGESTION"
            else:
                candidate_state = "NORMAL"

        # -----------------------------
        # hold
        # -----------------------------
        if candidate_state == self.prev_state:
            self.state_hold_count = 0
            final_state = self.prev_state
        else:
            self.state_hold_count += 1

            if self.state_hold_count >= self.STATE_HOLD_FRAMES:
                self.prev_state = candidate_state
                self.state_hold_count = 0

            final_state = self.prev_state

        # -----------------------------
        # 디버그 저장
        # -----------------------------
        self.last_debug = {
            "frame_id": frame_id,
            "vehicle_ids": list(ema_speeds.keys()),
            "vehicle_dx": dx_values,
            "vehicle_dy": dy_values,
            "vehicle_move_mag": move_mag_values,
            "vehicle_proj_speed": proj_speed_values,
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
        }

        return {
            "state": final_state,
            "debug": self.last_debug,
        }

    def get_debug_info(self):
        return self.last_debug