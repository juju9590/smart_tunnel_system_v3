# -*- coding: utf-8 -*-
from collections import deque


class VentilationRiskManager:
    """
    CCTV 기반 교통정보로 터널 공기질 위험도를 추정하고
    환기 대응 알람을 판단하는 클래스

    기준 요약:
    - 차량 수: ROI 내부 차량 수
    - 평균속도: ROI 내부 차량 EMA 속도 평균
    - 교통밀도: 가중 차량 수 / ROI 최대 수용량
    - 체류시간: ROI 진입 ~ ROI 탈출 시간
    - bbox 가중치: ROI 진입 시점 bbox 크기 기준
    """

    def __init__(self, fps=10):
        # =========================
        # 1. 기본 설정값
        # =========================
        self.fps = fps

        # 차선당 최대 수용량
        self.capacity_per_lane = 10

        # bbox weight 제한값
        self.weight_clip_min = 0.8
        self.weight_clip_max = 1.5

        # 사고 상태 보정치
        self.accident_state_bonus = 0.15

        # 자유주행 기준속도
        self.free_flow_speed = 8.0   # 현재 프로젝트 정상 속도 기준에 맞게 조정 가능

        # 최대 기준 체류시간(초)
        self.max_dwell_time = 60.0

        # =========================
        # 2. 가중치
        # =========================
        self.W_N = 0.30
        self.W_D = 0.30
        self.W_T = 0.25
        self.W_V = 0.15

        # =========================
        # 3. 상태 보정값
        # =========================
        self.STATE_BONUS = {
            "NORMAL": 0.00,
            "CONGESTION": 0.05,
            "JAM": 0.10,
            "ACCIDENT": self.accident_state_bonus
        }

        # =========================
        # 4. 위험 단계 기준
        # =========================
        self.THRESHOLDS = {
            "NORMAL": 0.30,
            "CAUTION": 0.50,
            "WARNING": 0.70
        }

        # =========================
        # 5. 알람 유지 조건
        # =========================
        self.CAUTION_HOLD_FRAMES = int(3 * fps)
        self.WARNING_HOLD_FRAMES = int(5 * fps)
        self.DANGER_HOLD_FRAMES = int(3 * fps)
        self.RELEASE_HOLD_FRAMES = int(5 * fps)

        # =========================
        # 6. 차선 수별 기준 승용차 bbox
        # 실제 영상 보고 튜닝 필요
        # bbox size는 보통 w*h 또는 bbox area 사용
        # =========================
        self.bbox_ref_by_lane = {
            2: 3500.0,
            3: 2600.0,
            4: 2000.0
        }

        # =========================
        # 7. ROI 길이 추정용 설정
        # 선택 기능
        # =========================
        self.enable_500m_scaling = False
        self.default_roi_est_length = 50.0  # 추정 ROI 유효 길이(m), 초기값
        self.tunnel_length_m = 500.0

        # =========================
        # 8. 내부 상태
        # =========================
        self.current_level = "NORMAL"
        self.pending_level = "NORMAL"
        self.pending_count = 0
        self.release_count = 0

        self.last_result = {
            "risk_score_base": 0.0,
            "risk_score_final": 0.0,
            "risk_level": "NORMAL",
            "alarm": False,
            "message": "공기질 상태 정상",
            "hold_seconds": 0.0,
        }

        # 차량별 ROI 진입 정보
        # {
        #   track_id: {
        #       "entry_frame": int,
        #       "entry_bbox_size": float,
        #       "weight": float,
        #       "lane_count_at_entry": int
        #   }
        # }
        self.vehicle_entry_memory = {}

        # 최근 결과 로그
        self.history = deque(maxlen=300)

    # =========================================================
    # 1) 기본 유틸
    # =========================================================
    def _clamp(self, value, min_v, max_v):
        return max(min_v, min(float(value), max_v))

    def _clamp01(self, value):
        return self._clamp(value, 0.0, 1.0)

    def _normalize(self, value, max_value):
        if max_value <= 0:
            return 0.0
        return self._clamp01(value / max_value)

    def _level_rank(self, level):
        order = {
            "NORMAL": 0,
            "CAUTION": 1,
            "WARNING": 2,
            "DANGER": 3
        }
        return order.get(level, 0)

    def _required_hold_frames(self, level):
        if level == "CAUTION":
            return self.CAUTION_HOLD_FRAMES
        if level == "WARNING":
            return self.WARNING_HOLD_FRAMES
        if level == "DANGER":
            return self.DANGER_HOLD_FRAMES
        return 0

    def _score_to_level(self, score):
        if score < self.THRESHOLDS["NORMAL"]:
            return "NORMAL"
        elif score < self.THRESHOLDS["CAUTION"]:
            return "CAUTION"
        elif score < self.THRESHOLDS["WARNING"]:
            return "WARNING"
        else:
            return "DANGER"

    def _get_message(self, level):
        if level == "NORMAL":
            return "공기질 상태 정상"
        if level == "CAUTION":
            return "차량 밀집 증가, 공기질 저하 가능성 주의"
        if level == "WARNING":
            return "체류시간 증가 및 혼잡 발생, 환기 상태 점검 권고"
        if level == "DANGER":
            return "공기질 악화 위험 높음, 즉시 환기 대응 확인 필요"
        return "상태 미정"

    def _extract_bbox_size(self, vehicle):
        """
        bbox 크기 추출
        우선순위:
        1) bbox_size
        2) bbox_area
        3) bbox=(x1,y1,x2,y2) -> area
        """
        if "bbox_size" in vehicle:
            return max(float(vehicle["bbox_size"]), 1.0)

        if "bbox_area" in vehicle:
            return max(float(vehicle["bbox_area"]), 1.0)

        bbox = vehicle.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            w = max(float(x2) - float(x1), 1.0)
            h = max(float(y2) - float(y1), 1.0)
            return w * h

        return 1.0

    def _get_bbox_ref(self, lane_count):
        lane_count = int(lane_count) if lane_count else 2
        if lane_count in self.bbox_ref_by_lane:
            return self.bbox_ref_by_lane[lane_count]

        # 예외 시 가장 가까운 값 사용
        if lane_count <= 2:
            return self.bbox_ref_by_lane[2]
        elif lane_count == 3:
            return self.bbox_ref_by_lane[3]
        else:
            return self.bbox_ref_by_lane[4]

    # =========================================================
    # 2) ROI / 차량 정보 처리
    # =========================================================
    def _register_vehicle_entry(self, frame_id, track_id, bbox_size, lane_count):
        """
        ROI에 처음 진입한 차량의 entry 정보 기록
        """
        if track_id in self.vehicle_entry_memory:
            return

        bbox_ref = self._get_bbox_ref(lane_count)
        raw_weight = bbox_size / bbox_ref if bbox_ref > 0 else 1.0
        weight = self._clamp(raw_weight, self.weight_clip_min, self.weight_clip_max)

        self.vehicle_entry_memory[track_id] = {
            "entry_frame": int(frame_id),
            "entry_bbox_size": float(bbox_size),
            "weight": float(weight),
            "lane_count_at_entry": int(lane_count),
        }

    def _cleanup_exited_vehicles(self, current_track_ids):
        """
        ROI 밖으로 나간 차량은 메모리에서 제거
        """
        current_track_ids = set(current_track_ids)
        remove_ids = [
            tid for tid in self.vehicle_entry_memory.keys()
            if tid not in current_track_ids
        ]
        for tid in remove_ids:
            del self.vehicle_entry_memory[tid]

    def _compute_roi_capacity(self, lane_count):
        lane_count = max(int(lane_count), 1)
        return float(lane_count * self.capacity_per_lane)

    def _compute_dwell_times(self, frame_id, current_track_ids):
        """
        현재 ROI 내부 차량들의 체류시간 계산
        """
        dwell_times = []
        for tid in current_track_ids:
            info = self.vehicle_entry_memory.get(tid)
            if not info:
                continue

            entry_frame = info["entry_frame"]
            dwell_sec = max((int(frame_id) - int(entry_frame)) / float(self.fps), 0.0)
            dwell_times.append(dwell_sec)

        return dwell_times

    def _compute_500m_scale(self, roi_est_length=None):
        """
        선택 기능:
        ROI 추정 길이를 사용하여 500m 환산 계수 계산
        """
        if roi_est_length is None or roi_est_length <= 0:
            roi_est_length = self.default_roi_est_length

        return float(self.tunnel_length_m / roi_est_length)

    # =========================================================
    # 3) 알람 확정 / 해제 로직
    # =========================================================
    def update_alarm_state(self, predicted_level):
        if predicted_level == self.current_level:
            self.pending_level = predicted_level
            self.pending_count = 0
            self.release_count = 0
            return self.current_level

        # 상향
        if self._level_rank(predicted_level) > self._level_rank(self.current_level):
            if predicted_level == self.pending_level:
                self.pending_count += 1
            else:
                self.pending_level = predicted_level
                self.pending_count = 1

            need_frames = self._required_hold_frames(predicted_level)
            if self.pending_count >= need_frames:
                self.current_level = predicted_level
                self.pending_count = 0
                self.release_count = 0

            return self.current_level

        # 하향
        if self._level_rank(predicted_level) < self._level_rank(self.current_level):
            self.release_count += 1
            if self.release_count >= self.RELEASE_HOLD_FRAMES:
                self.current_level = predicted_level
                self.pending_level = predicted_level
                self.pending_count = 0
                self.release_count = 0

            return self.current_level

        return self.current_level

    # =========================================================
    # 4) 위험도 계산
    # =========================================================
    def calculate_risk_score(
        self,
        weighted_vehicle_count,
        roi_capacity,
        traffic_density,
        avg_dwell_time,
        avg_speed,
        traffic_state
    ):
        # N' : 가중 차량 수 / roi_capacity
        n_norm = self._normalize(weighted_vehicle_count, roi_capacity)

        # D' : density 그대로 0~1 범위로 제한
        d_norm = self._clamp01(traffic_density)

        # T'
        t_norm = self._normalize(avg_dwell_time, self.max_dwell_time)

        # V'
        v_norm = self._normalize(avg_speed, self.free_flow_speed)

        risk_score_base = (
            self.W_N * n_norm +
            self.W_D * d_norm +
            self.W_T * t_norm +
            self.W_V * (1.0 - v_norm)
        )

        state_bonus = self.STATE_BONUS.get(str(traffic_state).upper(), 0.0)
        risk_score_final = self._clamp01(risk_score_base + state_bonus)

        detail = {
            "n_norm": round(n_norm, 4),
            "d_norm": round(d_norm, 4),
            "t_norm": round(t_norm, 4),
            "v_norm": round(v_norm, 4),
            "state_bonus": round(state_bonus, 4),
        }

        return risk_score_base, risk_score_final, detail

    # =========================================================
    # 5) 외부 호출용 메인 함수
    # =========================================================
    def update(
        self,
        frame_id,
        lane_count,
        traffic_state,
        vehicles_in_roi,
        avg_speed_roi,
        roi_est_length=None,
        accident_status="NONE",
        accident_applied=False
    ):
        """
        Parameters
        ----------
        frame_id : int
        lane_count : int
        traffic_state : str
            NORMAL / CONGESTION / JAM / ACCIDENT
        vehicles_in_roi : list[dict]
            각 차량 dict 예시:
            {
                "track_id": 12,
                "bbox": [x1, y1, x2, y2],   # optional
                "bbox_size": 3200,          # optional
                "ema_speed": 4.2            # optional
            }
        avg_speed_roi : float
            ROI 내부 평균속도(이미 계산된 값)
        roi_est_length : float | None
            ROI 유효 길이 추정값
        """

        lane_count = max(int(lane_count), 1)
        current_track_ids = []

        # 1) ROI 진입 차량 등록
        for vehicle in vehicles_in_roi:
            track_id = vehicle.get("track_id")
            if track_id is None:
                continue

            current_track_ids.append(track_id)
            bbox_size = self._extract_bbox_size(vehicle)
            self._register_vehicle_entry(
                frame_id=frame_id,
                track_id=track_id,
                bbox_size=bbox_size,
                lane_count=lane_count
            )

        # 2) ROI 탈출 차량 정리
        self._cleanup_exited_vehicles(current_track_ids)

        # 3) 차량 수 / 가중 차량 수
        vehicle_count_roi = len(current_track_ids)

        weighted_vehicle_count = 0.0
        vehicle_weights = {}

        for tid in current_track_ids:
            info = self.vehicle_entry_memory.get(tid)
            if not info:
                continue
            weight = float(info["weight"])
            weighted_vehicle_count += weight
            vehicle_weights[tid] = round(weight, 3)

        # 4) ROI capacity
        roi_capacity = self._compute_roi_capacity(lane_count)

        # 5) 교통밀도
        traffic_density = 0.0
        if roi_capacity > 0:
            traffic_density = weighted_vehicle_count / roi_capacity
        traffic_density = self._clamp01(traffic_density)

        # 6) 체류시간
        dwell_times_roi = self._compute_dwell_times(frame_id, current_track_ids)
        avg_dwell_time_roi = (
            sum(dwell_times_roi) / len(dwell_times_roi)
            if dwell_times_roi else 0.0
        )

        # 7) 선택: 500m 환산 체류시간
        tunnel_scale_factor = 1.0
        avg_dwell_time_used = avg_dwell_time_roi

        if self.enable_500m_scaling:
            tunnel_scale_factor = self._compute_500m_scale(roi_est_length)
            avg_dwell_time_used = avg_dwell_time_roi * tunnel_scale_factor

        # 8) 평균속도
        avg_speed_roi = float(avg_speed_roi) if avg_speed_roi is not None else 0.0

        # 9) 위험도 계산
        risk_score_base, risk_score_final, detail = self.calculate_risk_score(
            weighted_vehicle_count=weighted_vehicle_count,
            roi_capacity=roi_capacity,
            traffic_density=traffic_density,
            avg_dwell_time=avg_dwell_time_used,
            avg_speed=avg_speed_roi,
            traffic_state=traffic_state
        )

        # 10) 단계 판정 및 hold 로직
        predicted_level = self._score_to_level(risk_score_final)
        final_level = self.update_alarm_state(predicted_level)

        # 11) 결과
        result = {
            "frame_id": int(frame_id),
            "lane_count": lane_count,
            "traffic_state": str(traffic_state).upper(),
            "accident_status": str(accident_status or "NONE").upper(),
            "accident_applied": bool(accident_applied),

            "vehicle_count_roi": vehicle_count_roi,
            "weighted_vehicle_count": round(weighted_vehicle_count, 3),
            "roi_capacity": round(roi_capacity, 3),
            "traffic_density": round(traffic_density, 3),

            "avg_speed_roi": round(avg_speed_roi, 3),

            "avg_dwell_time_roi": round(avg_dwell_time_roi, 3),
            "avg_dwell_time_used": round(avg_dwell_time_used, 3),
            "tunnel_scale_factor": round(tunnel_scale_factor, 3),

            "risk_score_base": round(risk_score_base, 3),
            "risk_score_final": round(risk_score_final, 3),
            "risk_level": final_level,
            "alarm": final_level in ["CAUTION", "WARNING", "DANGER"],
            "message": self._get_message(final_level),
            "hold_seconds": round(self.pending_count / self.fps, 1),

            "detail": {
                **detail,
                "vehicle_weights": vehicle_weights,
                "bbox_ref_used": round(self._get_bbox_ref(lane_count), 3),
            }
        }

        self.last_result = result
        self.history.append(result)
        return result

    # =========================================================
    # 6) 외부 조회
    # =========================================================
    def get_status(self):
        return self.last_result

    def set_bbox_ref(self, lane_count, bbox_ref_value):
        lane_count = int(lane_count)
        self.bbox_ref_by_lane[lane_count] = float(bbox_ref_value)

    def set_free_flow_speed(self, speed_value):
        self.free_flow_speed = float(speed_value)

    def set_max_dwell_time(self, dwell_time_sec):
        self.max_dwell_time = float(dwell_time_sec)

    def enable_tunnel_scaling(self, enable=True, default_roi_est_length=50.0):
        self.enable_500m_scaling = bool(enable)
        self.default_roi_est_length = float(default_roi_est_length)
