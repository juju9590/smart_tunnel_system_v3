# ==========================================
# 파일명 예시: traffic_state_V4_1.py
# 용도: 원근/차량크기 영향 줄인 상태 판단 로직
# 핵심: bbox 크기 정규화 + 차량별 기준속도 대비 상대속도 사용
# ==========================================

import numpy as np


class TrafficState:
    def __init__(
        self,
        roi_y1_ratio=0.30,          # ROI 시작 비율
        roi_y2_ratio=0.85,          # ROI 끝 비율
        alpha=0.35,                 # 차량별 속도 smoothing
        base_alpha=0.05,            # 차량별 기준속도 갱신 비율(천천히)
        state_buffer_size=120,      # 상태 안정화 버퍼
        track_history_size=10,      # 차량별 좌표 저장 길이
        min_box_h=18,               # 너무 작은 박스 제외
        stop_ratio=0.35,            # 기준속도 대비 이하면 정체 수준
        congestion_ratio=0.65,      # 기준속도 대비 이하면 혼잡 수준
        min_active_tracks=2         # 상태판단 최소 차량 수
    ):
        """
        [V4_1 핵심 개선]
        1) 절대 픽셀속도 대신 bbox 높이로 정규화한 속도 사용
        2) 차량별 기준속도(baseline) 대비 현재 속도 비율(relative ratio) 사용
        3) 상태 판단을 절대값이 아니라 상대 비율 기반으로 수행

        기대 효과:
        - 가까운 차가 무조건 빠르게 나오는 문제 완화
        - 먼 차가 무조건 느리게 나오는 문제 완화
        - 2차선/4차선, 카메라 거리 차이에도 덜 흔들림
        """

        self.roi_y1_ratio = roi_y1_ratio
        self.roi_y2_ratio = roi_y2_ratio

        self.alpha = alpha
        self.base_alpha = base_alpha

        self.state_buffer_size = state_buffer_size
        self.track_history_size = track_history_size

        self.min_box_h = min_box_h
        self.stop_ratio = stop_ratio
        self.congestion_ratio = congestion_ratio
        self.min_active_tracks = min_active_tracks

        # 차량별 좌표 history
        # tid -> [(cx, cy), ...]
        self.track_history = {}

        # 차량별 현재 smoothed normalized speed
        # tid -> float
        self.speed_memory = {}

        # 차량별 기준속도(baseline)
        # tid -> float
        self.base_speed_memory = {}

        # 프레임별 상태지표 버퍼
        self.state_buffer = []

    # ==========================================
    # ROI 계산
    # ==========================================
    def get_roi(self, frame):
        h = frame.shape[0]
        roi_y1 = int(h * self.roi_y1_ratio)
        roi_y2 = int(h * self.roi_y2_ratio)
        return roi_y1, roi_y2

    # ==========================================
    # tracks -> 속도 계산
    # ==========================================
    def extract_boxes_and_speeds(self, frame, tracks):
        """
        반환:
            boxes          : tid -> (x1,y1,x2,y2)
            norm_speeds    : tid -> bbox 높이 정규화 속도
            rel_speeds     : tid -> 기준속도 대비 상대속도 비율
            base_speeds    : tid -> 차량별 기준속도
            roi_y1, roi_y2
        """
        roi_y1, roi_y2 = self.get_roi(frame)

        boxes = {}
        norm_speeds = {}
        rel_speeds = {}
        base_speeds = {}

        for obj in tracks:
            tid = int(obj["id"])
            x1, y1, x2, y2 = map(int, obj["bbox"])

            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            # 너무 작은 박스는 제외 (먼 차량 노이즈 방지)
            if h < self.min_box_h:
                continue

            cx = int((x1 + x2) / 2)
            cy = int(y2)   # 바닥점 기준

            # ROI 밖 제외
            if cy < roi_y1 or cy > roi_y2:
                continue

            boxes[tid] = (x1, y1, x2, y2)

            # history 저장
            self.track_history.setdefault(tid, []).append((cx, cy))
            if len(self.track_history[tid]) > self.track_history_size:
                self.track_history[tid].pop(0)

            # 이전 좌표가 있어야 속도 계산 가능
            if len(self.track_history[tid]) < 2:
                continue

            x_prev, y_prev = self.track_history[tid][-2]
            x_curr, y_curr = self.track_history[tid][-1]

            dx = x_curr - x_prev
            dy = y_curr - y_prev

            # 이동량 magnitude 사용
            disp = np.sqrt(dx * dx + dy * dy)

            # ------------------------------
            # 핵심 개선 1
            # bbox 높이로 나눈 정규화 속도
            # 가까운 차량은 박스가 크므로 나눠서 보정
            # 먼 차량은 박스가 작으므로 속도가 너무 작아지는 문제 완화
            # ------------------------------
            norm_speed = disp / (h + 1e-6)

            # smoothing
            if tid in self.speed_memory:
                norm_speed = self.alpha * norm_speed + (1 - self.alpha) * self.speed_memory[tid]
            self.speed_memory[tid] = norm_speed

            # ------------------------------
            # 핵심 개선 2
            # 차량별 기준속도(baseline) 업데이트
            # 너무 급하게 변하지 않도록 천천히 반영
            # ------------------------------
            if tid not in self.base_speed_memory:
                self.base_speed_memory[tid] = norm_speed
            else:
                prev_base = self.base_speed_memory[tid]

                # 현재 속도가 너무 0에 가까운 정체 상황이어도
                # baseline이 한 번에 무너지지 않게 천천히 갱신
                new_base = (1 - self.base_alpha) * prev_base + self.base_alpha * norm_speed

                # baseline 바닥값
                self.base_speed_memory[tid] = max(new_base, 0.03)

            base_speed = self.base_speed_memory[tid]

            # ------------------------------
            # 핵심 개선 3
            # 기준속도 대비 현재 속도 비율
            # 1.0 근처면 평소 흐름
            # 0.5면 평소보다 절반 수준
            # 0.2면 거의 멈춤
            # ------------------------------
            rel_speed = norm_speed / (base_speed + 1e-6)

            norm_speeds[tid] = float(norm_speed)
            rel_speeds[tid] = float(rel_speed)
            base_speeds[tid] = float(base_speed)

        return boxes, norm_speeds, rel_speeds, base_speeds, roi_y1, roi_y2

    # ==========================================
    # 상태 판단
    # ==========================================
    def classify_state(self, rel_speeds, vehicle_count):
        """
        상태 판단은 절대 속도값이 아니라
        '현재 차량들이 평소 대비 얼마나 느려졌는가' 로 본다.
        """

        if vehicle_count < self.min_active_tracks or len(rel_speeds) == 0:
            indicator = 1.0
        else:
            # 중앙값 사용 -> 튀는 차 1대 영향 줄임
            indicator = float(np.median(list(rel_speeds.values())))

        self.state_buffer.append(indicator)
        if len(self.state_buffer) > self.state_buffer_size:
            self.state_buffer.pop(0)

        global_indicator = float(np.mean(self.state_buffer)) if self.state_buffer else 1.0

        # 비율이 낮을수록 느림
        if global_indicator < self.stop_ratio:
            state = "JAM"
        elif global_indicator < self.congestion_ratio:
            state = "CONGESTION"
        else:
            state = "NORMAL"

        return state, indicator, global_indicator

    # ==========================================
    # 메인 update
    # ==========================================
    def update(self, frame_idx, frame, tracks):
        boxes, norm_speeds, rel_speeds, base_speeds, roi_y1, roi_y2 = self.extract_boxes_and_speeds(frame, tracks)

        vehicle_count = len(boxes)

        avg_norm_speed = float(np.mean(list(norm_speeds.values()))) if norm_speeds else 0.0
        avg_rel_speed = float(np.mean(list(rel_speeds.values()))) if rel_speeds else 0.0

        state, frame_indicator, global_indicator = self.classify_state(rel_speeds, vehicle_count)

        result = {
            "frame_idx": frame_idx,
            "state": state,

            # 디버깅용
            "avg_speed": round(avg_norm_speed, 4),          # 정규화 속도 평균
            "avg_relative_speed": round(avg_rel_speed, 4), # 기준속도 대비 평균
            "frame_indicator": round(frame_indicator, 4),
            "avg_global": round(global_indicator, 4),

            "vehicle_count": vehicle_count,
            "boxes": boxes,
            "speeds": {k: round(v, 4) for k, v in norm_speeds.items()},
            "relative_speeds": {k: round(v, 4) for k, v in rel_speeds.items()},
            "base_speeds": {k: round(v, 4) for k, v in base_speeds.items()},

            "roi_y1": roi_y1,
            "roi_y2": roi_y2
        }

        return result