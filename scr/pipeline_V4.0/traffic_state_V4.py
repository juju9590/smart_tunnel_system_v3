# ==========================================
# 파일명 예시: traffic_state_V4.py
# 용도: 파이프라인용 상태 판단 로직 클래스
# ==========================================
# [설명]
# 이 파일은 기존 "SMART TUNNEL V1" 코드에서
# 상태판단에 필요한 부분만 분리하여
# 파이프라인에서 바로 사용할 수 있도록 만든 버전이다.
#
# 원본 코드에서는
# - YOLO 추론
# - 영상 읽기
# - 영상 저장
# - 로그 저장
# - 상태 판단
# - 사고 판단
# 이 모두 한 파일에 들어 있었지만,
#
# 여기서는 "상태 판단 전용 클래스"만 남겼다.
#
# 사용 예:
#   state_model = TrafficState()
#   result = state_model.update(frame_idx, frame, tracks)
#
# tracks 형식 예:
#   [
#       {"id": 1, "bbox": [x1, y1, x2, y2]},
#       {"id": 2, "bbox": [x1, y1, x2, y2]},
#   ]
#
# 반환 예:
#   {
#       "state": "NORMAL",
#       "avg_speed": 5.21,
#       "avg_global": 4.87,
#       "vehicle_count": 4,
#       "boxes": {...},
#       "speeds": {...},
#       "roi_y1": 120,
#       "roi_y2": 320
#   }
# ==========================================

import numpy as np


class TrafficState:
    def __init__(
        self,
        roi_y1_ratio=0.3,          # ROI 시작 비율
        roi_y2_ratio=0.8,          # ROI 끝 비율
        alpha=0.3,                 # 속도 smoothing 비율
        jam_enter=4,             # JAM 진입 임계값
        congestion_enter=12,      # CONGESTION 진입 임계값
        state_buffer_size=300,     # 상태 안정화를 위한 평균속도 버퍼 길이
        track_history_size=20      # 차량별 좌표 저장 길이
    ):
        """
        상태 판단기 초기화

        [핵심 아이디어]
        1. 각 차량의 바운딩박스 하단 y값(y2)을 이용해 픽셀 속도를 계산
        2. ROI 안에 들어온 차량만 사용해서 노이즈 감소
        3. 프레임별 평균 속도를 다시 장기 버퍼로 평균내서 상태를 안정화
        4. 최종적으로 NORMAL / CONGESTION / JAM 판정
        """

        # ------------------------------
        # 설정값 저장
        # ------------------------------
        self.roi_y1_ratio = roi_y1_ratio
        self.roi_y2_ratio = roi_y2_ratio
        self.alpha = alpha
        self.jam_enter = jam_enter
        self.congestion_enter = congestion_enter
        self.state_buffer_size = state_buffer_size
        self.track_history_size = track_history_size

        # ------------------------------
        # 차량별 좌표 기록
        # tid -> [(cx, cy), ...]
        # ------------------------------
        self.track_history = {}

        # ------------------------------
        # 차량별 속도 memory
        # smoothing에 사용
        # tid -> previous smoothed speed
        # ------------------------------
        self.speed_memory = {}

        # ------------------------------
        # 프레임별 평균속도 buffer
        # 상태판단 안정화용
        # ------------------------------
        self.state_buffer = []

    # ==========================================
    # 1. ROI 계산
    # ==========================================
    def get_roi(self, frame):
        """
        현재 프레임 높이를 기준으로 ROI 구간 계산

        [왜 ROI를 쓰는가?]
        영상 맨 위/맨 아래 영역은
        - 차량이 너무 작거나
        - 너무 가까워서 속도값이 튈 수 있다.
        그래서 중간 관심영역만 써서 상태판단을 안정화한다.
        """
        h = frame.shape[0]
        roi_y1 = int(h * self.roi_y1_ratio)
        roi_y2 = int(h * self.roi_y2_ratio)
        return roi_y1, roi_y2

    # ==========================================
    # 2. tracks -> boxes, speeds 변환
    # ==========================================
    def extract_boxes_and_speeds(self, frame, tracks):
        """
        파이프라인에서 받은 tracks를 이용해
        - 박스 정보(boxes)
        - 차량별 속도(speeds)
        를 계산한다.

        [속도 계산 방식]
        원본 코드와 동일하게 bbox 하단 y2 기준 이동량을 사용한다.
        그리고 원근 보정 대신 ROI 위치에 따라 scale 보정을 적용한다.

        speed = abs(dy)/(scale+0.15)
        speed *= (1.2+scale)

        이 방식은 원본 V1/V4 계열에서 사용하던 방식이라
        현재 복구 목적에 맞춰 유지한다.
        """
        roi_y1, roi_y2 = self.get_roi(frame)

        boxes = {}
        speeds = {}

        for obj in tracks:
            tid = int(obj["id"])
            x1, y1, x2, y2 = map(int, obj["bbox"])

            cx = int((x1 + x2) / 2)
            cy = y2

            # ----------------------------------
            # ROI 밖 차량은 상태판단용 속도 계산에서 제외
            # ----------------------------------
            if cy < roi_y1 or cy > roi_y2:
                continue

            boxes[tid] = (x1, y1, x2, y2)

            # ----------------------------------
            # 궤적 저장
            # ----------------------------------
            self.track_history.setdefault(tid, []).append((cx, cy))
            if len(self.track_history[tid]) > self.track_history_size:
                self.track_history[tid].pop(0)

            # ----------------------------------
            # 속도 계산
            # 최소 2개 좌표가 있어야 dy 계산 가능
            # ----------------------------------
            if len(self.track_history[tid]) >= 2:
                (_, yp) = self.track_history[tid][-2]
                (_, yc) = self.track_history[tid][-1]

                dy = yc - yp

                # ----------------------------------
                # ROI 내부 상대 위치 기반 scale
                # 가까운 차량/먼 차량의 픽셀 변화량 차이를
                # 조금 완화하기 위한 보정
                # ----------------------------------
                scale = (cy - roi_y1) / (roi_y2 - roi_y1 + 1e-6)

                speed = abs(dy) / (scale + 0.15)
                speed *= (1.2 + scale)

                # ----------------------------------
                # smoothing 적용
                # 현재 속도가 너무 튀지 않게 완만하게 만든다
                # ----------------------------------
                if tid in self.speed_memory:
                    speed = self.alpha * speed + (1 - self.alpha) * self.speed_memory[tid]

                self.speed_memory[tid] = speed
                speeds[tid] = speed

        return boxes, speeds, roi_y1, roi_y2

    # ==========================================
    # 3. 상태 판단
    # ==========================================
    def classify_state(self, avg_speed):
        """
        프레임 평균속도를 state_buffer에 누적한 뒤,
        장기 평균(avg_global)으로 상태를 판단한다.

        [중요]
        한 프레임 평균속도만 보면 흔들림이 커서
        NORMAL ↔ CONGESTION ↔ JAM 이 자꾸 튈 수 있다.
        그래서 state_buffer를 사용해 더 안정적으로 판정한다.
        """
        self.state_buffer.append(avg_speed)
        if len(self.state_buffer) > self.state_buffer_size:
            self.state_buffer.pop(0)

        avg_global = np.mean(self.state_buffer) if self.state_buffer else 0

        if avg_global < self.jam_enter:
            state = "JAM"
        elif avg_global < self.congestion_enter:
            state = "CONGESTION"
        else:
            state = "NORMAL"

        return state, avg_global

    # ==========================================
    # 4. 메인 update 함수
    # ==========================================
    def update(self, frame_idx, frame, tracks):
        """
        파이프라인에서 매 프레임마다 호출하는 함수

        입력:
            frame_idx : 현재 프레임 번호
            frame     : 현재 영상 프레임
            tracks    : 추적 결과 리스트

        반환:
            상태판단 결과 dict
        """
        # ----------------------------------
        # 1) 차량 bbox / 속도 계산
        # ----------------------------------
        boxes, speeds, roi_y1, roi_y2 = self.extract_boxes_and_speeds(frame, tracks)

        # ----------------------------------
        # 2) 현재 프레임 평균 속도
        # ----------------------------------
        avg_speed = np.mean(list(speeds.values())) if speeds else 0

        # ----------------------------------
        # 3) 장기 평균 기반 상태 판정
        # ----------------------------------
        state, avg_global = self.classify_state(avg_speed)

        # ----------------------------------
        # 4) 결과 반환
        # ----------------------------------
        result = {
            "frame_idx": frame_idx,
            "state": state,                     # NORMAL / CONGESTION / JAM
            "avg_speed": round(float(avg_speed), 2),   # 현재 프레임 평균속도
            "avg_global": round(float(avg_global), 2), # 버퍼 평균속도
            "vehicle_count": len(boxes),       # ROI 안 차량 수
            "boxes": boxes,                    # 시각화용 bbox
            "speeds": speeds,                  # 차량별 속도
            "roi_y1": roi_y1,                  # ROI 시작선
            "roi_y2": roi_y2                   # ROI 끝선
        }

        return result