# ==========================================
# 파일명: pipeline_core_V5_FINAL.py
# 용도: 상태로직 + 사고로직 통합 파이프라인 코어
# ==========================================
# [설명]
# 이 파일은
#   - traffic_state_V4.py
#   - traffic_accident_V4.py
# 를 하나로 묶어서,
# 한 프레임 단위로 최종 상태를 판단하는 파이프라인 코어이다.
#
# 흐름:
#   1) 상태로직 실행
#   2) 상태 결과를 바탕으로 사고 후보인지 먼저 거름
#   3) 사고 후보일 때만 사고로직 실행
#   4) 사고 결과를 HOLD 버퍼로 안정화
#   5) 최종 상태 반환
#
# 이렇게 하면:
#   - 모든 프레임마다 무조건 사고로직을 돌리지 않아도 되고
#   - 오탐도 줄이고
#   - 파이프라인 구조도 깔끔하게 유지할 수 있다.
# ==========================================

from traffic_state_V4 import TrafficState
from traffic_accident_V4 import AccidentDetector


class PipelineCore:
    def __init__(self):
        """
        파이프라인 코어 초기화

        [구성]
        - 상태 판단 모델 1개
        - 사고 판단 모델 1개
        - 사고 HOLD 버퍼 1개
        """
        self.state_model = TrafficState()
        self.accident_model = AccidentDetector()

        # ----------------------------------
        # 사고 최종 안정화를 위한 버퍼
        # 최근 60프레임 동안 사고로 판단된 비율을 본다.
        # ----------------------------------
        self.accident_window = []

    def process(self, frame_id, frame, tracks):
        """
        프레임 1장에 대해 최종 상태를 판단하는 함수

        입력:
            frame_id : 현재 프레임 번호
            frame    : 현재 영상 프레임
            tracks   : 추적 결과 리스트
                      예) [{"id":1, "bbox":[x1,y1,x2,y2]}, ...]

        반환:
            최종 파이프라인 결과 dict
        """

        # =====================================================
        # 1️⃣ 상태로직 실행
        # =====================================================
        # 상태로직은:
        # - ROI 안 차량만 사용
        # - 픽셀 속도 계산
        # - 평균 속도 기반으로
        #   NORMAL / CONGESTION / JAM 판단
        # 을 수행한다.
        # =====================================================
        state_result = self.state_model.update(frame_id, frame, tracks)

        state = state_result["state"]
        avg_speed = state_result["avg_speed"]
        vehicle_count = state_result["vehicle_count"]

        # -----------------------------------------------------
        # speed_std 계산
        # 현재 프레임 차량별 속도의 표준편차
        #
        # [왜 필요한가?]
        # 전체 평균속도만 보면
        # "한 대는 빠르고 한 대는 거의 멈춤" 같은
        # 사고 직전 비정상 패턴을 놓칠 수 있다.
        # 그래서 프레임 안 차량 속도 분산도 같이 본다.
        # -----------------------------------------------------
        speeds = list(state_result["speeds"].values())
        if len(speeds) >= 2:
            import numpy as np
            speed_std = float(np.std(speeds))
        else:
            speed_std = 0.0

        # -----------------------------------------------------
        # accident_hint 생성
        #
        # [의도]
        # 사고로직을 무조건 돌리면
        # 오탐도 늘고 계산도 무거워진다.
        # 그래서 상태로직 단계에서
        # "수상한 프레임인지" 먼저 약하게 체크한다.
        #
        # 현재 기준:
        # - 속도 편차가 큰가?
        # - 정체/혼잡 상태인가?
        #
        # 이 조건은 나중에 로그 보면서 튜닝 가능
        # -----------------------------------------------------
        accident_hint = (
            speed_std > 1.5
            or state in ["JAM", "CONGESTION"]
        )

        # =====================================================
        # 2️⃣ 사고 후보 판단
        # =====================================================
        # 상태로직에서 나온 약한 힌트를 바탕으로
        # 사고로직을 실제로 돌릴지 말지 결정한다.
        # =====================================================
        accident_candidate = (
            accident_hint
            and avg_speed > 0.8
            and vehicle_count >= 2
        )

        # =====================================================
        # 3️⃣ 사고로직 실행
        # =====================================================
        # 사고 후보일 때만 사고로직 실행
        # 아니면 False 처리
        # =====================================================
        if accident_candidate:
            accident_result = self.accident_model.update(frame_id, frame, tracks)
            accident_raw = accident_result["accident"]
        else:
            accident_result = None
            accident_raw = False

        # =====================================================
        # 4️⃣ 사고 HOLD 안정화
        # =====================================================
        # 최근 60프레임 중 사고 비율이 일정 수준 이상이면
        # 최종 사고로 확정한다.
        #
        # [왜 필요한가?]
        # 단일 프레임 사고 판단은 흔들릴 수 있기 때문
        # =====================================================
        self.accident_window.append(1 if accident_raw else 0)

        if len(self.accident_window) > 60:
            self.accident_window.pop(0)

        acc_ratio = sum(self.accident_window) / len(self.accident_window)

        final_accident = acc_ratio > 0.4

        # =====================================================
        # 5️⃣ 최종 상태 결정
        # =====================================================
        if final_accident:
            final_state = "ACCIDENT"
        else:
            final_state = state

        # =====================================================
        # 6️⃣ 최종 결과 반환
        # =====================================================
        return {
            # 최종 사용자용 상태
            "state": final_state,

            # 상태로직 원본 결과
            "raw_state": state,

            # 사고 최종 여부
            "accident": final_accident,

            # 상태 디버깅 정보
            "avg_speed": avg_speed,
            "speed_std": round(speed_std, 2),
            "vehicle_count": vehicle_count,

            # 사고 후보 단계 디버깅
            "accident_hint": accident_hint,
            "accident_candidate": accident_candidate,
            "acc_ratio": round(acc_ratio, 2),

            # 상태로직 세부 결과
            "state_result": state_result,

            # 사고로직 세부 결과
            # 사고 후보가 아니면 None
            "accident_result": accident_result
        }