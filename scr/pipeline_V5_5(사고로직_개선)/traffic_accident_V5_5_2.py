# ==========================================
# 파일명: traffic_accident_V5_5_2.py
# 설명:
# V5.5 사고 판단 로직 (최소 수정 버전)
# - 기존 V5.5 구조 유지
# - pair 후보 + frame 누적 + 사고 상태 lock 유지
# - ACCIDENT_WINDOW / ACCIDENT_CONFIRM_COUNT 유지 가능
# - 추가:
#   pair_accident_candidate의 짧은 반복(60프레임 내 2회)을
#   frame_accident_prediction의 보조 조건으로 사용
# - 목적:
#   strong 후보가 아니어도 초기 사고 징후가 짧은 구간에서 반복되면
#   frame_accident_prediction이 조금 더 일찍 뜨도록 보완
# ==========================================

from collections import deque
import numpy as np


class AccidentDetector:
    def __init__(self):
        # -----------------------------
        # pair 메모리
        # -----------------------------
        self.pair_memory = {}                 # (id1,id2) -> {"dist": ..., "gap": ...}
        self.pair_score = {}                  # pair별 사고 누적 점수
        self.impact_persist_counter = {}      # 근접/이상 지속 카운터

        # 반복성 메모리
        self.pair_candidate_frames = {}       # key -> [frame_id, ...]
        self.pair_last_candidate_frame = {}   # key -> 마지막 candidate frame

        # [추가] early repeat 메모리
        # pair_accident_candidate가 짧은 구간에 반복되는지 확인
        self.pair_early_candidate_frames = {}  # key -> [frame_id, ...]

        # frame 사고 예측 history
        self.frame_prediction_history = deque()

        # 사고 상태 lock
        self.accident_locked = False
        self.accident_start_frame = None

        # -----------------------------
        # 임계값
        # -----------------------------
        self.VERTICAL_X_THR = 30
        self.SAME_LANE_X_THR = 55

        self.DIST_DROP_RATIO = 0.6
        self.GAP_UP_RATIO = 1.2
        self.GAP_MIN_ABS = 3.0

        # gap 절대값 기준
        self.GAP_ABS_WEAK_THR = 4.0
        self.GAP_ABS_STRONG_THR = 6.0

        # impact persist 보조 임계값
        self.IMPACT_DIST_THR = 45
        self.IMPACT_PERSIST_HOLD = 3

        # pair 반복성 (기존)
        self.PAIR_REPEAT_WINDOW = 20
        self.PAIR_REPEAT_COUNT = 3
        self.PAIR_CONSEC_GAP = 2

        # [추가] early repeat 조건
        # pair_accident_candidate가 짧은 구간 안에서 2번 이상 반복되면
        # frame_accident_prediction 보조 신호로 사용
        self.EARLY_REPEAT_WINDOW = 60
        self.EARLY_REPEAT_COUNT = 2

        # pair 점수
        self.PAIR_SCORE_FRAME_THR = 3
        self.PAIR_SCORE_MAX = 10

        # frame 누적 사고 확정
        # 네가 지금 실험 중인 값 유지
        self.ACCIDENT_WINDOW = 200
        self.ACCIDENT_CONFIRM_COUNT = 3

        # 메모리 정리용
        self.STALE_FRAME_GAP = 60

        # 디버그 정보
        self.last_debug = {
            "frame_id": 0,
            "accident": False,
            "acc_ratio": 0.0,
            "frame_accident_prediction": False,
            "recent_prediction_count": 0,
            "accident_locked": False,
            "pairs": []
        }

    # =========================================================
    # 외부 수동 해제용
    # =========================================================
    def clear_accident(self):
        self.accident_locked = False
        self.accident_start_frame = None
        self.frame_prediction_history.clear()

    # =========================================================
    # 내부 유틸
    # =========================================================
    def _cleanup_stale_pairs(self, frame_id):
        stale_keys = []

        for key, last in self.pair_last_candidate_frame.items():
            if frame_id - last > self.STALE_FRAME_GAP:
                stale_keys.append(key)

        for key in stale_keys:
            self.pair_candidate_frames.pop(key, None)
            self.pair_last_candidate_frame.pop(key, None)
            self.pair_early_candidate_frames.pop(key, None)

    def _update_candidate_repeat(self, key, frame_id, pair_accident_candidate):
        """
        같은 pair의 candidate 반복성 관리
        반환:
            pair_repeat_candidate
            pair_consecutive_candidate
            repeat_count_window
        """
        self.pair_candidate_frames.setdefault(key, [])

        pair_repeat_candidate = False
        pair_consecutive_candidate = False

        if pair_accident_candidate:
            last_frame = self.pair_last_candidate_frame.get(key, None)

            if last_frame is not None and (frame_id - last_frame) <= self.PAIR_CONSEC_GAP:
                pair_consecutive_candidate = True

            self.pair_candidate_frames[key].append(frame_id)
            self.pair_last_candidate_frame[key] = frame_id

        recent_frames = [
            f for f in self.pair_candidate_frames.get(key, [])
            if frame_id - f <= self.PAIR_REPEAT_WINDOW
        ]
        self.pair_candidate_frames[key] = recent_frames

        repeat_count_window = len(recent_frames)
        pair_repeat_candidate = repeat_count_window >= self.PAIR_REPEAT_COUNT

        return pair_repeat_candidate, pair_consecutive_candidate, repeat_count_window

    def _update_early_repeat(self, key, frame_id, pair_accident_candidate):
        """
        [추가]
        pair_accident_candidate가 짧은 구간 안에서 반복되는지 확인
        - 용도: frame_accident_prediction 보조 조건
        - 강한 확정 신호가 아니라 '초기 반복 징후'만 보기 위한 것
        """
        self.pair_early_candidate_frames.setdefault(key, [])

        if pair_accident_candidate:
            self.pair_early_candidate_frames[key].append(frame_id)

        recent_frames = [
            f for f in self.pair_early_candidate_frames.get(key, [])
            if frame_id - f <= self.EARLY_REPEAT_WINDOW
        ]
        self.pair_early_candidate_frames[key] = recent_frames

        early_repeat_count_window = len(recent_frames)
        early_repeat_candidate = early_repeat_count_window >= self.EARLY_REPEAT_COUNT

        return early_repeat_candidate, early_repeat_count_window

    def _update_frame_prediction_history(self, frame_id, frame_accident_prediction):
        self.frame_prediction_history.append((frame_id, frame_accident_prediction))

        while self.frame_prediction_history and (frame_id - self.frame_prediction_history[0][0] > self.ACCIDENT_WINDOW):
            self.frame_prediction_history.popleft()

        recent_prediction_count = sum(
            1 for _, pred in self.frame_prediction_history if pred
        )

        return recent_prediction_count

    # =========================================================
    # 사고 판단
    # =========================================================
    def update(self, frame_id, tracks, analysis):
        """
        입력:
            frame_id : 현재 프레임 번호
            tracks   : [{"id": tid, "bbox": (...)}, ...]
            analysis : TrackAnalyzer + LaneTemplate 결과 dict

        출력:
            {
                "accident": bool,
                "acc_ratio": float,
                "frame_accident_prediction": bool,
                "accident_locked": bool
            }
        """

        self._cleanup_stale_pairs(frame_id)

        boxes = analysis.get("boxes", {})
        speeds = analysis.get("speeds", {})
        avg_speed = float(analysis.get("avg_speed", 0.0))
        lane_map = analysis.get("lane_map", {})
        smoke_fire_map = analysis.get("smoke_fire_map", {})

        ids = list(boxes.keys())

        pair_debug = []
        total_pairs = 0
        positive_pairs = 0

        frame_has_strong_candidate = False
        frame_has_repeat_strong_candidate = False
        frame_has_high_score_pair = False

        # [추가]
        # pair_accident_candidate의 짧은 반복으로 frame prediction 보조
        frame_has_early_repeat_candidate = False

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                total_pairs += 1

                box1 = boxes[id1]
                box2 = boxes[id2]

                cx1 = int((box1[0] + box1[2]) / 2)
                cy1 = int(box1[3])
                cx2 = int((box2[0] + box2[2]) / 2)
                cy2 = int(box2[3])

                dist = float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))

                s1 = float(speeds.get(id1, 0.0))
                s2 = float(speeds.get(id2, 0.0))
                gap = abs(s1 - s2)

                key = tuple(sorted((id1, id2)))

                prev = self.pair_memory.get(key, {
                    "dist": dist,
                    "gap": gap
                })

                # -----------------------------
                # 변화량
                # -----------------------------
                dist_drop = dist < prev["dist"] * self.DIST_DROP_RATIO
                gap_up = (gap > prev["gap"] * self.GAP_UP_RATIO) or (gap > self.GAP_MIN_ABS)

                # -----------------------------
                # 차선 / 배치 관계
                # -----------------------------
                lane1 = lane_map.get(id1, None)
                lane2 = lane_map.get(id2, None)

                same_lane = (lane1 is not None and lane2 is not None and lane1 == lane2)

                vertical = abs(cx1 - cx2) < self.VERTICAL_X_THR
                vertical_or_lane = vertical or (same_lane and abs(cx1 - cx2) < self.SAME_LANE_X_THR)

                lane_break = (same_lane and abs(cx1 - cx2) > self.SAME_LANE_X_THR)

                lane_break_acc = (
                    same_lane and
                    (dist < self.IMPACT_DIST_THR) and
                    (not vertical_or_lane)
                )

                # -----------------------------
                # impact persist
                # -----------------------------
                self.impact_persist_counter.setdefault(key, 0)

                if (dist < self.IMPACT_DIST_THR) and (gap_up or dist_drop):
                    self.impact_persist_counter[key] += 1
                else:
                    self.impact_persist_counter[key] = 0

                impact_persist = self.impact_persist_counter[key] >= self.IMPACT_PERSIST_HOLD

                # -----------------------------
                # 비정상 정지
                # -----------------------------
                abnormal = ((s1 < 2.0 or s2 < 2.0) and avg_speed > 3.0)
                abnormal_stop = abnormal and gap_up

                # -----------------------------
                # 핵심 패턴
                # -----------------------------
                rear_core = dist_drop and gap_up and (same_lane or vertical)

                gap_weak = gap >= self.GAP_ABS_WEAK_THR
                gap_strong = gap >= self.GAP_ABS_STRONG_THR

                persist_evidence = impact_persist and rear_core
                stop_evidence = abnormal_stop
                lane_break_evidence = lane_break_acc

                smoke_fire = False
                if isinstance(smoke_fire_map, dict):
                    smoke_fire = bool(smoke_fire_map.get(key, False))

                post_evidence = (
                    persist_evidence or
                    stop_evidence or
                    lane_break_evidence or
                    smoke_fire
                )

                abnormal_pose = lane_break_acc or abnormal_stop

                # -----------------------------
                # pair 후보
                # -----------------------------
                weak_pair_candidate = rear_core and gap_weak
                strong_pair_candidate = rear_core and post_evidence and gap_strong

                pair_accident_candidate = weak_pair_candidate or strong_pair_candidate

                # -----------------------------
                # pair 반복성 (기존)
                # -----------------------------
                pair_repeat_candidate, pair_consecutive_candidate, repeat_count_window = \
                    self._update_candidate_repeat(key, frame_id, pair_accident_candidate)

                repeat_strong_candidate = pair_repeat_candidate or pair_consecutive_candidate

                # -----------------------------
                # [추가] early repeat
                # pair_accident_candidate의 짧은 반복 여부
                # -----------------------------
                early_repeat_candidate, early_repeat_count_window = \
                    self._update_early_repeat(key, frame_id, pair_accident_candidate)

                # early repeat를 frame prediction 보조로 쓸 때는
                # 혼잡 오탐 방지를 위해 최소 관계 조건을 같이 본다.
                # - same_lane
                # - vertical
                # - gap 약한 기준 이상
                # 이 조합일 때만 early repeat를 인정
                early_repeat_valid = (
                    early_repeat_candidate
                    and same_lane
                    and vertical
                    and gap_weak
                )

                # -----------------------------
                # pair 점수
                # strong / repeat strong +2
                # weak                  +1
                # else                  +0
                # -----------------------------
                self.pair_score.setdefault(key, 0)

                if strong_pair_candidate or repeat_strong_candidate:
                    self.pair_score[key] += 2
                elif weak_pair_candidate:
                    self.pair_score[key] += 1
                else:
                    self.pair_score[key] += 0

                self.pair_score[key] = max(
                    0,
                    min(self.PAIR_SCORE_MAX, self.pair_score[key])
                )

                pair_high_score = self.pair_score[key] >= self.PAIR_SCORE_FRAME_THR

                if strong_pair_candidate:
                    frame_has_strong_candidate = True
                if repeat_strong_candidate:
                    frame_has_repeat_strong_candidate = True
                if pair_high_score:
                    frame_has_high_score_pair = True
                if early_repeat_valid:
                    frame_has_early_repeat_candidate = True

                if pair_high_score:
                    positive_pairs += 1

                # -----------------------------
                # pair 메모리 업데이트
                # -----------------------------
                self.pair_memory[key] = {
                    "dist": dist,
                    "gap": gap
                }

                pair_debug.append({
                    "pair": key,
                    "lane1": lane1,
                    "lane2": lane2,
                    "same_lane": same_lane,

                    "dist": round(dist, 2),
                    "gap": round(gap, 2),

                    "dist_drop": dist_drop,
                    "gap_up": gap_up,
                    "vertical": vertical,
                    "vertical_or_lane": vertical_or_lane,

                    "lane_break": lane_break,
                    "lane_break_acc": lane_break_acc,

                    "impact_persist": impact_persist,
                    "persist_evidence": persist_evidence,
                    "stop_evidence": stop_evidence,
                    "lane_break_evidence": lane_break_evidence,
                    "smoke_fire": smoke_fire,

                    "abnormal": abnormal,
                    "abnormal_stop": abnormal_stop,
                    "abnormal_pose": abnormal_pose,

                    "rear_core": rear_core,
                    "gap_weak": gap_weak,
                    "gap_strong": gap_strong,
                    "post_evidence": post_evidence,

                    "weak_pair_candidate": weak_pair_candidate,
                    "strong_pair_candidate": strong_pair_candidate,
                    "pair_accident_candidate": pair_accident_candidate,

                    "pair_repeat_candidate": pair_repeat_candidate,
                    "pair_consecutive_candidate": pair_consecutive_candidate,
                    "repeat_count_window": repeat_count_window,
                    "repeat_strong_candidate": repeat_strong_candidate,

                    # [추가]
                    "early_repeat_candidate": early_repeat_candidate,
                    "early_repeat_count_window": early_repeat_count_window,
                    "early_repeat_valid": early_repeat_valid,

                    "pair_score": self.pair_score[key],
                    "pair_high_score": pair_high_score
                })

        # =====================================================
        # frame 사고 예측
        # 기존 strong / repeat strong / high score 유지
        # + early repeat 보조 조건 추가
        # =====================================================
        frame_accident_prediction = (
            frame_has_strong_candidate
            or frame_has_repeat_strong_candidate
            or frame_has_high_score_pair
            or frame_has_early_repeat_candidate
        )

        recent_prediction_count = self._update_frame_prediction_history(
            frame_id, frame_accident_prediction
        )

        # =====================================================
        # 사고 확정 / 상태 lock
        # =====================================================
        if (not self.accident_locked) and (recent_prediction_count >= self.ACCIDENT_CONFIRM_COUNT):
            self.accident_locked = True
            self.accident_start_frame = frame_id

        accident_flag = self.accident_locked

        acc_ratio = (positive_pairs / total_pairs) if total_pairs > 0 else 0.0

        self.last_debug = {
            "frame_id": frame_id,
            "accident": accident_flag,
            "acc_ratio": round(acc_ratio, 4),
            "frame_accident_prediction": frame_accident_prediction,
            "recent_prediction_count": recent_prediction_count,
            "accident_locked": self.accident_locked,
            "accident_start_frame": self.accident_start_frame,
            "pairs": pair_debug
        }

        return {
            "accident": accident_flag,
            "acc_ratio": round(acc_ratio, 4),
            "frame_accident_prediction": frame_accident_prediction,
            "recent_prediction_count": recent_prediction_count,
            "accident_locked": self.accident_locked
        }

    def get_debug_info(self):
        return self.last_debug