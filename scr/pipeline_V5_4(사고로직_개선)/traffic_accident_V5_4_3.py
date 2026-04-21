# ==========================================
# 파일명: traffic_accident_V5_4_3.py
# 설명:
# V5.4-3 사고 판단 로직
# - IoU 완전 제거 유지
# - 일반 사고형(115134, 114618) 튜닝 버전
# - 사고 핵심 특징: rear_core
#   = dist_drop + gap_up + (same_lane or vertical)
# - 보조 특징: post_evidence
#   = persist_evidence / abnormal_stop / lane_break_acc / smoke_fire
# - 같은 pair의 candidate 반복성을 반영
#   * 최근 20프레임 내 candidate 3회 이상
#   * 2프레임 gap까지 연속으로 인정
# - 점수 임계값 완화
#   * ACCIDENT_SCORE_THR = 4
#   * 비후보 시 감점하지 않고 0 유지
# ==========================================

import numpy as np


class AccidentDetector:
    def __init__(self):
        # -----------------------------
        # pair 메모리
        # -----------------------------
        self.pair_memory = {}                 # (id1,id2) -> {"dist": ..., "gap": ...}
        self.accident_counter = {}            # pair별 사고 누적 점수
        self.impact_persist_counter = {}      # 근접/이상 지속 카운터

        # 반복성 메모리
        self.pair_candidate_frames = {}       # key -> [frame_id, frame_id, ...]
        self.pair_last_candidate_frame = {}   # key -> 마지막 candidate frame

        # -----------------------------
        # 임계값
        # -----------------------------
        self.VERTICAL_X_THR = 30
        self.SAME_LANE_X_THR = 55

        self.DIST_DROP_RATIO = 0.6
        self.GAP_UP_RATIO = 1.2
        self.GAP_MIN_ABS = 3.0

        # impact persist 보조 임계값
        self.IMPACT_DIST_THR = 45
        self.IMPACT_PERSIST_HOLD = 3

        # 누적 점수 기반 사고 확정
        self.ACCIDENT_SCORE_THR = 4
        self.ACCIDENT_SCORE_MAX = 10

        # 반복성 파라미터
        self.CANDIDATE_REPEAT_WINDOW = 20     # 최근 20프레임
        self.CANDIDATE_REPEAT_COUNT = 3       # 3회 이상
        self.CANDIDATE_CONSEC_GAP = 2         # 2프레임 gap까지 연속 인정

        # 메모리 정리용
        self.STALE_FRAME_GAP = 60

        # 디버그 정보
        self.last_debug = {
            "frame_id": 0,
            "accident": False,
            "acc_ratio": 0.0,
            "pairs": []
        }

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

    def _update_candidate_repeat(self, key, frame_id, accident_candidate):
        """
        같은 pair의 candidate 반복성 관리
        반환:
            pair_repeat_candidate: 최근 window 내 candidate 3회 이상
            pair_consecutive_candidate: 직전 candidate와의 frame gap이 허용 범위 이내
            repeat_count_window: 최근 window 내 candidate 횟수
        """
        self.pair_candidate_frames.setdefault(key, [])

        pair_repeat_candidate = False
        pair_consecutive_candidate = False

        if accident_candidate:
            last_frame = self.pair_last_candidate_frame.get(key, None)

            if last_frame is not None and (frame_id - last_frame) <= self.CANDIDATE_CONSEC_GAP:
                pair_consecutive_candidate = True

            self.pair_candidate_frames[key].append(frame_id)
            self.pair_last_candidate_frame[key] = frame_id

        # 최근 window 안의 frame만 유지
        recent_frames = [
            f for f in self.pair_candidate_frames.get(key, [])
            if frame_id - f <= self.CANDIDATE_REPEAT_WINDOW
        ]
        self.pair_candidate_frames[key] = recent_frames

        repeat_count_window = len(recent_frames)
        pair_repeat_candidate = repeat_count_window >= self.CANDIDATE_REPEAT_COUNT

        return pair_repeat_candidate, pair_consecutive_candidate, repeat_count_window

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
                "acc_ratio": float
            }
        """

        self._cleanup_stale_pairs(frame_id)

        boxes = analysis.get("boxes", {})
        speeds = analysis.get("speeds", {})
        avg_speed = float(analysis.get("avg_speed", 0.0))
        lane_map = analysis.get("lane_map", {})

        # 향후 확장용
        smoke_fire_map = analysis.get("smoke_fire_map", {})

        ids = list(boxes.keys())

        accident_flag = False
        pair_debug = []
        positive_pairs = 0
        total_pairs = 0

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                total_pairs += 1

                box1 = boxes[id1]
                box2 = boxes[id2]

                # -----------------------------
                # 중심 / foot point
                # -----------------------------
                cx1 = int((box1[0] + box1[2]) / 2)
                cy1 = int(box1[3])
                cx2 = int((box2[0] + box2[2]) / 2)
                cy2 = int(box2[3])

                # pair 거리
                dist = float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))

                # 속도 / 속도차
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

                # 세로 정렬 여부
                vertical = abs(cx1 - cx2) < self.VERTICAL_X_THR

                # 같은 차선이면 x 허용범위 약간 완화
                vertical_or_lane = vertical or (same_lane and abs(cx1 - cx2) < self.SAME_LANE_X_THR)

                # -----------------------------
                # 사고성 차선 붕괴
                # lane 품질이 있을 때만 보조 증거로 사용
                # -----------------------------
                lane_break = (same_lane and abs(cx1 - cx2) > self.SAME_LANE_X_THR)

                lane_break_acc = (
                    same_lane and
                    (dist < self.IMPACT_DIST_THR) and
                    (not vertical_or_lane)
                )

                # -----------------------------
                # impact persist
                # "급접근/속도변화가 있는 근접 지속"만 본다
                # -----------------------------
                self.impact_persist_counter.setdefault(key, 0)

                if (dist < self.IMPACT_DIST_THR) and (gap_up or dist_drop):
                    self.impact_persist_counter[key] += 1
                else:
                    self.impact_persist_counter[key] = 0

                impact_persist = self.impact_persist_counter[key] >= self.IMPACT_PERSIST_HOLD

                # -----------------------------
                # 비정상 정지
                # 전체 평균은 흐르는데 일부만 거의 정지
                # -----------------------------
                abnormal = ((s1 < 2.0 or s2 < 2.0) and avg_speed > 3.0)
                abnormal_stop = abnormal and gap_up

                # -----------------------------
                # [V5.4-3 핵심] rear_core
                # 사고 핵심 특징:
                # dist_drop + gap_up + (same_lane or vertical)
                # -----------------------------
                rear_core = dist_drop and gap_up and (same_lane or vertical)

                # -----------------------------
                # 보조 증거
                # -----------------------------
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

                # 참고용
                abnormal_pose = lane_break_acc or abnormal_stop

                # -----------------------------
                # 기본 candidate
                # -----------------------------
                weak_candidate = rear_core
                strong_candidate = rear_core and post_evidence

                # 기본 사고 후보
                accident_candidate = weak_candidate or strong_candidate

                # -----------------------------
                # 반복성(candidate repeat)
                # -----------------------------
                pair_repeat_candidate, pair_consecutive_candidate, repeat_count_window = \
                    self._update_candidate_repeat(key, frame_id, accident_candidate)

                repeat_strong_candidate = pair_repeat_candidate or pair_consecutive_candidate

                # -----------------------------
                # 점수 누적
                # strong/repeat_strong +2
                # weak                   +1
                # else                   +0 (감점 없음)
                # -----------------------------
                self.accident_counter.setdefault(key, 0)

                if strong_candidate or repeat_strong_candidate:
                    self.accident_counter[key] += 2
                elif weak_candidate:
                    self.accident_counter[key] += 1
                else:
                    self.accident_counter[key] += 0

                self.accident_counter[key] = max(
                    0,
                    min(self.ACCIDENT_SCORE_MAX, self.accident_counter[key])
                )

                confirmed = self.accident_counter[key] >= self.ACCIDENT_SCORE_THR

                if confirmed:
                    accident_flag = True
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
                    "post_evidence": post_evidence,

                    "weak_candidate": weak_candidate,
                    "strong_candidate": strong_candidate,
                    "accident_candidate": accident_candidate,

                    "pair_repeat_candidate": pair_repeat_candidate,
                    "pair_consecutive_candidate": pair_consecutive_candidate,
                    "repeat_count_window": repeat_count_window,
                    "repeat_strong_candidate": repeat_strong_candidate,

                    "accident_score": self.accident_counter[key],
                    "confirmed": confirmed
                })

        acc_ratio = (positive_pairs / total_pairs) if total_pairs > 0 else 0.0

        self.last_debug = {
            "frame_id": frame_id,
            "accident": accident_flag,
            "acc_ratio": round(acc_ratio, 4),
            "pairs": pair_debug
        }

        return {
            "accident": accident_flag,
            "acc_ratio": round(acc_ratio, 4)
        }

    def get_debug_info(self):
        return self.last_debug