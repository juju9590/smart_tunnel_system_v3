# ==========================================
# 파일명: traffic_accident_V5_4.py
# 설명:
# V5.4 사고 판단 로직 (IoU 완전 제거 버전)
# - 혼잡 오탐을 줄이기 위해 IoU 기반 판정 완전 제거
# - 후보(candidate)는 넓게, 확정(confirmed)은 시간 누적 기반으로 좁게
# - 후방추돌형 / 비정상 자세 / 충돌 후 지속 패턴 중심
# ==========================================

import numpy as np


class AccidentDetector:
    def __init__(self):
        # -----------------------------
        # pair 메모리
        # -----------------------------
        self.pair_memory = {}              # (id1,id2) -> {"dist": ..., "gap": ...}
        self.accident_counter = {}         # pair별 사고 누적 점수
        self.impact_persist_counter = {}   # 근접/이상 지속 카운터

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
        self.ACCIDENT_SCORE_THR = 6
        self.ACCIDENT_SCORE_MAX = 10

        # 디버그 정보
        self.last_debug = {
            "frame_id": 0,
            "accident": False,
            "acc_ratio": 0.0,
            "pairs": []
        }

    # =========================================================
    # 사고 판단
    # =========================================================
    def update(self, frame_id, tracks, analysis):
        """
        입력:
            frame_id : 현재 프레임 번호
            tracks   : [{"id": tid, "bbox": (...)}, ...]
            analysis : TrackAnalyzer 결과 dict

        출력:
            {
                "accident": bool,
                "acc_ratio": float
            }
        """

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
                # 중심/foot point
                # -----------------------------
                cx1 = int((box1[0] + box1[2]) / 2)
                cy1 = int(box1[3])   # foot point y
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
                # IoU 제거: x축 이탈 + 근접만 사용
                # -----------------------------
                lane_break_acc = (
                    same_lane and
                    (dist < self.IMPACT_DIST_THR) and
                    (not vertical_or_lane)
                )

                lane_break = (same_lane and abs(cx1 - cx2) > self.SAME_LANE_X_THR)

                # -----------------------------
                # impact persist
                # 단순 혼잡 밀집이 아니라
                # "접근/속도변화가 있는 근접 지속"만 본다
                # IoU 제거: dist만 사용
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
                # 핵심 사고 패턴
                # IoU 기반 side / hard_overlap 제거
                # -----------------------------
                rear = same_lane and dist_drop and gap_up and vertical_or_lane

                # 향후 확장용
                smoke_fire = False
                if isinstance(smoke_fire_map, dict):
                    smoke_fire = bool(smoke_fire_map.get(key, False))

                # flow_break는 현재 임시 추정 제거
                flow_break = False

                # impact persist도 충돌성 조건이 있을 때만 강화
                impact_persist_strong = impact_persist and (
                    rear or
                    lane_break_acc or
                    (vertical and dist_drop and gap_up)
                )

                # -----------------------------
                # [V5.4 핵심] 그룹 정의
                # -----------------------------
                # collision_sign:
                # - vertical 단독 금지
                # - dist_drop 또는 gap_up 단독도 금지
                collision_sign = rear or (vertical and dist_drop and gap_up)

                # abnormal_pose:
                # - abnormal 단독 금지
                abnormal_pose = lane_break_acc or (abnormal and same_lane and gap_up) or abnormal_stop

                # post_evidence:
                # - 충돌 후속 증거 중심
                post_evidence = (
                    impact_persist_strong or
                    smoke_fire
                )

                group_count = sum([
                    collision_sign,
                    abnormal_pose,
                    post_evidence
                ])

                # -----------------------------
                # [V5.4 핵심] 후보(candidate) 2단 구조
                # -----------------------------
                # A: 형상 기반 사고
                candidate_A = collision_sign and (abnormal_pose or post_evidence)

                # B: 시간 기반 사고
                # 충돌 형상이 약해도,
                # 비정상 자세 + 후속 현상이 같이 나오면 후보 허용
                candidate_B = abnormal_pose and post_evidence

                accident_candidate = candidate_A or candidate_B

                # -----------------------------
                # 누적 점수형 hold
                # 강한 후보는 +2
                # 약한 후보는 +1
                # 아니면 감소
                # -----------------------------
                self.accident_counter.setdefault(key, 0)

                if candidate_A:
                    self.accident_counter[key] += 2
                elif candidate_B:
                    self.accident_counter[key] += 1
                else:
                    self.accident_counter[key] -= 1

                # 하한/상한 제한
                self.accident_counter[key] = max(0, min(self.ACCIDENT_SCORE_MAX, self.accident_counter[key]))

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
                    "impact_persist_strong": impact_persist_strong,
                    "smoke_fire": smoke_fire,
                    "flow_break": flow_break,

                    "abnormal": abnormal,
                    "abnormal_stop": abnormal_stop,

                    "rear": rear,

                    "collision_sign": collision_sign,
                    "abnormal_pose": abnormal_pose,
                    "post_evidence": post_evidence,
                    "group_count": group_count,

                    "candidate_A": candidate_A,
                    "candidate_B": candidate_B,
                    "accident_candidate": accident_candidate,

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