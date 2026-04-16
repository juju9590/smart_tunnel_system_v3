# ==========================================
# 파일명: traffic_accident_V5_2.py
# 설명:
# V5_2_3 사고 판단 로직
# - 후보(candidate)는 넓게
# - 확정(confirmed)은 시간 누적 기반으로 좁게
# - 형상 기반 사고 + 시간 기반 사고를 함께 받는 구조
# ==========================================

import numpy as np


class AccidentDetector:
    def __init__(self):
        # -----------------------------
        # pair 메모리
        # -----------------------------
        self.pair_memory = {}              # (id1,id2) -> {"dist": ..., "gap": ...}
        self.accident_counter = {}         # pair별 사고 누적 점수
        self.iou_stop_counter = {}         # 큰 겹침 지속 카운터
        self.impact_persist_counter = {}   # 근접/겹침 지속 카운터

        # -----------------------------
        # 임계값
        # -----------------------------
        self.VERTICAL_X_THR = 30
        self.SAME_LANE_X_THR = 55

        self.DIST_DROP_RATIO = 0.6
        self.GAP_UP_RATIO = 1.2
        self.GAP_MIN_ABS = 3.0

        self.IOU_SIDE_THR = 0.30
        self.IOU_HARD_THR = 0.90

        self.STOP_IOU_HOLD = 3
        self.IMPACT_PERSIST_HOLD = 3

        # 누적 점수 기반 사고 확정
        self.ACCIDENT_SCORE_THR = 6
        self.ACCIDENT_SCORE_MAX = 10

        # impact persist 보조 임계값
        self.IMPACT_IOU_THR = 0.20
        self.IMPACT_DIST_THR = 45

        # 디버그 정보
        self.last_debug = {
            "frame_id": 0,
            "accident": False,
            "acc_ratio": 0.0,
            "pairs": []
        }

    # =========================================================
    # 기본 유틸
    # =========================================================
    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(1, (box1[2] - box1[0])) * max(1, (box1[3] - box1[1]))
        area2 = max(1, (box2[2] - box2[0])) * max(1, (box2[3] - box2[1]))
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

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

        # 향후 확장용. 현재 없으면 기본 False
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

                cx1 = int((box1[0] + box1[2]) / 2)
                cy1 = int(box1[3])   # foot point y
                cx2 = int((box2[0] + box2[2]) / 2)
                cy2 = int(box2[3])

                dist = float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))

                s1 = float(speeds.get(id1, 0.0))
                s2 = float(speeds.get(id2, 0.0))
                gap = abs(s1 - s2)

                iou = self.compute_iou(box1, box2)

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

                # 사고성 차선 붕괴
                lane_break_acc = (
                    (same_lane and not vertical_or_lane and iou > 0.10) or
                    (same_lane and dist < self.IMPACT_DIST_THR and not vertical_or_lane)
                )

                lane_break = (same_lane and abs(cx1 - cx2) > self.SAME_LANE_X_THR)

                # -----------------------------
                # 겹침 유지
                # -----------------------------
                self.iou_stop_counter.setdefault(key, 0)

                if iou > self.IOU_HARD_THR:
                    self.iou_stop_counter[key] += 1
                else:
                    self.iou_stop_counter[key] = 0

                stop_confirm = self.iou_stop_counter[key] > self.STOP_IOU_HOLD

                # -----------------------------
                # impact persist
                # 단순 혼잡 밀집 말고
                # "접근/속도변화가 있는 근접 지속"만 본다
                # -----------------------------
                self.impact_persist_counter.setdefault(key, 0)

                if ((iou > self.IMPACT_IOU_THR) or (dist < self.IMPACT_DIST_THR)) and (gap_up or dist_drop):
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
                # 기존 핵심 사고 패턴
                # -----------------------------
                rear = same_lane and dist_drop and gap_up and vertical_or_lane
                side = (iou > self.IOU_SIDE_THR) and gap_up
                hard_overlap = (iou > self.IOU_HARD_THR and stop_confirm)

                # -----------------------------
                # 확장 입력
                # -----------------------------
                smoke_fire = False
                if isinstance(smoke_fire_map, dict):
                    smoke_fire = bool(smoke_fire_map.get(key, False))

                # flow_break는 현재 임시 추정 제거
                flow_break = False

                # impact persist도 충돌성 조건이 있을 때만 강화
                impact_persist_strong = impact_persist and (
                    rear or side or hard_overlap or (vertical and dist_drop)
                )

                # -----------------------------
                # [V5_2_3 핵심] 그룹 정의
                # -----------------------------
                # collision_sign:
                # - vertical 단독 금지
                collision_sign = rear or side or hard_overlap or (vertical and (dist_drop or gap_up))

                # abnormal_pose:
                # - abnormal 단독 금지
                abnormal_pose = lane_break_acc or (abnormal and same_lane and gap_up) or abnormal_stop

                # post_evidence:
                # - 충돌 후속 증거 중심
                post_evidence = (
                    ((stop_confirm and (rear or side or hard_overlap)) or
                     impact_persist_strong or
                     smoke_fire)
                )

                group_count = sum([
                    collision_sign,
                    abnormal_pose,
                    post_evidence
                ])

                # -----------------------------
                # [V5_2_3 핵심] 후보(candidate) 2단 구조
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
                    "iou": round(iou, 3),

                    "dist_drop": dist_drop,
                    "gap_up": gap_up,
                    "vertical": vertical,
                    "vertical_or_lane": vertical_or_lane,

                    "lane_break": lane_break,
                    "lane_break_acc": lane_break_acc,

                    "stop_confirm": stop_confirm,
                    "impact_persist": impact_persist,
                    "impact_persist_strong": impact_persist_strong,
                    "smoke_fire": smoke_fire,
                    "flow_break": flow_break,

                    "abnormal": abnormal,
                    "abnormal_stop": abnormal_stop,

                    "rear": rear,
                    "side": side,
                    "hard_overlap": hard_overlap,

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
    