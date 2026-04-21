# ==========================================
# 파일명: traffic_accident_V5_1.py
# 설명:
# V5_1 사고 판단 로직
# - TrackAnalyzer가 계산한 공통 분석 결과 사용
# - 차선 추정은 여기서 하지 않음
# - 차량 간 거리, 속도차, IOU, 같은 차선 여부로 사고 판단
# ==========================================

import numpy as np


class AccidentDetector:
    def __init__(self):
        # -----------------------------
        # pair 메모리
        # -----------------------------
        self.pair_memory = {}          # (id1,id2) -> {"dist": ..., "gap": ...}
        self.accident_counter = {}     # 사고 hold 카운터
        self.iou_stop_counter = {}     # 큰 겹침 지속 카운터

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
        self.ACCIDENT_HOLD = 4

        # 디버그 정보
        self.last_debug = {
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
                cy1 = int(box1[3])
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
                # 비정상 정지
                # 전체 평균은 흐르는데 일부만 거의 정지
                # -----------------------------
                abnormal = ((s1 < 2.0 or s2 < 2.0) and avg_speed > 3.0)

                # -----------------------------
                # 사고 후보 조건
                # -----------------------------
                # 1) 후방추돌형
                rear = same_lane and dist_drop and gap_up and vertical_or_lane

                # 2) 측면 접촉형
                side = (iou > self.IOU_SIDE_THR) and gap_up

                # 3) 매우 큰 겹침이 수프레임 지속
                hard_overlap = (iou > self.IOU_HARD_THR and stop_confirm)

                # 4) 평균은 흐르는데 일부만 비정상 정지 + 속도차
                abnormal_stop = abnormal and gap_up

                accident_now = (
                    rear or
                    side or
                    hard_overlap or
                    abnormal_stop
                )

                # -----------------------------
                # HOLD 안정화
                # -----------------------------
                self.accident_counter.setdefault(key, 0)

                if accident_now:
                    self.accident_counter[key] += 1
                else:
                    self.accident_counter[key] = max(0, self.accident_counter[key] - 1)

                confirmed = self.accident_counter[key] > self.ACCIDENT_HOLD

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
                    "rear": rear,
                    "side": side,
                    "hard_overlap": hard_overlap,
                    "abnormal_stop": abnormal_stop,
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