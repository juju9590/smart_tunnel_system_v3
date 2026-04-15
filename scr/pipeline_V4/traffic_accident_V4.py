# ==========================================
# traffic_accident_V4.py
# V3 로직 → 파이프라인용 최적화
# ==========================================

import numpy as np


class AccidentDetector:
    def __init__(self):
        self.track_history = {}
        self.pair_memory = {}
        self.accident_counter = {}
        self.iou_stop_counter = {}

    # ==========================
    # IOU 계산
    # ==========================
    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

        union = area1 + area2 - inter
        return inter/union if union>0 else 0

    # ==========================
    # 차선 이탈
    # ==========================
    def get_lane_break(self, track):
        if len(track) < 10:
            return 0
        return abs(track[-1][0] - track[0][0])

    # ==========================
    # 메인
    # ==========================
    def update(self, frame_id, tracks):

        boxes = {}
        speeds = {}

        # -----------------------------
        # 1️⃣ 위치 + 속도
        # -----------------------------
        for t in tracks:
            tid = t["id"]
            x1, y1, x2, y2 = t["bbox"]

            cx = int((x1 + x2) / 2)
            cy = y2

            boxes[tid] = (x1, y1, x2, y2)

            self.track_history.setdefault(tid, []).append((cx, cy))
            if len(self.track_history[tid]) > 30:
                self.track_history[tid].pop(0)

            speed = 0
            if len(self.track_history[tid]) >= 2:
                dy = self.track_history[tid][-1][1] - self.track_history[tid][-2][1]
                speed = abs(dy)

            speeds[tid] = speed

        # -----------------------------
        # 🔥 전체 평균 속도 (V3 핵심)
        # -----------------------------
        if len(speeds) > 0:
            avg_speed = np.mean(list(speeds.values()))
        else:
            avg_speed = 0

        # -----------------------------
        # 2️⃣ 차량 간 분석
        # -----------------------------
        accident_flag = False
        ids = list(boxes.keys())

        for i in range(len(ids)):
            for j in range(i+1, len(ids)):

                id1, id2 = ids[i], ids[j]

                box1 = boxes[id1]
                box2 = boxes[id2]

                cx1 = int((box1[0]+box1[2])/2)
                cy1 = box1[3]
                cx2 = int((box2[0]+box2[2])/2)
                cy2 = box2[3]

                dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)

                s1 = speeds.get(id1,0)
                s2 = speeds.get(id2,0)
                gap = abs(s1 - s2)

                iou = self.compute_iou(box1, box2)

                # -----------------------------
                # 이전 상태
                # -----------------------------
                key = f"{id1}-{id2}"

                prev = self.pair_memory.get(key, {
                    "dist": dist,
                    "gap": gap
                })

                dist_drop = dist < prev["dist"] * 0.6
                gap_up = gap > prev["gap"] * 1.2 or gap > 3

                # -----------------------------
                # 위치 관계
                # -----------------------------
                vertical = abs(cx1 - cx2) < 30

                # -----------------------------
                # 차선 이탈
                # -----------------------------
                lane_break = (
                    self.get_lane_break(self.track_history.get(id1,[])) > 50 or
                    self.get_lane_break(self.track_history.get(id2,[])) > 50
                )

                # -----------------------------
                # 🔥 IoU 정지 (V3 핵심)
                # -----------------------------
                self.iou_stop_counter.setdefault(key, 0)

                if iou > 0.9:
                    self.iou_stop_counter[key] += 1
                else:
                    self.iou_stop_counter[key] = 0

                stop_confirm = self.iou_stop_counter[key] > 3

                # -----------------------------
                # 🔥 이상 차량 (V3 핵심)
                # -----------------------------
                abnormal = (
                    (s1 < 2 or s2 < 2) and avg_speed > 3
                )

                # -----------------------------
                # 🚨 사고 판단 (V4 핵심)
                # -----------------------------
                rear = dist_drop and gap_up and vertical
                side = (iou > 0.3) and gap_up
                lane_break_acc = lane_break and gap_up

                accident = (
                    rear or
                    side or
                    (iou > 0.9 and stop_confirm) or
                    lane_break_acc or
                    (abnormal and gap_up)
                )

                # -----------------------------
                # HOLD 안정화
                # -----------------------------
                self.accident_counter.setdefault(key, 0)

                if accident:
                    self.accident_counter[key] += 1
                else:
                    self.accident_counter[key] = 0

                final = self.accident_counter[key] > 4

                if final:
                    accident_flag = True

                # -----------------------------
                # 메모리 업데이트
                # -----------------------------
                self.pair_memory[key] = {
                    "dist": dist,
                    "gap": gap
                }

        return accident_flag