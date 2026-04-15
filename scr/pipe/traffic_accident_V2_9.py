# ==========================================
# traffic_accident_V2_9.py (PIPELINE VERSION)
# V2.9 사고 로직 → 파이프라인용 변환
# ==========================================

import numpy as np

class AccidentDetector:
    def __init__(self):
        # 차량 궤적
        self.track_history = {}

        # 차량 쌍 관계 기억 (거리/속도 변화)
        self.pair_memory = {}

        # 사고 HOLD 카운터
        self.accident_counter = {}

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
    # 메인 업데이트
    # ==========================
    def update(self, frame_id, tracks):

        boxes = {}
        speeds = {}

        # -----------------------------
        # 1️⃣ 차량 위치 + 속도 계산
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

            # 속도 계산
            speed = 0
            if len(self.track_history[tid]) >= 2:
                dy = self.track_history[tid][-1][1] - self.track_history[tid][-2][1]
                speed = abs(dy)

            speeds[tid] = speed

        # -----------------------------
        # 2️⃣ 차량 간 사고 판단
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

                # 👉 이전 프레임 정보
                key = f"{id1}-{id2}"
                prev = self.pair_memory.get(key, {
                    "dist": dist,
                    "gap": gap
                })

                dist_drop = dist < prev["dist"] * 0.6
                gap_up = gap > prev["gap"] * 1.5

                # 👉 차선 이탈
                lane_break = (
                    self.get_lane_break(self.track_history.get(id1,[])) > 50 or
                    self.get_lane_break(self.track_history.get(id2,[])) > 50
                )

                # -----------------------------
                # 🚨 사고 조건 (V2.9 유지)
                # -----------------------------
                after_slow = (s1 < 2 and s2 < 2)
                vertical = abs(cx1 - cx2) < 30 and abs(cy1 - cy2) < 80
                rear = dist_drop and gap_up and vertical and after_slow
                side = (iou > 0.3) and gap_up
                lane_break_acc = lane_break and gap_up

                accident = rear or side or lane_break_acc

                # -----------------------------
                # HOLD 안정화
                # -----------------------------
                self.accident_counter.setdefault(key, 0)

                if accident:
                    self.accident_counter[key] += 1
                else:
                    self.accident_counter[key] = 0

                hold = self.accident_counter[key]

                # if accident:
                if hold > 3.5:
                    accident_flag = True             

                print(f"[DEBUG] frame:{frame_id} accident:{accident}")

                # 메모리 업데이트
                self.pair_memory[key] = {
                    "dist": dist,
                    "gap": gap
                }

        return accident_flag