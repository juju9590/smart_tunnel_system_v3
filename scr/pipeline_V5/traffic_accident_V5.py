# ==========================================
# traffic_accident_V5.py (STABLE FIX)
# ==========================================

import numpy as np


class AccidentDetector:
    def __init__(self):
        self.track_history = {}
        self.pair_memory = {}
        self.accident_counter = {}
        self.iou_stop_counter = {}
        self.prev_speeds = {}

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
        return inter/union if union > 0 else 0

    # ==========================
    # 차선 이탈 (x 이동량)
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
        frame_height = 720

        ROI_Y1 = int(frame_height * 0.3)
        ROI_Y2 = int(frame_height * 0.8)

        # -----------------------------
        # 1️⃣ 위치 + 속도 계산
        # -----------------------------
        for t in tracks:
            tid = t["id"]
            x1, y1, x2, y2 = t["bbox"]

            cx = int((x1 + x2) / 2)
            cy = y2

            # ROI 밖 → 이전 속도 유지 (🔥 중요)
            if cy < ROI_Y1 or cy > ROI_Y2:
                speeds[tid] = self.prev_speeds.get(tid, 0)
                continue

            boxes[tid] = (x1, y1, x2, y2)

            self.track_history.setdefault(tid, []).append((cx, cy))
            if len(self.track_history[tid]) > 30:
                self.track_history[tid].pop(0)

            # 초기 보호
            if len(self.track_history[tid]) < 3:
                speed = self.prev_speeds.get(tid, 0)
                self.prev_speeds[tid] = speed
                speeds[tid] = speed
                continue

            # 🔥 좌표 수정 (핵심)
            xp, yp = self.track_history[tid][-2]
            xc, yc = self.track_history[tid][-1]

            dx = abs(xc - xp)
            dy = yc - yp

            # 이동 이상 제거
            if abs(dy) > 40 or dx > 60:
                speed = self.prev_speeds.get(tid, 0)
                speeds[tid] = speed
                continue

            # 원근 보정
            scale = (yc - ROI_Y1) / (ROI_Y2 - ROI_Y1 + 1e-6)
            scale = max(0.1, min(scale, 1.0))

            speed = abs(dy) / (scale + 0.15)
            speed *= (1.2 + scale)

            prev_speed = self.prev_speeds.get(tid, speed)

            # 필터
            if speed > 20:
                speed = prev_speed

            if abs(speed - prev_speed) > 10:
                speed = prev_speed

            # smoothing
            speed = 0.3 * speed + 0.7 * prev_speed

            speed = max(0, min(speed, 20))

            self.prev_speeds[tid] = speed
            speeds[tid] = speed

        # -----------------------------
        # 평균 속도 (안정화)
        # -----------------------------
        valid = [s for s in speeds.values() if s < 20]

        if len(valid) == 0:
            avg_speed = 0
        else:
            avg_speed = np.mean(valid)

        # -----------------------------
        # 2️⃣ 차량 간 분석
        # -----------------------------
        accident_flag = False
        ids = list(boxes.keys())

        for i in range(len(ids)):
            for j in range(i+1, len(ids)):

                id1, id2 = ids[i], ids[j]

                box1, box2 = boxes[id1], boxes[id2]

                cx1 = int((box1[0]+box1[2])/2)
                cy1 = box1[3]
                cx2 = int((box2[0]+box2[2])/2)
                cy2 = box2[3]

                dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)

                s1 = speeds.get(id1, 0)
                s2 = speeds.get(id2, 0)
                gap = abs(s1 - s2)

                iou = self.compute_iou(box1, box2)

                # 🔥 key 안정화 (순서 고정)
                key = tuple(sorted((id1, id2)))

                prev = self.pair_memory.get(key, {
                    "dist": dist,
                    "gap": gap
                })

                dist_drop = dist < prev["dist"] * 0.6
                gap_up = gap > prev["gap"] * 1.2 or gap > 3

                # 위치 관계
                vertical = abs(cx1 - cx2) < 30

                # 차선 이탈
                lane_break = (
                    self.get_lane_break(self.track_history.get(id1, [])) > 50 or
                    self.get_lane_break(self.track_history.get(id2, [])) > 50
                )

                # IoU 정지
                self.iou_stop_counter.setdefault(key, 0)

                if iou > 0.9:
                    self.iou_stop_counter[key] += 1
                else:
                    self.iou_stop_counter[key] = 0

                stop_confirm = self.iou_stop_counter[key] > 3

                # 이상 차량
                abnormal = (
                    (s1 < 2 or s2 < 2) and avg_speed > 3
                )

                # -----------------------------
                # 🚨 사고 판단
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
                    self.accident_counter[key] = max(0, self.accident_counter[key] - 1)

                if self.accident_counter[key] > 4:
                    accident_flag = True

                # 메모리 업데이트
                self.pair_memory[key] = {
                    "dist": dist,
                    "gap": gap
                }

        return accident_flag