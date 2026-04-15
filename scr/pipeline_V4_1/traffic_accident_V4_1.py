# ==========================================
# 파일명: traffic_accident_V4_1.py
# 용도: 사고판단 전용 로직 (상태판단 속도와 분리)
# 핵심:
#   1) 차량쌍 거리 급감
#   2) IoU 기반 정지 유지
#   3) 흐름 붕괴(flow break)
#   4) 전체 흐름 vs 특정 차량 이상(abnormal)
#   5) fragment / confidence 보조증거
# ==========================================

import cv2
import numpy as np


class AccidentDetector:
    def __init__(
        self,
        lane_count=2,
        iou_collision_th=0.30,       # 기본 측면 충돌 후보
        iou_stop_th=0.90,            # 거의 완전 겹침
        iou_stop_frames=10,          # 일정 프레임 유지
        dist_drop_norm_th=0.12,      # 거리 급감 정규화 임계값
        gap_up_th=0.08,              # 접근도 증가 기준
        flow_break_th=0.10,          # 블록 점유율 편차
        abnormal_global_th=0.05,     # 전체 흐름은 살아있는 기준
        abnormal_local_th=0.015,     # 특정 차량은 거의 멈춤
        fragment_th=12.0,            # 프레임 차이 노이즈 기준
        conf_std_th=0.01,            # confidence 고정성
        memory_size=20
    ):
        self.lane_count = lane_count
        self.iou_collision_th = iou_collision_th
        self.iou_stop_th = iou_stop_th
        self.iou_stop_frames = iou_stop_frames
        self.dist_drop_norm_th = dist_drop_norm_th
        self.gap_up_th = gap_up_th
        self.flow_break_th = flow_break_th
        self.abnormal_global_th = abnormal_global_th
        self.abnormal_local_th = abnormal_local_th
        self.fragment_th = fragment_th
        self.conf_std_th = conf_std_th
        self.memory_size = memory_size

        # 차량별 이동 이력
        self.track_history = {}

        # 차량쌍 이전 상태
        self.pair_memory = {}

        # IoU 정지 유지 카운트
        self.iou_stop_count = {}

        # 사고 HOLD
        self.accident_hold = {}
        self.hold_frames = 20

        # 이전 프레임
        self.prev_frame = None

        # confidence history
        self.conf_stack = {}

    # ==========================================
    # 유틸
    # ==========================================
    def _calc_iou(self, b1, b2):
        x11, y11, x12, y12 = b1
        x21, y21, x22, y22 = b2

        xx1 = max(x11, x21)
        yy1 = max(y11, y21)
        xx2 = min(x12, x22)
        yy2 = min(y12, y22)

        inter_w = max(0, xx2 - xx1)
        inter_h = max(0, yy2 - yy1)
        inter = inter_w * inter_h

        a1 = max(1, (x12 - x11) * (y12 - y11))
        a2 = max(1, (x22 - x21) * (y22 - y21))
        union = a1 + a2 - inter + 1e-6

        return inter / union

    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _bbox_bottom_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, y2

    def _bbox_size(self, bbox):
        x1, y1, x2, y2 = bbox
        return max(1, x2 - x1), max(1, y2 - y1)

    def _lane_id(self, bbox, frame_w):
        cx, _ = self._bbox_center(bbox)
        lane_w = frame_w / max(1, self.lane_count)
        return int(cx // lane_w)

    def _update_track_motion(self, tid, bbox):
        cx, cy = self._bbox_bottom_center(bbox)

        self.track_history.setdefault(tid, []).append((cx, cy))
        if len(self.track_history[tid]) > self.memory_size:
            self.track_history[tid].pop(0)

    def _get_track_motion(self, tid, bbox):
        """
        사고용 로컬 이동량
        상태판단 speed와 분리해서,
        bbox 높이로 정규화한 아주 단순한 '움직임 크기' 사용
        """
        _, h = self._bbox_size(bbox)

        hist = self.track_history.get(tid, [])
        if len(hist) < 2:
            return 0.0

        x1, y1 = hist[-2]
        x2, y2 = hist[-1]
        disp = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return disp / (h + 1e-6)

    def _frame_fragment(self, frame):
        if frame is None:
            return False, 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0.0

        diff = cv2.absdiff(self.prev_frame, gray)
        noise = float(np.mean(diff))
        self.prev_frame = gray

        return noise > self.fragment_th, noise

    def _calc_flow_break(self, tracks, frame_h):
        """
        화면 상/하 점유 불균형
        사고 시 뒤는 쌓이고 앞은 비는 패턴 반영
        """
        if len(tracks) == 0:
            return False, 0.0

        upper = 0
        lower = 0

        for obj in tracks:
            _, y1, _, y2 = obj["bbox"]
            cy = (y1 + y2) / 2.0
            if cy < frame_h / 2:
                upper += 1
            else:
                lower += 1

        total = max(1, upper + lower)
        occupancy_diff = abs(upper - lower) / total

        return occupancy_diff > self.flow_break_th, occupancy_diff

    def _conf_static(self, tid, conf):
        if conf is None:
            return False, 0.0

        self.conf_stack.setdefault(tid, []).append(conf)
        if len(self.conf_stack[tid]) > 10:
            self.conf_stack[tid].pop(0)

        if len(self.conf_stack[tid]) < 5:
            return False, 0.0

        std = float(np.std(self.conf_stack[tid]))
        return std < self.conf_std_th, std

    # ==========================================
    # 메인 업데이트
    # ==========================================
    def update(self, frame_idx, tracks, frame=None):
        """
        입력:
            frame_idx : 현재 프레임 번호
            tracks    : [{"id":1, "bbox":[x1,y1,x2,y2], "conf":0.88}, ...]
            frame     : 선택사항 (fragment 분석용)

        반환:
            사고 분석 결과 dict
        """

        # ----------------------------------
        # 차량별 motion 갱신
        # ----------------------------------
        for obj in tracks:
            tid = int(obj["id"])
            bbox = obj["bbox"]
            self._update_track_motion(tid, bbox)

        frame_h = frame.shape[0] if frame is not None else 720
        frame_w = frame.shape[1] if frame is not None else 1280

        # 전체 평균 로컬 움직임
        local_speeds = {}
        conf_static_map = {}

        for obj in tracks:
            tid = int(obj["id"])
            bbox = obj["bbox"]
            conf = obj.get("conf", None)

            s = self._get_track_motion(tid, bbox)
            local_speeds[tid] = s

            conf_static, conf_std = self._conf_static(tid, conf)
            conf_static_map[tid] = {
                "conf_static": conf_static,
                "conf_std": conf_std
            }

        avg_speed = float(np.mean(list(local_speeds.values()))) if local_speeds else 0.0

        # 보조 증거들
        fragment, fragment_noise = self._frame_fragment(frame)
        flow_break, occupancy_diff = self._calc_flow_break(tracks, frame_h)

        accident = False
        accident_pairs = []
        debug_rows = []

        # ----------------------------------
        # 차량쌍 비교
        # ----------------------------------
        n = len(tracks)

        for i in range(n):
            for j in range(i + 1, n):
                obj1 = tracks[i]
                obj2 = tracks[j]

                id1 = int(obj1["id"])
                id2 = int(obj2["id"])
                b1 = obj1["bbox"]
                b2 = obj2["bbox"]

                key = tuple(sorted([id1, id2]))

                c1x, c1y = self._bbox_bottom_center(b1)
                c2x, c2y = self._bbox_bottom_center(b2)

                w1, h1 = self._bbox_size(b1)
                w2, h2 = self._bbox_size(b2)
                avg_h = (h1 + h2) / 2.0
                avg_w = (w1 + w2) / 2.0

                dist = float(np.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2))
                iou = self._calc_iou(b1, b2)

                lane1 = self._lane_id(b1, frame_w)
                lane2 = self._lane_id(b2, frame_w)

                same_lane = (lane1 == lane2)
                lane_break = abs(lane1 - lane2) >= 1

                # 앞뒤 정렬 비슷한지
                vertical = abs(c1x - c2x) < (avg_w * 0.7)

                # 차량별 로컬 움직임
                s1 = local_speeds.get(id1, 0.0)
                s2 = local_speeds.get(id2, 0.0)

                # 이전 메모리
                if key not in self.pair_memory:
                    self.pair_memory[key] = {
                        "dist": dist,
                        "gap": abs(s1 - s2),
                    }

                prev = self.pair_memory[key]

                # ------------------------------
                # 핵심 pair 지표
                # ------------------------------
                dist_drop = prev["dist"] - dist
                dist_drop_norm = dist_drop / (avg_h + 1e-6)

                gap = abs(s1 - s2)
                gap_up = (gap - prev["gap"]) > self.gap_up_th

                # 추돌 방향성(간단)
                rear = same_lane and vertical and (dist_drop_norm > self.dist_drop_norm_th) and gap_up
                side = (iou > self.iou_collision_th) and gap_up
                lane_break_acc = lane_break and gap_up

                # ------------------------------
                # IoU 기반 정지 유지
                # ------------------------------
                iou_stop = iou > self.iou_stop_th
                self.iou_stop_count.setdefault(key, 0)

                if iou_stop:
                    self.iou_stop_count[key] += 1
                else:
                    self.iou_stop_count[key] = 0

                stop_confirm = self.iou_stop_count[key] >= self.iou_stop_frames

                # ------------------------------
                # 전체 흐름 vs 특정 차량 이상
                # 전체는 움직이는데 특정 쌍만 거의 멈춤
                # ------------------------------
                pair_slow = (s1 < self.abnormal_local_th or s2 < self.abnormal_local_th)
                abnormal = pair_slow and (avg_speed > self.abnormal_global_th)

                # ------------------------------
                # confidence 기반 정지
                # ------------------------------
                conf_static_1 = conf_static_map[id1]["conf_static"]
                conf_static_2 = conf_static_map[id2]["conf_static"]
                conf_static = conf_static_1 or conf_static_2

                # ------------------------------
                # 최종 판단
                # 기본 충돌 후보 + 보조증거 1개 이상
                # ------------------------------
                base_collision = rear or side or lane_break_acc
                support_count = sum([
                    stop_confirm,
                    flow_break,
                    abnormal,
                    fragment,
                    conf_static
                ])

                pair_accident = base_collision and (support_count >= 1)

                # HOLD 안정화
                self.accident_hold.setdefault(key, 0)
                if pair_accident:
                    self.accident_hold[key] = self.hold_frames
                elif self.accident_hold[key] > 0:
                    self.accident_hold[key] -= 1

                final_pair_accident = pair_accident or (self.accident_hold[key] > 0)

                if final_pair_accident:
                    accident = True
                    accident_pairs.append(key)

                debug_rows.append({
                    "frame_idx": frame_idx,
                    "id1": id1,
                    "id2": id2,
                    "lane1": lane1,
                    "lane2": lane2,

                    "s1": round(s1, 4),
                    "s2": round(s2, 4),
                    "avg_speed": round(avg_speed, 4),

                    "dist": round(dist, 2),
                    "prev_dist": round(prev["dist"], 2),
                    "dist_drop": round(dist_drop, 4),
                    "dist_drop_norm": round(dist_drop_norm, 4),

                    "gap": round(gap, 4),
                    "prev_gap": round(prev["gap"], 4),
                    "gap_up": gap_up,

                    "iou": round(iou, 4),
                    "iou_stop": iou_stop,
                    "stop_confirm": stop_confirm,

                    "rear": rear,
                    "side": side,
                    "lane_break_acc": lane_break_acc,

                    "flow_break": flow_break,
                    "occupancy_diff": round(occupancy_diff, 4),

                    "abnormal": abnormal,
                    "fragment": fragment,
                    "fragment_noise": round(fragment_noise, 4),

                    "conf_static": conf_static,
                    "support_count": support_count,

                    "pair_accident": pair_accident,
                    "final_accident": final_pair_accident,
                })

                # pair memory 업데이트
                self.pair_memory[key] = {
                    "dist": dist,
                    "gap": gap,
                }

        result = {
            "frame_idx": frame_idx,
            "accident": accident,
            "accident_pairs": accident_pairs,
            "avg_speed": round(avg_speed, 4),
            "flow_break": flow_break,
            "occupancy_diff": round(occupancy_diff, 4),
            "fragment": fragment,
            "fragment_noise": round(fragment_noise, 4),
            "local_speeds": {k: round(v, 4) for k, v in local_speeds.items()},
            "debug_rows": debug_rows,
        }

        return result