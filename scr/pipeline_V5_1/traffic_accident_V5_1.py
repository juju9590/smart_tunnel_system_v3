# ==========================================
# 파일명: traffic_accident_V5_1.py
# 설명:
# V5 사고 로직 + V5_1 차선 추정 로직 적용 버전
#
# [핵심 변경점]
# 1) 기존의 x 이동량 기반 lane_break 제거
# 2) 차량 궤적을 x=f(y) 형태로 피팅
# 3) 피팅 계수들을 자동 군집화하여 차선(centerline) 추정
# 4) 각 차량을 가장 가까운 centerline에 할당
# 5) lane assignment는 최근 5프레임 동일할 때만 확정
#
# [호환성]
# - update(frame_id, tracks) -> accident_flag (bool) 반환 유지
# - tracks 형식은 기존과 동일:
#   [{"id": 1, "bbox": [x1, y1, x2, y2]}, ...]
# ==========================================

import numpy as np
from collections import defaultdict, deque


class AccidentDetector:
    def __init__(self):
        # -----------------------------
        # 기존 V5 메모리
        # -----------------------------
        self.track_history = {}          # tid -> [(cx, cy), ...]
        self.pair_memory = {}            # (id1,id2) -> {"dist": ..., "gap": ...}
        self.accident_counter = {}       # pair hold counter
        self.iou_stop_counter = {}       # pair iou stop counter
        self.prev_speeds = {}            # tid -> smoothed speed

        # -----------------------------
        # V5_1 차선 추정용 메모리
        # -----------------------------
        self.track_models = {}           # tid -> fitted trajectory model
        self.track_fit_error = {}        # tid -> fit rmse
        self.track_stable_motion = {}    # tid -> stable motion bool

        self.centerlines = []            # [{"cluster_id":0, "model_type":"linear", ...}, ...]
        self.current_lane_raw = {}       # tid -> raw nearest lane id
        self.current_lane_stable = {}    # tid -> confirmed lane id
        self.lane_vote_history = defaultdict(lambda: deque(maxlen=5))   # tid -> last 5 raw lane ids
        self.lane_change_memory = defaultdict(lambda: deque(maxlen=10)) # tid -> recent stable lanes

        # 디버깅/가시화용
        self.last_debug = {
            "avg_speed": 0,
            "lane_count": 0,
            "centerlines": [],
            "lanes": {}
        }

        # -----------------------------
        # 파라미터
        # -----------------------------
        self.frame_height = 720
        self.ROI_Y1_RATIO = 0.30
        self.ROI_Y2_RATIO = 0.80

        self.MAX_HISTORY = 60
        self.FIT_MIN_POINTS = 30               # 최근 30프레임 이상
        self.LANE_CONFIRM_FRAMES = 5           # 5프레임 이상 동일 lane일 때 확정
        self.MIN_TOTAL_MOTION = 35             # 너무 안 움직인 차량 제외
        self.MAX_DY_JUMP = 40
        self.MAX_DX_JUMP = 60

        self.MAX_SPEED = 20
        self.SPEED_JUMP_LIMIT = 10

        # 군집화 threshold
        self.LINEAR_CLUSTER_THR = 0.085
        self.QUAD_CLUSTER_THR = 0.12

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

    def _roi_bounds(self):
        roi_y1 = int(self.frame_height * self.ROI_Y1_RATIO)
        roi_y2 = int(self.frame_height * self.ROI_Y2_RATIO)
        return roi_y1, roi_y2

    def _clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    # =========================================================
    # 속도 계산 (기존 V5 유지)
    # =========================================================
    def _update_tracks_and_speeds(self, tracks):
        boxes = {}
        speeds = {}

        roi_y1, roi_y2 = self._roi_bounds()

        for t in tracks:
            tid = t["id"]
            x1, y1, x2, y2 = t["bbox"]

            cx = int((x1 + x2) / 2)
            cy = int(y2)   # 하단 발점 기준

            # track history 저장
            self.track_history.setdefault(tid, []).append((cx, cy))
            if len(self.track_history[tid]) > self.MAX_HISTORY:
                self.track_history[tid].pop(0)

            # ROI 밖이면 속도는 이전값 유지
            if cy < roi_y1 or cy > roi_y2:
                speeds[tid] = self.prev_speeds.get(tid, 0)
                continue

            boxes[tid] = (x1, y1, x2, y2)

            # 초기 보호
            if len(self.track_history[tid]) < 3:
                speed = self.prev_speeds.get(tid, 0)
                self.prev_speeds[tid] = speed
                speeds[tid] = speed
                continue

            xp, yp = self.track_history[tid][-2]
            xc, yc = self.track_history[tid][-1]

            dx = abs(xc - xp)
            dy = yc - yp

            # 비정상 점프 제거
            if abs(dy) > self.MAX_DY_JUMP or dx > self.MAX_DX_JUMP:
                speed = self.prev_speeds.get(tid, 0)
                speeds[tid] = speed
                continue

            # 원근 보정
            scale = (yc - roi_y1) / (roi_y2 - roi_y1 + 1e-6)
            scale = self._clamp(scale, 0.1, 1.0)

            speed = abs(dy) / (scale + 0.15)
            speed *= (1.2 + scale)

            prev_speed = self.prev_speeds.get(tid, speed)

            if speed > self.MAX_SPEED:
                speed = prev_speed

            if abs(speed - prev_speed) > self.SPEED_JUMP_LIMIT:
                speed = prev_speed

            # smoothing
            speed = 0.3 * speed + 0.7 * prev_speed
            speed = self._clamp(speed, 0, self.MAX_SPEED)

            self.prev_speeds[tid] = speed
            speeds[tid] = speed

        valid = [s for s in speeds.values() if s < self.MAX_SPEED]
        avg_speed = float(np.mean(valid)) if len(valid) > 0 else 0.0

        return boxes, speeds, avg_speed

    # =========================================================
    # V5_1 차선 추정용: 궤적 안정성 검사
    # =========================================================
    def _is_stable_moving_track(self, pts):
        """
        최근 30프레임 이상 + 안정적으로 움직인 차량만 사용
        """
        if len(pts) < self.FIT_MIN_POINTS:
            return False

        recent = pts[-self.FIT_MIN_POINTS:]

        total_motion = 0.0
        jump_bad = 0
        ys = []
        xs = []

        for i in range(1, len(recent)):
            x0, y0 = recent[i - 1]
            x1, y1 = recent[i]
            dx = x1 - x0
            dy = y1 - y0
            total_motion += np.sqrt(dx * dx + dy * dy)

            if abs(dy) > self.MAX_DY_JUMP or abs(dx) > self.MAX_DX_JUMP:
                jump_bad += 1

            xs.append(x1)
            ys.append(y1)

        # y 변화 범위가 너무 작으면 차선 피팅 가치 낮음
        y_span = (max(ys) - min(ys)) if ys else 0

        if total_motion < self.MIN_TOTAL_MOTION:
            return False
        if jump_bad > 3:
            return False
        if y_span < 25:
            return False

        return True

    # =========================================================
    # V5_1 차선 추정용: trajectory fitting
    # x = a*y + b 또는 x = a*y^2 + b*y + c
    # =========================================================
    def _fit_trajectory_model(self, pts):
        """
        최근 30프레임 기준으로 linear / quadratic 둘 다 피팅 후
        더 안정적인 쪽 선택
        """
        recent = pts[-self.FIT_MIN_POINTS:]
        ys = np.array([p[1] for p in recent], dtype=np.float32)
        xs = np.array([p[0] for p in recent], dtype=np.float32)

        # y 값이 너무 동일하면 피팅 불가
        if len(np.unique(ys)) < 6:
            return None, 1e9

        # linear fit: x = a*y + b
        try:
            lin_coef = np.polyfit(ys, xs, 1)  # [a, b]
            lin_pred = np.polyval(lin_coef, ys)
            lin_rmse = float(np.sqrt(np.mean((xs - lin_pred) ** 2)))
        except Exception:
            lin_coef = None
            lin_rmse = 1e9

        # quadratic fit: x = a*y^2 + b*y + c
        try:
            quad_coef = np.polyfit(ys, xs, 2)  # [a, b, c]
            quad_pred = np.polyval(quad_coef, ys)
            quad_rmse = float(np.sqrt(np.mean((xs - quad_pred) ** 2)))
        except Exception:
            quad_coef = None
            quad_rmse = 1e9

        # 기본은 linear
        if lin_coef is None and quad_coef is None:
            return None, 1e9

        # quadratic이 linear보다 충분히 좋을 때만 사용
        # 너무 과적합되는 걸 막기 위한 조건
        if quad_coef is not None and lin_coef is not None:
            improve_ratio = (lin_rmse - quad_rmse) / max(lin_rmse, 1e-6)
            if improve_ratio > 0.18 and abs(quad_coef[0]) > 1e-5:
                model = {
                    "type": "quadratic",
                    "coef": quad_coef.astype(float).tolist()
                }
                return model, quad_rmse
            else:
                model = {
                    "type": "linear",
                    "coef": lin_coef.astype(float).tolist()
                }
                return model, lin_rmse

        if lin_coef is not None:
            return {"type": "linear", "coef": lin_coef.astype(float).tolist()}, lin_rmse

        return {"type": "quadratic", "coef": quad_coef.astype(float).tolist()}, quad_rmse

    def _predict_x(self, model, y):
        if model is None:
            return None
        if model["type"] == "linear":
            a, b = model["coef"]
            return a * y + b
        else:
            a, b, c = model["coef"]
            return a * (y ** 2) + b * y + c

    # =========================================================
    # V5_1 차선 추정용: 계수 벡터화 / 거리
    # =========================================================
    def _coef_vector(self, model):
        """
        군집화를 위해 서로 다른 model type도 비교 가능하게 정규화
        """
        if model["type"] == "linear":
            a, b = model["coef"]
            # quadratic 형태로 맞춰줌: [0, a, b_norm]
            return np.array([0.0, a, b / max(self.frame_height, 1)], dtype=np.float32)

        a, b, c = model["coef"]
        return np.array([
            a * self.frame_height,   # 너무 작아서 scale up
            b,
            c / max(self.frame_height, 1)
        ], dtype=np.float32)

    def _model_distance(self, model1, model2):
        """
        단순 계수 거리 + ROI 구간 예측 위치 차이 혼합
        """
        v1 = self._coef_vector(model1)
        v2 = self._coef_vector(model2)
        coef_dist = float(np.linalg.norm(v1 - v2))

        roi_y1, roi_y2 = self._roi_bounds()
        sample_ys = np.linspace(roi_y1, roi_y2, 5)

        diffs = []
        for y in sample_ys:
            x1 = self._predict_x(model1, y)
            x2 = self._predict_x(model2, y)
            diffs.append(abs(x1 - x2))

        shape_dist = float(np.mean(diffs)) / max(self.frame_height, 1)
        return coef_dist * 0.5 + shape_dist * 0.5

    # =========================================================
    # V5_1 차선 추정용: 자동 군집화
    # =========================================================
    def _cluster_track_models(self, stable_models):
        """
        stable_models: [(tid, model), ...]
        군집 수 자동 생성
        간단한 greedy clustering 사용
        """
        if not stable_models:
            return []

        clusters = []

        for tid, model in stable_models:
            assigned = False
            thr = self.QUAD_CLUSTER_THR if model["type"] == "quadratic" else self.LINEAR_CLUSTER_THR

            for cluster in clusters:
                dist = self._model_distance(model, cluster["rep_model"])
                if dist < thr:
                    cluster["items"].append((tid, model))
                    # 대표 모델 갱신
                    cluster["rep_model"] = self._aggregate_models([m for _, m in cluster["items"]])
                    assigned = True
                    break

            if not assigned:
                clusters.append({
                    "cluster_id": len(clusters),
                    "rep_model": model,
                    "items": [(tid, model)]
                })

        # 너무 작은 군집 제거 (노이즈)
        filtered = []
        for c in clusters:
            if len(c["items"]) >= 1:
                filtered.append(c)

        # centerline 생성 + 좌우 정렬
        centerlines = []
        roi_y1, roi_y2 = self._roi_bounds()
        y_mid = (roi_y1 + roi_y2) / 2.0

        for c in filtered:
            agg_model = self._aggregate_models([m for _, m in c["items"]])

            centerlines.append({
                "cluster_id": c["cluster_id"],
                "model_type": agg_model["type"],
                "coef": agg_model["coef"],
                "member_ids": [tid for tid, _ in c["items"]],
                "x_mid": self._predict_x(agg_model, y_mid)
            })

        centerlines.sort(key=lambda x: x["x_mid"])

        # 정렬 후 lane id 재부여
        for idx, cl in enumerate(centerlines):
            cl["lane_id"] = idx

        return centerlines

    def _aggregate_models(self, models):
        """
        군집 중심 모델 생성
        type은 다수결, coef는 중앙값
        """
        if not models:
            return None

        linear_models = [m for m in models if m["type"] == "linear"]
        quad_models = [m for m in models if m["type"] == "quadratic"]

        if len(quad_models) > len(linear_models):
            arr = np.array([m["coef"] for m in quad_models], dtype=np.float32)
            coef = np.median(arr, axis=0).tolist()
            return {"type": "quadratic", "coef": coef}
        else:
            arr = np.array([m["coef"] for m in linear_models], dtype=np.float32)
            if len(arr) == 0:
                # fallback
                arr = np.array([m["coef"] for m in models], dtype=np.float32)
                coef = np.median(arr, axis=0).tolist()
                if len(coef) == 2:
                    return {"type": "linear", "coef": coef}
                return {"type": "quadratic", "coef": coef}

            coef = np.median(arr, axis=0).tolist()
            return {"type": "linear", "coef": coef}

    # =========================================================
    # V5_1 차선 추정용: 각 차량을 가장 가까운 centerline에 할당
    # =========================================================
    def _assign_lane_raw(self, tid, current_point):
        """
        raw lane assignment
        가장 가까운 centerline으로 할당
        """
        if not self.centerlines:
            return None

        x, y = current_point
        best_lane = None
        best_dist = 1e9

        for cl in self.centerlines:
            model = {
                "type": cl["model_type"],
                "coef": cl["coef"]
            }
            cx = self._predict_x(model, y)
            dist = abs(x - cx)

            if dist < best_dist:
                best_dist = dist
                best_lane = cl["lane_id"]

        # 너무 멀면 lane 미확정 처리
        if best_dist > 80:
            return None

        return best_lane

    def _update_lane_confirmation(self, tid, raw_lane):
        """
        lane assignment는 5프레임 이상 동일할 때만 확정
        """
        self.lane_vote_history[tid].append(raw_lane)

        if len(self.lane_vote_history[tid]) < self.LANE_CONFIRM_FRAMES:
            return self.current_lane_stable.get(tid, None)

        last_votes = list(self.lane_vote_history[tid])

        # 최근 5프레임 모두 같아야 확정
        if all(v == last_votes[0] and v is not None for v in last_votes):
            new_lane = last_votes[0]
            old_lane = self.current_lane_stable.get(tid, None)

            self.current_lane_stable[tid] = new_lane
            self.lane_change_memory[tid].append(new_lane)

            return new_lane

        return self.current_lane_stable.get(tid, None)

    def _recent_lane_changed(self, tid):
        """
        최근 stable lane 이 변경되었는지 확인
        """
        hist = list(self.lane_change_memory.get(tid, []))
        hist = [h for h in hist if h is not None]

        if len(hist) < 2:
            return False

        return len(set(hist[-3:])) >= 2

    def _update_lane_estimation(self, tracks):
        """
        전체 차선 추정 갱신
        """
        # 1) 안정적으로 움직인 차량만 피팅
        stable_models = []

        for t in tracks:
            tid = t["id"]
            pts = self.track_history.get(tid, [])

            stable = self._is_stable_moving_track(pts)
            self.track_stable_motion[tid] = stable

            if not stable:
                continue

            model, rmse = self._fit_trajectory_model(pts)
            if model is None:
                continue

            self.track_models[tid] = model
            self.track_fit_error[tid] = rmse
            stable_models.append((tid, model))

        # 2) 군집 수 자동 생성
        self.centerlines = self._cluster_track_models(stable_models)

        # 3) 각 차량을 가장 가까운 centerline에 할당
        lane_map = {}
        for t in tracks:
            tid = t["id"]
            pts = self.track_history.get(tid, [])
            if len(pts) == 0:
                lane_map[tid] = None
                continue

            current_point = pts[-1]
            raw_lane = self._assign_lane_raw(tid, current_point)
            self.current_lane_raw[tid] = raw_lane

            stable_lane = self._update_lane_confirmation(tid, raw_lane)
            lane_map[tid] = stable_lane

        return lane_map

    # =========================================================
    # 사고 판단
    # =========================================================
    def update(self, frame_id, tracks):
        """
        입력:
            frame_id: 현재 프레임 번호
            tracks: [{"id": tid, "bbox": [x1,y1,x2,y2]}, ...]

        출력:
            accident_flag: bool
        """

        # -----------------------------
        # 1) 속도 / 박스 / 평균속도 갱신
        # -----------------------------
        boxes, speeds, avg_speed = self._update_tracks_and_speeds(tracks)

        # -----------------------------
        # 2) 차선 추정 갱신 (V5_1 핵심)
        # -----------------------------
        lane_map = self._update_lane_estimation(tracks)

        # -----------------------------
        # 3) 차량 간 분석
        # -----------------------------
        accident_flag = False
        ids = list(boxes.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]

                box1 = boxes[id1]
                box2 = boxes[id2]

                cx1 = int((box1[0] + box1[2]) / 2)
                cy1 = int(box1[3])
                cx2 = int((box2[0] + box2[2]) / 2)
                cy2 = int(box2[3])

                dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

                s1 = speeds.get(id1, 0)
                s2 = speeds.get(id2, 0)
                gap = abs(s1 - s2)

                iou = self.compute_iou(box1, box2)

                # pair key 안정화
                key = tuple(sorted((id1, id2)))

                prev = self.pair_memory.get(key, {
                    "dist": dist,
                    "gap": gap
                })

                dist_drop = dist < prev["dist"] * 0.6
                gap_up = gap > prev["gap"] * 1.2 or gap > 3

                # -----------------------------
                # 위치 / 차선 관계
                # -----------------------------
                lane1 = lane_map.get(id1, None)
                lane2 = lane_map.get(id2, None)

                same_lane = (lane1 is not None and lane2 is not None and lane1 == lane2)

                # 기존 vertical 완화: 같은 차선이면 x 차이를 조금 더 허용
                vertical = abs(cx1 - cx2) < 30
                vertical_or_lane = vertical or (same_lane and abs(cx1 - cx2) < 55)

                # 최근 stable lane 변경 여부
                lane_change_like = self._recent_lane_changed(id1) or self._recent_lane_changed(id2)

                # IoU 정지
                self.iou_stop_counter.setdefault(key, 0)
                if iou > 0.9:
                    self.iou_stop_counter[key] += 1
                else:
                    self.iou_stop_counter[key] = 0

                stop_confirm = self.iou_stop_counter[key] > 3

                # 이상 차량
                abnormal = ((s1 < 2 or s2 < 2) and avg_speed > 3)

                # -----------------------------
                # 사고 판단
                # -----------------------------
                # 1) 후방추돌형: 같은 차선 + 거리 급감 + 속도차 증가
                rear = same_lane and dist_drop and gap_up and vertical_or_lane

                # 2) 측면/접촉형: IoU + 속도차
                side = (iou > 0.3) and gap_up

                # 3) 차선 변경/비정상 횡이동 중 충돌 의심
                lane_break_acc = lane_change_like and (gap_up or iou > 0.2)

                # 4) 매우 큰 겹침이 수프레임 지속
                hard_overlap = (iou > 0.9 and stop_confirm)

                # 5) 전체 평균은 흐르는데 일부만 비정상 정지 + 간격/속도 이상
                abnormal_stop = abnormal and gap_up

                accident = (
                    rear or
                    side or
                    hard_overlap or
                    lane_break_acc or
                    abnormal_stop
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

        # -----------------------------
        # 4) 디버깅 정보 저장
        # -----------------------------
        self.last_debug = {
            "avg_speed": round(avg_speed, 2),
            "lane_count": len(self.centerlines),
            "centerlines": self.centerlines,
            "lanes": dict(lane_map),
            "raw_lanes": dict(self.current_lane_raw),
        }

        return accident_flag

    # =========================================================
    # 디버깅용 getter
    # =========================================================
    def get_debug_info(self):
        return self.last_debug

    def get_lane_of_track(self, tid):
        return self.current_lane_stable.get(tid, None)

    def get_centerlines(self):
        return self.centerlines